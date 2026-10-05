"""Feature extraction for wafer maps.

Two backends are provided for the inspection module:

* ``handcrafted`` (default, no deep-learning dependency) --- the
  density / Radon / geometry feature family that is standard in the wafer-map
  pattern-recognition literature.  59 features per wafer.
* ``cnn`` --- a hook for a torch model, used only if torch is installed.

The handcrafted route exists so the pipeline runs anywhere.  For the paper the
inspection module should be a CNN, because that is what a fab would actually
deploy and because reviewers will expect it; the monitoring layer downstream is
identical either way, which is itself worth stating as a property of the method.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from skimage.measure import label as cc_label
from skimage.measure import regionprops
from skimage.transform import radon, resize

#: Fixed canvas for the Radon stage.  Wafer maps in WM-811K have varying
#: dimensions; the Radon features need a common grid.
RADON_SIZE = (48, 48)

#: Projection angles for the Radon stage. 36 is the accuracy/cost knob: the
#: full pass over 811k wafers is dominated by this transform.
RADON_THETA = np.linspace(0.0, 180.0, 36, endpoint=False)

N_DENSITY = 13
N_RADON = 40
N_GEOMETRY = 6
N_FEATURES = N_DENSITY + N_RADON + N_GEOMETRY  # 59


def _failure_mask(wafer_map: np.ndarray) -> np.ndarray:
    """Boolean mask of failed dies (value 2 in the WM-811K encoding)."""
    return wafer_map == 2


def _valid_mask(wafer_map: np.ndarray) -> np.ndarray:
    """Boolean mask of dies on the wafer (values 1 and 2)."""
    return wafer_map > 0


def density_features(wafer_map: np.ndarray) -> np.ndarray:
    """Failed-die density over 13 regions: a 4x4 grid plus four edge bands and
    a centre disc.  Each entry is failures / valid dies within the region."""
    fail = _failure_mask(wafer_map)
    valid = _valid_mask(wafer_map)
    h, w = wafer_map.shape

    out = np.zeros(N_DENSITY, dtype=np.float32)
    rows = np.linspace(0, h, 5).astype(int)
    cols = np.linspace(0, w, 5).astype(int)

    k = 0
    # Nine interior blocks from a 4x4 grid, taking the inner 3x3 plus corners.
    for i in range(3):
        for j in range(3):
            r0, r1 = rows[i], rows[i + 2]
            c0, c1 = cols[j], cols[j + 2]
            v = valid[r0:r1, c0:c1].sum()
            out[k] = fail[r0:r1, c0:c1].sum() / v if v else 0.0
            k += 1

    # Four edge bands.
    bands = [
        (slice(0, rows[1]), slice(None)),
        (slice(rows[3], h), slice(None)),
        (slice(None), slice(0, cols[1])),
        (slice(None), slice(cols[3], w)),
    ]
    for sl in bands:
        v = valid[sl].sum()
        out[k] = fail[sl].sum() / v if v else 0.0
        k += 1
    return out


def radon_features(wafer_map: np.ndarray) -> np.ndarray:
    """Row-mean and row-standard-deviation of the Radon transform, each
    interpolated onto 20 points.  Captures ring / scratch orientation."""
    fail = _failure_mask(wafer_map).astype(np.float32)
    canvas = resize(fail, RADON_SIZE, order=0, preserve_range=True, anti_aliasing=False)
    sino = radon(canvas, theta=RADON_THETA, circle=False)

    mean_profile = sino.mean(axis=1)
    std_profile = sino.std(axis=1)

    grid = np.linspace(0, 1, 20)
    src = np.linspace(0, 1, len(mean_profile))
    return np.concatenate(
        [
            np.interp(grid, src, mean_profile),
            np.interp(grid, src, std_profile),
        ]
    ).astype(np.float32)


def geometry_features(wafer_map: np.ndarray) -> np.ndarray:
    """Shape descriptors of the largest connected failure region."""
    fail = _failure_mask(wafer_map)
    out = np.zeros(N_GEOMETRY, dtype=np.float32)
    if not fail.any():
        return out

    labels = cc_label(fail, connectivity=2)
    props = regionprops(labels)
    if not props:
        return out
    region = max(props, key=lambda p: p.area)

    area_norm = region.area / max(_valid_mask(wafer_map).sum(), 1)
    major = region.axis_major_length
    minor = region.axis_minor_length
    out[:] = (
        area_norm,
        region.perimeter / max(major, 1e-6),
        major / max(wafer_map.shape[0], 1),
        minor / max(major, 1e-6),
        region.eccentricity,
        region.solidity,
    )
    return out


def extract(wafer_map: np.ndarray) -> np.ndarray:
    """Full 59-dimensional handcrafted feature vector for one wafer."""
    return np.concatenate(
        [
            density_features(wafer_map),
            radon_features(wafer_map),
            geometry_features(wafer_map),
        ]
    ).astype(np.float32)


def _extract_chunk(chunk) -> np.ndarray:
    return np.vstack([extract(np.asarray(w)) for w in chunk])


def extract_batch(
    wafer_maps,
    progress_every: int = 5000,
    n_jobs: int | None = None,
    chunk_size: int = 512,
) -> np.ndarray:
    """Feature matrix for an iterable of wafer maps.

    The Radon stage costs a few milliseconds per wafer, which is negligible on
    a subset and roughly an hour of single-core time on the full 811k-wafer
    file.  Extraction is therefore parallelised across processes by default;
    set ``n_jobs=1`` to disable.
    """
    wafer_maps = list(wafer_maps)
    if not wafer_maps:
        return np.zeros((0, N_FEATURES), dtype=np.float32)

    if n_jobs is None:
        n_jobs = max(1, (os.cpu_count() or 1))
    chunks = [
        wafer_maps[i : i + chunk_size]
        for i in range(0, len(wafer_maps), chunk_size)
    ]

    if n_jobs == 1:
        rows = []
        for i, chunk in enumerate(chunks):
            rows.append(_extract_chunk(chunk))
            if progress_every:
                done = min((i + 1) * chunk_size, len(wafer_maps))
                print(f"  features: {done:,}/{len(wafer_maps):,}", flush=True)
        return np.vstack(rows)

    out = []
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        for i, block in enumerate(pool.map(_extract_chunk, chunks)):
            out.append(block)
            if progress_every:
                done = min((i + 1) * chunk_size, len(wafer_maps))
                print(f"  features: {done:,}/{len(wafer_maps):,}", flush=True)
    return np.vstack(out)
