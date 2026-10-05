"""Controlled degradation of the inspection input stream.

Every transform here corresponds to a documented way a deployed wafer-map
inspection module loses accuracy in a fab, so that each degradation mode has a
physical counterpart rather than being generic injected noise.

  resolution_loss   Probe-card or test-program change alters effective die
                    grid; maps arrive coarser than the training distribution.
  rotation          Wafer handling or notch-alignment change; orientation-
                    sensitive patterns (Scratch, Edge-Loc) are misread.
  edge_exclusion    Edge-die exclusion rule tightened; the ring region that
                    carries Edge-Ring evidence is progressively removed.
  bin_noise         Tester bin-map flakiness; isolated dies flip pass/fail.
  new_signature     A tool begins producing a pattern absent from training;
                    the classifier has no class for it and mass redistributes.

Each accepts a ``severity`` in [0, 1] so the experiment can sweep gradual
degradation, which is the realistic case; abrupt shifts are the easy case and
should not be the only one reported.
"""

from __future__ import annotations

import numpy as np
from skimage.transform import resize, rotate

from .features import _failure_mask, _valid_mask


def resolution_loss(wafer_map: np.ndarray, severity: float, rng) -> np.ndarray:
    if severity <= 0:
        return wafer_map
    h, w = wafer_map.shape
    factor = 1.0 - 0.6 * severity
    small = resize(
        wafer_map,
        (max(4, int(h * factor)), max(4, int(w * factor))),
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    )
    restored = resize(
        small, (h, w), order=0, preserve_range=True, anti_aliasing=False
    )
    return restored.astype(np.uint8)


def rotation(wafer_map: np.ndarray, severity: float, rng) -> np.ndarray:
    if severity <= 0:
        return wafer_map
    angle = float(rng.uniform(-1, 1)) * 45.0 * severity
    rotated = rotate(
        wafer_map, angle, order=0, preserve_range=True, mode="constant", cval=0
    )
    return rotated.astype(np.uint8)


def edge_exclusion(wafer_map: np.ndarray, severity: float, rng) -> np.ndarray:
    if severity <= 0:
        return wafer_map
    out = wafer_map.copy()
    h, w = out.shape
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    radius = np.sqrt(((yy - cy) / max(cy, 1)) ** 2 + ((xx - cx) / max(cx, 1)) ** 2)
    out[radius > (1.0 - 0.35 * severity)] = 0
    return out


def bin_noise(wafer_map: np.ndarray, severity: float, rng) -> np.ndarray:
    if severity <= 0:
        return wafer_map
    out = wafer_map.copy()
    valid = out > 0
    flip = rng.random(out.shape) < (0.08 * severity)
    target = valid & flip
    out[target] = np.where(out[target] == 1, 2, 1)
    return out


def new_signature(wafer_map: np.ndarray, severity: float, rng) -> np.ndarray:
    """Impose an unseen defect geometry: a diagonal band of failed dies."""
    if severity <= 0:
        return wafer_map
    out = wafer_map.copy()
    h, w = out.shape
    yy, xx = np.mgrid[0:h, 0:w]
    offset = rng.integers(-h // 4, h // 4 + 1)
    width = max(1, int(0.08 * severity * max(h, w)))
    band = np.abs((yy - offset) - xx * (h / max(w, 1))) < width
    out[(out > 0) & band] = 2
    return out


def defect_remix(wafer_map: np.ndarray, severity: float, rng) -> np.ndarray:
    """Redistribute defect *geometry* while holding the defect *level* fixed.

    Every other mode here changes how much of the wafer fails, which moves the
    normal share and is therefore detectable by monitoring proportions alone.
    This one models a process change that swaps one failure mechanism for
    another --- a tool replaced by one that produces blobs where the old one
    produced scratches --- at constant yield.

    Two properties are enforced deliberately:

    * wafers with little failure are left untouched, so the share of wafers
      predicted ``none`` barely moves;
    * the total number of failed dies is held constant, by removing as many
      dies as the dilation adds.

    That leaves the signal almost entirely in the *ratios among defect classes*,
    which is the regime where a log-ratio treatment should matter and a raw
    proportion chart should struggle. It is the discriminating experiment
    between the two, and it is designed to be losable.
    """
    if severity <= 0:
        return wafer_map

    fail = _failure_mask(wafer_map)
    valid = _valid_mask(wafer_map)
    n_valid = int(valid.sum())
    n_fail = int(fail.sum())
    if n_valid == 0 or n_fail == 0:
        return wafer_map
    if n_fail / n_valid < 0.02:
        # Near-clean wafer: leave it alone so the normal share is preserved.
        return wafer_map

    from scipy.ndimage import binary_dilation, binary_erosion
    from scipy.ndimage import label as nd_label

    labels, n_components = nd_label(fail)
    if n_components == 0:
        return wafer_map
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    largest = int(sizes.argmax())

    # Severity controls what FRACTION of eligible wafers is remixed, not how
    # hard each one is dilated. Dilation iterations are integers, so tying
    # severity to them gave the mode almost no resolution: severity 0.25 and
    # 0.50 both rounded to one iteration and produced identical streams, which
    # looked like a robustness result and was an artefact. A fraction is also
    # the more faithful model of a partial tool changeover, where some wafers
    # go through the new mechanism and the rest do not.
    if rng.random() >= severity:
        return wafer_map
    grown = binary_dilation(labels == largest, iterations=2) & valid
    union = grown | fail

    out = wafer_map.copy()
    out[union] = 2

    # Hold the defect level exactly. Dilating one component adds dies, so the
    # same number is removed: first from the other components (which is the
    # point --- mass moves between mechanisms), then from the grown region's
    # own border if that is not enough. Without the second step a wafer whose
    # failures are one connected scratch gains area with nothing to trade
    # against, and the mode silently becomes a defect-level change like all
    # the others.
    excess = int(union.sum()) - n_fail
    flat = out.ravel()
    if excess > 0:
        others = np.flatnonzero((fail & ~grown).ravel())
        if len(others):
            take = min(excess, len(others))
            flat[rng.choice(others, size=take, replace=False)] = 1
            excess -= take
    # Peel the grown region's border repeatedly: one pass is not always
    # enough, and stopping early leaves a defect-level change masquerading as
    # a geometry change.
    guard = 0
    while excess > 0 and guard < 50:
        guard += 1
        current = (flat == 2).reshape(out.shape) & grown
        if not current.any():
            break
        border = (current & ~binary_erosion(current)).ravel()
        idx = np.flatnonzero(border)
        if not len(idx):
            break
        take = min(excess, len(idx))
        flat[rng.choice(idx, size=take, replace=False)] = 1
        excess -= take
    return flat.reshape(out.shape)


TRANSFORMS = {
    "defect_remix": defect_remix,
    "resolution_loss": resolution_loss,
    "rotation": rotation,
    "edge_exclusion": edge_exclusion,
    "bin_noise": bin_noise,
    "new_signature": new_signature,
}


def apply_degradation(
    wafer_maps: list[np.ndarray],
    lots: np.ndarray,
    lot_sequence: list[str],
    mode: str,
    change_point: int,
    max_severity: float = 1.0,
    ramp_lots: int = 0,
    seed: int = 0,
) -> list[np.ndarray]:
    """Degrade every wafer in lots at or after ``change_point``.

    ``ramp_lots = 0`` gives a step change; a positive value ramps severity
    linearly over that many lots, which is the gradual-drift scenario.
    """
    if mode not in TRANSFORMS:
        raise ValueError(f"unknown degradation mode {mode!r}")
    transform = TRANSFORMS[mode]
    rng = np.random.default_rng(seed)

    position_of = {str(lot): i for i, lot in enumerate(lot_sequence)}
    out = []
    for wmap, lot in zip(wafer_maps, lots):
        position = position_of.get(str(lot))
        if position is None or position < change_point:
            out.append(wmap)
            continue
        if ramp_lots > 0:
            severity = min(1.0, (position - change_point + 1) / ramp_lots)
        else:
            severity = 1.0
        out.append(transform(np.asarray(wmap), severity * max_severity, rng))
    return out
