"""Per-lot compositions of predicted defect classes, and the ILR transform.

The monitored quantity is the *composition* of classes predicted by the
inspection module over one production lot.  No labels are used at any point
here --- that is the whole premise.

Two ways of forming the composition are supported:

``hard``
    Counts of arg-max predictions.  Interpretable, but produces many structural
    zeros at a lot size of ~25 wafers across nine classes.

``soft``
    Column sums of predicted class probabilities.  Zero-free by construction
    and lower variance; the sensible default, though it assumes the classifier
    is reasonably calibrated.  Report both in the paper --- the difference is
    a legitimate ablation, not a detail to hide.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def lot_compositions(
    lots: np.ndarray,
    probabilities: np.ndarray,
    lot_sequence: list[str],
    mode: str = "hard",  # Default is "hard": counts. "soft" sums predicted
    # probabilities, which the paper shows breaks the multinomial reference
    # covariance (29-fold overall, 75-fold in the rarest balance). Kept only
    # to reproduce the superseded results.
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Aggregate wafer-level predictions into one composition per lot.

    Parameters
    ----------
    lots
        Lot identifier for each wafer, aligned with ``probabilities``.
    probabilities
        ``(n_wafers, n_classes)`` matrix of predicted class probabilities.
    lot_sequence
        Lots in production order; defines the row order of the output.
    mode
        ``'soft'`` or ``'hard'``.

    Returns
    -------
    counts
        ``(n_lots, n_classes)`` unnormalised composition (probability mass or
        hard counts).
    sizes
        ``(n_lots,)`` number of wafers in each lot.  Needed downstream: the
        sampling variance of a composition scales as 1/n, which is what makes
        the monitor rate-invariant rather than merely scale-invariant.
    kept
        The lots actually present, in order.
    """
    if mode not in {"soft", "hard"}:
        raise ValueError(f"mode must be 'soft' or 'hard', got {mode!r}")

    lots = np.asarray(lots, dtype=object)
    n_classes = probabilities.shape[1]
    index: dict[str, list[int]] = {}
    for position, lot in enumerate(lots):
        index.setdefault(str(lot), []).append(position)

    rows, sizes, kept = [], [], []
    for lot in lot_sequence:
        positions = index.get(str(lot))
        if not positions:
            continue
        block = probabilities[positions]
        if mode == "soft":
            row = block.sum(axis=0)
        else:
            hard = np.zeros(n_classes, dtype=float)
            for j in block.argmax(axis=1):
                hard[j] += 1.0
            row = hard
        rows.append(row)
        sizes.append(len(positions))
        kept.append(str(lot))

    if not rows:
        return (
            np.zeros((0, n_classes)),
            np.zeros(0, dtype=int),
            [],
        )
    return np.vstack(rows), np.asarray(sizes, dtype=int), kept


def bayesian_multiplicative_replacement(
    counts: np.ndarray,
    prior: float = 0.5,
) -> np.ndarray:
    """Replace zeros in count compositions, then close to the simplex.

    Uses the Bayesian-multiplicative rule of Martin-Fernandez et al.: zero
    cells receive posterior mass from a Dirichlet prior (``prior=0.5`` is the
    Jeffreys prior) and non-zero cells are shrunk multiplicatively so that the
    ratios among observed parts are preserved --- which is the property that
    makes the result safe to log-ratio transform.

    Rows with no zeros are only closed, not modified.
    """
    counts = np.asarray(counts, dtype=float)
    if np.any(counts < 0):
        raise ValueError("counts must be non-negative")

    out = np.empty_like(counts)
    n_parts = counts.shape[1]
    for i, row in enumerate(counts):
        total = row.sum()
        if total <= 0:
            out[i] = 1.0 / n_parts
            continue
        closed = row / total
        zeros = closed == 0
        if not zeros.any():
            out[i] = closed
            continue
        # Posterior mean under a Dirichlet(prior) prior for the zero cells.
        delta = prior / (total + n_parts * prior)
        filled = closed.copy()
        filled[zeros] = delta
        filled[~zeros] = closed[~zeros] * (1.0 - zeros.sum() * delta)
        out[i] = filled / filled.sum()
    return out


def ilr_basis(n_parts: int) -> np.ndarray:
    """Orthonormal (Helmert) contrast matrix, ``(n_parts - 1, n_parts)``.

    Row ``i`` implements the balance
    ``sqrt(i/(i+1)) * log( g(x_1..x_i) / x_{i+1} )``.
    """
    if n_parts < 2:
        raise ValueError("need at least two parts")
    basis = np.zeros((n_parts - 1, n_parts))
    for i in range(1, n_parts):
        scale = np.sqrt(i / (i + 1.0))
        basis[i - 1, :i] = scale / i
        basis[i - 1, i] = -scale
    return basis


def ilr(compositions: np.ndarray, basis: np.ndarray | None = None) -> np.ndarray:
    """Isometric log-ratio coordinates of closed, strictly positive rows."""
    compositions = np.asarray(compositions, dtype=float)
    if np.any(compositions <= 0):
        raise ValueError(
            "ILR requires strictly positive parts; run "
            "bayesian_multiplicative_replacement first"
        )
    if basis is None:
        basis = ilr_basis(compositions.shape[1])
    return np.log(compositions) @ basis.T


def ilr_inverse(coords: np.ndarray, basis: np.ndarray | None = None) -> np.ndarray:
    """Back-transform ILR coordinates to the simplex (for interpretation)."""
    coords = np.asarray(coords, dtype=float)
    if basis is None:
        basis = ilr_basis(coords.shape[1] + 1)
    unclosed = np.exp(coords @ basis)
    return unclosed / unclosed.sum(axis=1, keepdims=True)


def build_stream(
    lots: np.ndarray,
    probabilities: np.ndarray,
    lot_sequence: list[str],
    mode: str = "hard",
    prior: float = 0.5,
) -> pd.DataFrame:
    """End-to-end: wafer predictions -> per-lot ILR coordinates.

    Returns a frame indexed by production position with the lot id, lot size,
    the closed composition and the ILR coordinates.
    """
    counts, sizes, kept = lot_compositions(
        lots, probabilities, lot_sequence, mode=mode
    )
    closed = bayesian_multiplicative_replacement(counts, prior=prior)
    coords = ilr(closed)

    frame = pd.DataFrame(
        {
            "position": np.arange(len(kept)),
            "lot": kept,
            "lot_size": sizes,
        }
    )
    for j in range(closed.shape[1]):
        frame[f"p{j}"] = closed[:, j]
    for j in range(coords.shape[1]):
        frame[f"ilr{j}"] = coords[:, j]
    return frame


def ilr_matrix(frame: pd.DataFrame) -> np.ndarray:
    """Extract the ILR block from a stream frame."""
    cols = [c for c in frame.columns if c.startswith("ilr")]
    return frame[cols].to_numpy(dtype=float)


def proportion_matrix(frame: pd.DataFrame) -> np.ndarray:
    """Extract the closed-composition block from a stream frame."""
    cols = [c for c in frame.columns if c.startswith("p") and c[1:].isdigit()]
    return frame[cols].to_numpy(dtype=float)
