"""Weighted-quantile bin edges for 2D template histograms.

Used by the analysis evaluator to make d2 bin edges that put approximately
equal weighted count in each bin. d1 uses uniform bins on [0, 1] because the
ML1 sigmoid output is by construction in [0, 1] and approximately uniform on
that range.

Why we don't just use `np.quantile`: the calibration set is signal κ=1 + bkg
test, weighted by physics weights, not the empirical CDF of all events.
"""
from __future__ import annotations
import numpy as np


def weighted_quantile_edges(
    values: np.ndarray,
    weights: np.ndarray,
    n_bins: int,
    eps: float | None = None,
) -> np.ndarray:
    """Bin edges (length n_bins+1) so each bin has ~equal weighted count.

    Parameters
    ----------
    values : (N,) score values
    weights : (N,) non-negative weights; events with w==0 are ignored
    n_bins : number of bins
    eps : padding added to outermost edges. Default = `4 * ULP(max(|values|))`,
          guaranteeing every input event lands strictly inside an edge.

    Returns
    -------
    edges : (n_bins+1,) sorted, strictly-increasing edges.
    """
    values = np.asarray(values, dtype=np.float64).ravel()
    weights = np.asarray(weights, dtype=np.float64).ravel()
    if len(values) != len(weights):
        raise ValueError('values and weights length mismatch')

    keep = (weights > 0) & np.isfinite(values)
    if not keep.any():
        # Degenerate: empty calibration set. Return uniform [0,1] as fallback.
        return np.linspace(0.0, 1.0, n_bins + 1)

    v = values[keep]; w = weights[keep]
    order = np.argsort(v, kind='stable')
    vs = v[order]; ws = w[order]
    cw = np.cumsum(ws); cw /= cw[-1]

    # Adaptive epsilon: based on float64 ULP near the range max
    if eps is None:
        scale = max(abs(vs[0]), abs(vs[-1]), 1.0)
        eps = 4.0 * np.spacing(scale)        # ~4 × ULP near the extrema

    edges = np.empty(n_bins + 1, dtype=np.float64)
    edges[0]  = vs[0]  - eps
    edges[-1] = vs[-1] + eps

    quantile_targets = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    interior = np.searchsorted(cw, quantile_targets)
    interior = np.clip(interior, 0, len(vs) - 1)
    edges[1:-1] = vs[interior]

    # Enforce strict monotonicity (duplicates → bump by ULP × multiplier)
    for k in range(1, len(edges)):
        if edges[k] <= edges[k - 1]:
            edges[k] = edges[k - 1] + max(eps, np.spacing(edges[k - 1]))

    return edges


def uniform_edges(n_bins: int, lo: float = 0.0, hi: float = 1.0,
                   pad: float = 1e-6) -> np.ndarray:
    """Uniform `n_bins`-edge array on [lo-pad, hi+pad]. Used for the d1 axis
    when ML1 outputs a sigmoid probability."""
    return np.linspace(lo - pad, hi + pad, n_bins + 1)
