"""2D weighted histogram. NO `BIN_EPS` floor at the histogram level — the
DLL clipping is the only place a positive floor is applied (so external
consumers can detect true empty bins without confusion).

Reading: an empty bin returns 0.0, not 1e-12.
"""
from __future__ import annotations
import numpy as np


def hist2d(
    d1: np.ndarray,
    d2: np.ndarray,
    w:  np.ndarray,
    edges_d1: np.ndarray,
    edges_d2: np.ndarray,
) -> np.ndarray:
    """Weighted 2D histogram. Returns (n1, n2) array of weighted counts.

    Events outside the edges are silently dropped (np.histogram2d default).
    For our use case the edges should always be wider than the score range
    (uniform_edges on [-pad, 1+pad] for sigmoid d1, weighted_quantile_edges
    extending past min/max for d2). The evaluator's assertions catch
    out-of-range events at debug time.
    """
    d1 = np.asarray(d1, dtype=np.float64).ravel()
    d2 = np.asarray(d2, dtype=np.float64).ravel()
    w  = np.asarray(w,  dtype=np.float64).ravel()
    if not (len(d1) == len(d2) == len(w)):
        raise ValueError(f'shape mismatch: {len(d1)}, {len(d2)}, {len(w)}')
    h, _, _ = np.histogram2d(d1, d2, bins=[edges_d1, edges_d2], weights=w)
    return h


def fraction_out_of_range(d1: np.ndarray, d2: np.ndarray,
                          edges_d1: np.ndarray, edges_d2: np.ndarray,
                          w: np.ndarray | None = None) -> float:
    """Diagnostic: weighted fraction of events outside the (d1, d2) bin range.

    Useful as a sanity check before evaluation. A non-zero value usually
    indicates a score-domain mismatch (e.g. raw d2 fed when the binning was
    computed on clip-rescaled d2)."""
    d1 = np.asarray(d1, dtype=np.float64).ravel()
    d2 = np.asarray(d2, dtype=np.float64).ravel()
    if w is None:
        w = np.ones_like(d1)
    else:
        w = np.asarray(w, dtype=np.float64).ravel()
    in_range = ((d1 >= edges_d1[0]) & (d1 <= edges_d1[-1]) &
                (d2 >= edges_d2[0]) & (d2 <= edges_d2[-1]))
    total_w = w.sum()
    if total_w == 0:
        return 0.0
    return 1.0 - float(w[in_range].sum() / total_w)
