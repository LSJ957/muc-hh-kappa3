"""Asimov DLL + connected-region w68 extractor.

Single source of truth for w68 extraction.  The width is ALWAYS the
connected <0.5 interval containing the fitted minimum, never the max-min
span across disjoint regions (which double-counts secondary wiggles).

Entry point:
  • `poly4_w68(k, dll)` — 4th-order polyfit then connected width, for the
                          sparse per-κ scan (raw per-κ Asimov points).  This
                          is the extractor behind the paper's headline w68.

Convention: shift = fitted polynomial minimum, threshold 0.5 (68%) with the
connected region grown from the argmin.
"""
from __future__ import annotations
import numpy as np
from . import physics_constants as pc


def asimov_dll(n_array: np.ndarray, mu_array: np.ndarray,
               eps: float = pc.DLL_EPS) -> float:
    """Saturated Asimov −Δ log L:
        max( Σ_bins [n·log(n/μ) − (n−μ)], 0 )
    with both n and μ floored at `eps` before the log.  (Each bin term is
    analytically ≥ 0, so clipping the total — done here — and clipping per bin
    agree up to float noise; the clip only silences tiny negative residue
    from float cancellations.)

    Both inputs are flattened automatically; their shapes must match.
    """
    n  = np.asarray(n_array,  dtype=np.float64).ravel()
    mu = np.asarray(mu_array, dtype=np.float64).ravel()
    if n.shape != mu.shape:
        raise ValueError(f'shape mismatch: n {n.shape} vs μ {mu.shape}')
    n  = np.maximum(n,  eps)
    mu = np.maximum(mu, eps)
    dll = float((n * np.log(n / mu) - (n - mu)).sum())
    return max(dll, 0.0)


def poly4_w68(
    k3_grid: np.ndarray,
    dll_values: np.ndarray,
    n_fine: int = 5000,
) -> dict:
    """4-th order polyfit + shift to the fitted minimum + connected-region w68 + R².

    Parameters
    ----------
    k3_grid : (N,) ordered κ values
    dll_values : (N,) −Δ log L values at each κ (i.e. `asimov_dll(n, μ(k))`)
    n_fine : grid resolution for interpolation

    Returns dict with keys:
      w68_connected : width of the contiguous <0.5 region containing argmin
      k3_lo, k3_hi  : connected-region boundaries
      k3_min        : κ at the polyfit minimum
      n_regions     : number of disjoint <0.5 regions (diagnostic; if >1 the
                      polynomial wobbles, paper-grade σ likely under-quoted)
      r2            : coefficient of determination of poly4 vs the input scatter
      rmse          : sqrt(mean((dll - poly4(k))²))
      poly_coef     : numpy poly coefficients (highest-power first)
      poly_min_raw  : polynomial minimum BEFORE the shift
      touches_boundary : True if the connected region hits the edge of the κ
                      grid — the returned width is then a LOWER BOUND (the
                      true interval extends beyond the scanned range)
    """
    k = np.asarray(k3_grid, dtype=np.float64)
    d = np.asarray(dll_values, dtype=np.float64)
    if k.shape != d.shape:
        raise ValueError(f'shape mismatch: k {k.shape} vs dll {d.shape}')
    if len(k) < 5:
        raise ValueError('need at least 5 points for a 4-th order fit')

    poly = np.polyfit(k, d, 4)
    fit_at_pts = np.polyval(poly, k)

    # Scatter-quality metrics
    ss_res = float(np.sum((d - fit_at_pts) ** 2))
    ss_tot = float(np.sum((d - d.mean()) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-30)
    rmse = float(np.sqrt(ss_res / len(d)))

    # Fine-grid polyfit values
    kf = np.linspace(k.min(), k.max(), n_fine)
    df = np.polyval(poly, kf)
    poly_min_raw = float(df.min())
    k3_min = float(kf[int(df.argmin())])

    # Shift to the fitted minimum: dll -= dll.min(); clip ≥ 0
    df_shifted = np.maximum(df - poly_min_raw, 0.0)
    mask = df_shifted < 0.5

    if not mask.any():
        return dict(
            w68_connected=float('nan'),
            k3_lo=float('nan'),
            k3_hi=float('nan'),
            k3_min=k3_min,
            n_regions=0,
            r2=r2,
            rmse=rmse,
            poly_coef=poly.tolist(),
            poly_min_raw=poly_min_raw,
            touches_boundary=False,
        )

    # Canonical: max-min span of the entire mask
    w68_canon = float(kf[mask].max() - kf[mask].min())

    # Number of disjoint regions
    n_reg = int((np.diff(mask.astype(int)) == 1).sum()) + (1 if mask[0] else 0)

    # Connected region containing argmin
    imin = int(df_shifted.argmin())
    lo, hi = imin, imin
    while lo > 0 and mask[lo - 1]:
        lo -= 1
    while hi < len(mask) - 1 and mask[hi + 1]:
        hi += 1
    k3_lo = float(kf[lo])
    k3_hi = float(kf[hi])
    w68_conn = k3_hi - k3_lo

    return dict(
        w68_connected=w68_conn,
        w68_canonical=w68_canon,
        k3_lo=k3_lo,
        k3_hi=k3_hi,
        k3_min=k3_min,
        n_regions=n_reg,
        r2=r2,
        rmse=rmse,
        poly_coef=poly.tolist(),
        poly_min_raw=poly_min_raw,
        touches_boundary=bool(lo == 0 or hi == len(mask) - 1),
    )
