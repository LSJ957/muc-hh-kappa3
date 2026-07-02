"""Asimov DLL + connected-region w68 extractor.

Single source of truth for w68 extraction. NEVER use max-min span across
multiple disjoint <0.5 regions (that was the latent bug in `lib.py:extract_w68`
of the old codebase). Always use the connected interval containing the
global minimum.

Two entry points, one per DLL-curve type:
  • `poly4_w68(k, dll)` — 4th-order polyfit then connected width, for a SPARSE
                          per-κ scatter (raw per-κ Asimov points).  This is the
                          extractor behind the paper's headline w68 numbers.
  • `connected_w68_on_fine(kfine, dllB, fit_range)` — already-smooth DLL curve
                                      (e.g. from per-bin quadratic morphing,
                                      07_dll_morphing); reads the connected
                                      width on the fine grid directly.

Both honour the same convention: shift = min(DLL inside fit_range), threshold
0.5, connected region containing the in-range argmin.
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
      w68_canonical : raw max-min span of mask, for backwards compatibility
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
            w68_canonical=float('nan'),
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


def connected_w68_on_fine(kfine: np.ndarray, dllB: np.ndarray,
                          fit_range: tuple[float, float]) -> dict:
    """Extract the connected-region w68 from an already-smooth fine-grid DLL
    curve (e.g. the morphed Asimov DLL output of 07).  Does NOT do an extra
    polyfit — assumes `dllB` is already a smooth function of κ (per-bin
    quadratic morphing then Asimov).

    The reference minimum is taken INSIDE `fit_range` (the user-chosen κ
    interval, typically [0.2, 1.8]).  The connected <0.5 region is expanded
    from that in-range argmin both directions; expansion is restricted to
    `fit_range` so the well never bleeds out at κ < fit_range[0] or κ >
    fit_range[1] even if dllB happens to dip below the in-range minimum
    outside the requested range (a corner case that biased the previous
    inline implementation in 07).

    Returns
    -------
    dict with: w68_connected, k3_lo, k3_hi, k3_min, dll_min_inrange,
    touches_boundary (True → the connected region hits the edge of fit_range,
    so the returned width is a lower bound).
    """
    kfine = np.asarray(kfine, dtype=np.float64)
    dllB  = np.asarray(dllB,  dtype=np.float64)
    if kfine.shape != dllB.shape:
        raise ValueError(f'shape mismatch: kfine {kfine.shape} vs dllB {dllB.shape}')

    lo, hi = float(fit_range[0]), float(fit_range[1])
    in_range = (kfine >= lo - 1e-9) & (kfine <= hi + 1e-9)
    if not in_range.any():
        raise ValueError(f'no kfine points inside fit_range [{lo}, {hi}]')

    dll_min = float(dllB[in_range].min())
    sh = dllB - dll_min
    below = (sh < 0.5) & in_range            # restrict expansion to fit_range

    i0_rel = int(np.argmin(dllB[in_range]))
    i0 = int(np.where(in_range)[0][0]) + i0_rel
    a = b = i0
    while a > 0           and below[a-1]: a -= 1
    while b < len(below)-1 and below[b+1]: b += 1
    _in_idx = np.where(in_range)[0]
    return dict(
        w68_connected = float(kfine[b] - kfine[a]),
        k3_lo         = float(kfine[a]),
        k3_hi         = float(kfine[b]),
        k3_min        = float(kfine[i0]),
        dll_min_inrange = dll_min,
        touches_boundary = bool(a <= int(_in_idx[0]) or b >= int(_in_idx[-1])),
    )
