"""Per-bin κ³-quadratic morphing for the (ML1,ML2) 2-D template.

Given an iterable of (κ, weighted-template) per available κ-point sample,
fits  S_b(κ) = α_b + β_b·(κ-1) + γ_b·(κ-1)²  by per-bin least-squares.
Then S(κ) = α + β·(κ-1) + γ·(κ-1)² is callable on any κ.

This is *exact* at parton level (σ ∝ κ²) and well-defined per detector bin
because the SM, contact, and trilinear amplitudes interfere coherently:
  N_b(κ) = ∫ σ(x;κ) ε_b(x) dx  =  α_b + β_b κ + γ_b κ²
(ε_b κ-independent — cuts/ML scores see reconstructed observables only).
"""
import numpy as np


def fit_per_bin_quadratic(kappas: np.ndarray, templates: np.ndarray) -> dict:
    """kappas    : shape (Nκ,)              κ values, distinct per sample
       templates : shape (Nκ, Nbins)        per-κ flattened 2-D yield histogram
       Basis: u = κ - 1; coef rows are [α, β, γ] in S = α + β·u + γ·u².
       Returns dict with:
         coef       (3, Nbins)   α/β/γ per bin
         R2         float        GLOBAL R² aggregated over all bins
         per_bin_R2 (Nbins,)     per-bin R² (some may be near 0 in tail bins)
         kappas     (Nκ,)
         basis      str          documents the morphing basis used
    """
    kappas = np.asarray(kappas, dtype=np.float64)
    Y = np.asarray(templates, dtype=np.float64)
    assert Y.shape[0] == len(kappas), f'shape mismatch: Y[0]={Y.shape[0]} vs Nκ={len(kappas)}'
    u = kappas - 1.0
    D = np.vstack([np.ones_like(u), u, u**2]).T              # (Nκ × 3)
    coef, *_ = np.linalg.lstsq(D, Y, rcond=None)             # (3 × Nbins)
    Yfit = D @ coef
    ss_res_per_bin = np.sum((Y - Yfit)**2, axis=0)
    ss_tot_per_bin = np.sum((Y - Y.mean(axis=0, keepdims=True))**2, axis=0)
    per_bin_R2 = 1.0 - ss_res_per_bin / np.maximum(ss_tot_per_bin, 1e-30)
    ss_res_total = float(np.sum((Y - Yfit)**2))
    ss_tot_total = float(max(np.sum((Y - Y.mean(0))**2), 1e-30))
    R2_global = 1.0 - ss_res_total / ss_tot_total
    return dict(coef=coef, R2=R2_global, per_bin_R2=per_bin_R2, kappas=kappas,
                basis='(kappa-1)^[0,1,2]')


def evaluate(fit: dict, k: float) -> np.ndarray:
    """Morphed bin yields S(κ) ≥ 0 (clipped)."""
    a, b, c = fit['coef']
    uu = float(k) - 1.0
    return np.clip(a + b * uu + c * uu * uu, 0.0, None).astype(np.float64)
