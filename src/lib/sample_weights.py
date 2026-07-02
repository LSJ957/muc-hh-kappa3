"""Per-event sample weights for ML1 (sig/bg) and ML2 (κ-binary).

Single source of truth: tune scripts (04a/05a) and final-train scripts (04/05)
MUST call the same helpers so Optuna optimises the same loss landscape that
final training uses.  See review N-7 (round 2) for the prior asymmetry bug.

The recipe is the same for both heads:
  1) physics weight per event = xsec × BR(²) × LUMI / N_gen  (process-aware)
  2) within each binary class, normalise the mean weight to 1 so the relative
     process ratios are preserved but the absolute scale doesn't fight AdamW.
  3) multiply the *positive* class by class_weight = N_neg / N_pos so the two
     classes contribute equal total weight to the loss.

Returns float32 arrays sized like the input target arrays — drop directly
into `m.fit(..., sample_weight=...)`.
"""
import numpy as np
from .weights import sigbg_weights, kappa_weights


def _per_class_norm(w, y):
    w = w.copy().astype(np.float64)
    for cls_val in (0.0, 1.0):
        m = (y == cls_val)
        if m.any():
            mean_w = float(w[m].mean()) or 1.0
            w[m] = w[m] / mean_w
    return w


def _balance_to_class_ratio(w_norm, y):
    cw_ratio = float((y == 0).sum() / max((y == 1).sum(), 1))
    return np.where(y > 0.5, cw_ratio * w_norm, w_norm).astype(np.float32), cw_ratio


def ml1_sample_weights(target_sigbg, target_everytype, n_btag_total,
                       idx_train, idx_val,
                       y_train, y_val,
                       apply_btag_cut=False,
                       n_gen_per_process=None):
    """Return (sample_weight_tr, sample_weight_va, cw_ratio) for ML1.

    `target_sigbg`, `target_everytype`, `n_btag_total` are the full-pool
    arrays (pre-split), `idx_train`/`idx_val` are the row indices into them.
    `n_gen_per_process` forwards to lib.weights.sigbg_weights so a non-default
    MC generation budget (per the config) is honoured.
    """
    w_phys_all = sigbg_weights(target_sigbg, target_everytype, n_btag_total,
                               apply_btag_cut=apply_btag_cut,
                               n_gen_per_process=n_gen_per_process)
    w_tr_phys = w_phys_all[idx_train].astype(np.float64)
    w_va_phys = w_phys_all[idx_val].astype(np.float64)
    w_tr_norm = _per_class_norm(w_tr_phys, y_train)
    w_va_norm = _per_class_norm(w_va_phys, y_val)
    sw_tr, cw_ratio = _balance_to_class_ratio(w_tr_norm, y_train)
    sw_va, _        = _balance_to_class_ratio(w_va_norm, y_val)
    return sw_tr, sw_va, cw_ratio


def ml2_sample_weights(k3_pool, nbt_pool, idx_train, idx_val,
                       y_train, y_val, k_low, k_high,
                       apply_btag_cut=False):
    """Return (sample_weight_tr, sample_weight_va, cw_ratio) for ML2.

    `k3_pool`/`nbt_pool` are the masked-pool arrays (i.e. already restricted
    to events with κ₃ ∈ {k_low, k_high}); `idx_train`/`idx_val` are indices
    into THAT pool, not the full arrays.
    """
    w_phys_pool = kappa_weights(k3_pool, nbt_pool, apply_btag_cut=apply_btag_cut,
                                k3_subset=[k_low, k_high])
    w_tr_phys = w_phys_pool[idx_train].astype(np.float64)
    w_va_phys = w_phys_pool[idx_val].astype(np.float64)
    w_tr_norm = _per_class_norm(w_tr_phys, y_train)
    w_va_norm = _per_class_norm(w_va_phys, y_val)
    sw_tr, cw_ratio = _balance_to_class_ratio(w_tr_norm, y_train)
    sw_va, _        = _balance_to_class_ratio(w_va_norm, y_val)
    return sw_tr, sw_va, cw_ratio
