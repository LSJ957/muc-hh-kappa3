"""Universal physics constants and structural definitions shared across
the pipeline (Higgs mass, BR(H→bb), jet-pairing enumeration, HL feature
list).  Everything stage- or sample-dependent — luminosity, cross sections,
generation counts — is an analysis input and lives in config/<stage>.yaml
(`physics:` block and per-input `n_gen_*`).
"""
from __future__ import annotations
import numpy as np

# ─── Universal physics constants ────────────────────────────────────────
M_HIGGS_GEV       = 125.0
BR_HBB            = 0.58            # BR(H→bb) applied as BR² for HH→4b signal
BR_HBB_SQ         = BR_HBB * BR_HBB
PT_H_RESOLVED_MAX = 200.0           # resolved-region cut p_T(H) [GeV]
KAPPA_MATCH_TOL   = 0.005
DLL_EPS           = 1e-12
# Note: luminosity, per-process background cross sections and the κ3 signal
# cross-section table are ANALYSIS INPUTS, not constants — they live in the
# `physics:` block of config/<stage>.yaml.  ML2 binary κ endpoints likewise
# live in config (ml_usage.ml2.kappa_low/high).

# ─── 4-jet → 2 di-jet (Higgs) pairings.  Single source of truth — used by
#     SPANet (assignment classes), recompute_hl_from_assignment, and ml_arch
#     (Higgs-token reconstruction).
PAIRINGS_FLAT  = ((0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2))
PAIRINGS_PAIRS = (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2)))
N_PAIRINGS     = len(PAIRINGS_FLAT)

# ─── 45 stored HL features per event (h5 /hl/*) — single source of truth ──
HL_FEATURES_45 = (
    # Jet kinematics (4×3 = 12)
    'jet0_pt', 'jet1_pt', 'jet2_pt', 'jet3_pt',
    'jet0_mass', 'jet1_mass', 'jet2_mass', 'jet3_mass',
    'jet0_btag', 'jet1_btag', 'jet2_btag', 'jet3_btag',
    # Jet-MET angular (4)
    'dphi_jet0_met', 'dphi_jet1_met', 'dphi_jet2_met', 'dphi_jet3_met',
    # Jet pair ΔR (6)
    'dR_01', 'dR_02', 'dR_03', 'dR_12', 'dR_13', 'dR_23',
    # Higgs candidates (7)
    'H1_pt', 'H1_m', 'H2_pt', 'H2_m', 'H1_nbtag', 'H2_nbtag', 'dR_H1H2',
    # Di-Higgs system (6)
    'mHH', 'pT_HH', 'XHH', 'dphi_HH_met', 'mT_HH_met', 'cos_HH_lab',
    # Helicity + global (5)
    'cos_theta_hel', 'n_btag_total', 'met', 'scalar_sum_E', 'vec_sum_p',
    # TDA (5)
    'H_0', 'H_1', 'S_0', 'S_1', 'LB_1',
)
assert len(HL_FEATURES_45) == 45, f'Expected 45 HL features, got {len(HL_FEATURES_45)}'

# Subset that DOES depend on the SPANet jet→Higgs assignment.  Reference
# documentation: spanet_engine.recompute_hl_from_assignment returns exactly
# these keys (it hard-codes them; this tuple is the audited list to compare
# against when editing either side).
ASSIGNMENT_DEPENDENT_HL = (
    'H1_pt', 'H1_m', 'H2_pt', 'H2_m', 'H1_nbtag', 'H2_nbtag',
    'dR_H1H2', 'mHH', 'pT_HH', 'XHH', 'dphi_HH_met', 'mT_HH_met',
    'cos_HH_lab', 'cos_theta_hel',
)

# Default globals_non_tda features to drop from ML1/ML2 (kinematically degenerate).
DEFAULT_DROP_GLOBALS = ('mT_HH_met', 'cos_HH_lab', 'cos_theta_hel', 'vec_sum_p')

# ─── Stage-specific ─────────────────────────────────────────────────────
def kappa_match(values: np.ndarray, k_nominal: float,
                tol: float = KAPPA_MATCH_TOL) -> np.ndarray:
    return np.abs(np.asarray(values, dtype=np.float64) - float(k_nominal)) < tol
