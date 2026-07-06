"""Per-stage physics & analysis constants.  Stage dispatch via env var
`PIPELINE_STAGE` ∈ {'3tev', '10tev'} (default '3tev').  Set the env var
BEFORE importing this module — most easily, run scripts via run_all.sh or
the orchestrator, which `export PIPELINE_STAGE=<stage>` for you.

The stage-independent block holds physical / convention constants used
identically by both stages.  The stage-specific block (luminosity,
background MG cross sections, signal κ-scan cross sections) switches on
PIPELINE_STAGE at import time."""
from __future__ import annotations
import os
import numpy as np

_STAGE = os.environ.get('PIPELINE_STAGE', '3tev').lower()
if _STAGE not in ('3tev', '10tev'):
    raise ValueError(f"PIPELINE_STAGE={_STAGE!r}; expected '3tev' or '10tev'")

# ─── Stage-independent ──────────────────────────────────────────────────
M_HIGGS_GEV       = 125.0
BR_HBB            = 0.58            # PDG/LHCHWG round value used throughout the pipeline
BR_HBB_SQ         = BR_HBB * BR_HBB
PT_H_RESOLVED_MAX = 200.0           # resolved-cut p_T(H) [GeV]
BTAG_CUT          = 3               # n_btag_total ≥ this; used by
                                    # lib.weights.{sigbg,kappa}_weights weight-zeroing,
                                    # only when the caller sets apply_btag_cut=True.
                                    # The default analysis applies NO b-tag cut (the
                                    # b-tag information is an ML input feature instead).
                                    # The ML2 *training* BTAG filter is a separate knob:
                                    # config.training.ml2_btag_cut (-1 = no cut, ≥0 = cut).
KAPPA_MATCH_TOL   = 0.005
DLL_EPS           = 1e-12
N_GEN_SIGBG_PER_PROCESS = 500_000
N_GEN_KAPPA_PER_SLICE   = 100_000
PROC_ID_MAP       = {1: 'hqqvv', 2: 'wwvv', 3: 'zzvv', 4: 'ttvv',
                     5: 'ww',    6: 'zz',   7: 'tt'}
BKG_WITH_BR_HBB   = {'hqqvv'}
# Note: ML2 binary κ endpoints live in config (ml_usage.ml2.kappa_low/high),
# not as module constants — they are an analysis choice, not a physical one.

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
if _STAGE == '3tev':
    LUMI_FB_INV = 1000.0            # 1 ab⁻¹
    MG_BKG_XSEC_PB = {
        'hqqvv': 0.0054554757, 'wwvv': 0.0195176659, 'zzvv': 0.0121599876,
        'ttvv': 0.0022278844,  'ww':   0.0091913000, 'zz':   0.0007549620,
        'tt':   0.0077916000,
    }
    KAPPA3_XSEC_PB = {
        0.2: 0.0016855439, 0.3: 0.0015585708, 0.4: 0.0014420082,
        0.5: 0.0013356636, 0.6: 0.0012382819, 0.7: 0.0011468616,
        0.8: 0.0010634673, 0.9: 0.0009909105,
        0.91: 0.0009853129, 0.92: 0.0009789296, 0.93: 0.0009705545,
        0.94: 0.0009655240, 0.95: 0.0009596581, 0.96: 0.0009517012,
        0.97: 0.0009472637, 0.98: 0.0009393185, 0.99: 0.0009347636,
        1.00: 0.0009278950,
        1.01: 0.0009215055, 1.02: 0.0009166326, 1.03: 0.0009107009,
        1.04: 0.0009045958, 1.05: 0.0008981928, 1.06: 0.0008926530,
        1.07: 0.0008880907, 1.08: 0.0008833411, 1.09: 0.0008773227,
        1.1: 0.0008725252, 1.2: 0.0008250127, 1.3: 0.0007883169,
        1.4: 0.0007583037, 1.5: 0.0007395282, 1.6: 0.0007267047,
        1.7: 0.0007244452, 1.8: 0.0007312202, 1.9: 0.0007446261,
        2.0: 0.0007683928,
    }
elif _STAGE == '10tev':
    LUMI_FB_INV = 10000.0           # 10 ab⁻¹
    MG_BKG_XSEC_PB = {
        'hqqvv': 1.468999e-02, 'wwvv': 3.396777e-02, 'zzvv': 2.393354e-02,
        'ttvv': 6.982144e-03,  'ww':   2.607950e-06, 'zz':   3.858410e-07,
        'tt':   7.035200e-04,
    }
    KAPPA3_XSEC_PB = {
        # coarse (11)
        0.2: 5.607739e-03, 0.4: 5.021998e-03, 0.6: 4.538471e-03,
        0.8: 4.111436e-03,
        0.9: 3.937230e-03, 1.0: 3.784274e-03, 1.1: 3.642486e-03,
        1.2: 3.520894e-03,
        1.4: 3.349818e-03, 1.6: 3.246342e-03, 1.8: 3.226440e-03,
        # fine (older 0.05 / 0.01 grids near κ=1)
        0.85: 4.022992e-03,
        0.95: 3.863903e-03, 0.96: 3.840072e-03, 0.97: 3.829861e-03,
        0.98: 3.812337e-03, 0.99: 3.801538e-03,
        1.01: 3.769258e-03, 1.02: 3.751506e-03, 1.03: 3.738320e-03,
        1.04: 3.721660e-03, 1.05: 3.707396e-03,
        1.15: 3.582288e-03,
        # 0.04-grid additions near κ=1 (final 17-pt grid)
        0.84: 4.038709e-03, 0.88: 3.974985e-03, 0.92: 3.907216e-03,
        1.08: 3.668753e-03, 1.12: 3.618311e-03, 1.16: 3.571793e-03,
    }

# Derived
PROC_BKG_FB = {
    name: xs_pb * (BR_HBB if name in BKG_WITH_BR_HBB else 1.0) * 1000.0
    for name, xs_pb in MG_BKG_XSEC_PB.items()
}
def kappa_match(values: np.ndarray, k_nominal: float,
                tol: float = KAPPA_MATCH_TOL) -> np.ndarray:
    return np.abs(np.asarray(values, dtype=np.float64) - float(k_nominal)) < tol
