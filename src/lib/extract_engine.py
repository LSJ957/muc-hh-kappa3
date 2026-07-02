#!/usr/bin/env python3
"""extract_engine.py — ROOT → HDF5 feature extraction library.

Imported by 01_extract_features.py — there is no standalone entry point.

Provides:
  * process_one_root(...) — single Delphes .root file → dict containing
      /hl (45 HL features + metadata), /jets (N, 4, 10), /ll_cloud (N, 40, 4),
      truth_pairing/valid/match_dr, and per-file statistics.
  * save_h5(...)          — write the dict + metadata to one HDF5 file.

Per-stage paths, sample lists, cross-sections, and BTAG cuts are driven by
`config/<stage>.yaml` + `lib.physics_constants`.

HDF5 schema (per file produced by 01_extract_features.py):
  /hl/<col>          — 45 HL features + Event_ID, target_sigbg, target_everytype,
                        kappa3_value, diagram_label, met_phi (gzip compressed)
  /ll_cloud          — (N, 40, 4) particle cloud [pT_frac, Δη, Δφ, type]
                       (mask channel dropped 2026-06-02 to align with the
                       boosted analysis's no-mask convention; padded slots
                       are zero-vectors (0,0,0,0) and handled by the
                       network as such)
  /jets              — (N, 4, 10) jet features [pT, η, φ, mass, btag_wp70,
                        NCharged, NNeutrals, ChargedEFrac, PTD, MeanSqDeltaR]
  /truth_pairing     — (N,) int8: correct pairing index (0,1,2) or -1 if unmatched
  /truth_valid       — (N,) bool: whether all 4 gen b-quarks matched to 4 jets
  /truth_match_dr    — (N, 4) float32: ΔR of each gen b-quark to its matched jet
"""

import numpy as np
import h5py
import uproot
import awkward as ak
from multiprocessing import Pool
import time, os

from . import physics_constants as _pc

# ═════════════════════════════════════════════════════════════════════════
# 0.  CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════
# Path/sample-list constants from the legacy stand-alone runner were removed
# 2026-05-28 — they referenced machine-specific paths and a 3 TeV-only sample
# table.  This module is now only imported as a library (process_one_root /
# save_h5); all paths/x-sections live in `config/<stage>.yaml` and
# `physics_constants.py`.

JET_BRANCH  = "VLCjetR05N4"
N_JETS      = 4
MAX_CONST   = 10          # max constituents per jet for LL cloud
N_PARTICLES = N_JETS * MAX_CONST   # 40
BTAG_BIT    = 1           # WP70 (BitNumber=1 in Delphes card)
# (M_HIGGS removed 2026-06-02: no longer used in this module after the XHH
# formula switched to the ATLAS resolved asymmetric form (centres 120, 110);
# downstream files that need m_H still import it directly from physics_constants.)
TRUTH_DR_MAX = 0.5        # ΔR threshold for gen b-quark ↔ reco jet matching

NUM_CORES = min(os.cpu_count(), 16)

# (Legacy 3T-only sample tables SIGBG_SAMPLES / KAPPA3_XSEC_PB / DIAGRAM_SAMPLES
# were removed 2026-05-28.  All cross-sections and per-stage paths now live in
# `physics_constants.py` and the per-stage YAMLs under `config/`.)

# ─── Branches to read from ROOT ───
BRANCHES_TO_READ = [
    # Jets
    f"{JET_BRANCH}.PT", f"{JET_BRANCH}.Eta", f"{JET_BRANCH}.Phi",
    f"{JET_BRANCH}.Mass", f"{JET_BRANCH}.BTag",
    f"{JET_BRANCH}.Constituents",
    # Jet substructure
    f"{JET_BRANCH}.NCharged", f"{JET_BRANCH}.NNeutrals",
    f"{JET_BRANCH}.ChargedEnergyFraction",
    f"{JET_BRANCH}.PTD", f"{JET_BRANCH}.MeanSqDeltaR",
    f"{JET_BRANCH}.Flavor",
    # MET
    "MissingET.MET", "MissingET.Phi",
    # Leptons (for veto)
    "Electron.PT", "Electron.Eta",
    "Muon.PT", "Muon.Eta",
    # EFlow (for TDA)
    "EFlow.Eta", "EFlow.Phi", "EFlow.ET",
    # EFlow particles (for LL cloud)
    "EFlowTrack.fUniqueID", "EFlowTrack.PT", "EFlowTrack.Eta",
    "EFlowTrack.Phi", "EFlowTrack.PID",
    "EFlowPhoton.fUniqueID", "EFlowPhoton.ET", "EFlowPhoton.Eta",
    "EFlowPhoton.Phi",
    "EFlowNeutralHadron.fUniqueID", "EFlowNeutralHadron.ET",
    "EFlowNeutralHadron.Eta", "EFlowNeutralHadron.Phi",
]

# GenParticle branches (read separately for truth matching)
GEN_BRANCHES = [
    "Particle.PID", "Particle.Status",
    "Particle.D1", "Particle.D2",
    "Particle.PT", "Particle.Eta", "Particle.Phi", "Particle.Mass",
]

# Pruned 45 HL feature list — single source of truth in physics_constants.
HL_FEATURES_45 = list(_pc.HL_FEATURES_45)

# SPANet jet feature names (stored in /jets dataset)
JET_FEATURE_NAMES = [
    'pT', 'eta', 'phi', 'mass', 'btag_wp70',
    'NCharged', 'NNeutrals', 'ChargedEFrac', 'PTD', 'MeanSqDeltaR',
]
N_JET_FEATURES = len(JET_FEATURE_NAMES)

# 3 di-jet pairings — single source of truth in physics_constants.
PAIRINGS = list(_pc.PAIRINGS_FLAT)


# ═════════════════════════════════════════════════════════════════════════
# 1.  HELPER FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════
def vec_dphi(phi1, phi2):
    """Δφ in [0, π]"""
    d = np.abs(phi1 - phi2)
    return np.where(d > np.pi, 2*np.pi - d, d)


def vec_dr(eta1, phi1, eta2, phi2):
    """ΔR = √(Δη² + Δφ²)"""
    return np.sqrt((eta1 - eta2)**2 + vec_dphi(phi1, phi2)**2)


def vec_p4_kinematics(E, px, py, pz):
    """(E, px, py, pz) → (pT, η, φ, m)"""
    pt = np.sqrt(px**2 + py**2)
    p  = np.sqrt(pt**2 + pz**2)
    safe_p = np.where(p == np.abs(pz), p + 1e-9, p)
    eta = np.where(p > 1e-9,
                   0.5 * np.log((safe_p + pz) / (safe_p - pz)), 0.0)
    phi = np.arctan2(py, px)
    m   = np.sqrt(np.clip(E**2 - p**2, 0, None))
    return pt, eta, phi, m


# ═════════════════════════════════════════════════════════════════════════
# 2.  TDA WITH CYLINDRICAL DISTANCE MATRIX
# ═════════════════════════════════════════════════════════════════════════
def compute_tda_single(event_data):
    """
    Persistent homology from EFlow (η, φ, ET) with cylindrical metric.
    d(i,j) = √(Δη² + (2·sin(Δφ/2))²)
    """
    try:
        import ripser
    except ImportError:
        # Loud, once-per-process warning.  Note that
        # `compute_tda_single` is dispatched via `multiprocessing.Pool`, so each
        # worker process initialises its own `globals()`; on a clean install
        # missing ripser the warning therefore prints once per worker
        # (≤ NUM_CORES lines) rather than exactly once.  Acceptable: the
        # purpose is to break silent-zero behaviour, not to be ultra-quiet.
        global _RIPSER_WARNED
        if '_RIPSER_WARNED' not in globals() or not _RIPSER_WARNED:
            print('WARNING: ripser not installed — all 5 TDA features '
                  '(H_0, H_1, S_0, S_1, LB_1) will be 0; ML1/ML2 will train on '
                  'dead inputs.  pip install ripser to fix.', flush=True)
            _RIPSER_WARNED = True
        return 0.0, 0.0, 0.0, 0.0, 0.0

    eta, phi, et = np.array(event_data[0]), np.array(event_data[1]), np.array(event_data[2])
    mask = et > 0
    eta, phi, et = eta[mask], phi[mask], et[mask]
    if len(eta) < 3:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    n = len(eta)
    deta = eta[:, None] - eta[None, :]
    dphi = phi[:, None] - phi[None, :]
    dphi_cyl = 2.0 * np.sin(dphi / 2.0)
    dist = np.sqrt(deta**2 + dphi_cyl**2)

    dgms = ripser.ripser(dist, distance_matrix=True, maxdim=1)['dgms']

    h0_life = dgms[0][:-1, 1] - dgms[0][:-1, 0] if len(dgms[0]) > 1 else np.array([])
    h1_life = dgms[1][:, 1] - dgms[1][:, 0] if len(dgms[1]) > 0 else np.array([])

    L0 = np.sum(h0_life)
    s0 = -np.sum((h0_life / L0) * np.log2(h0_life / L0 + 1e-30)) if L0 > 0 else 0.0
    L1 = np.sum(h1_life)
    s1 = -np.sum((h1_life / L1) * np.log2(h1_life / L1 + 1e-30)) if L1 > 0 else 0.0
    lb1 = np.sum(dgms[1][:, 0] * h1_life) if len(dgms[1]) > 0 else 0.0

    return float(L0), float(L1), float(s0), float(s1), float(lb1)


# ═════════════════════════════════════════════════════════════════════════
# 3.  TRUTH MATCHING: GenParticle H→bb ↔ Reco Jets
# ═════════════════════════════════════════════════════════════════════════
def truth_match_event(gen_pid, gen_status, gen_d1, gen_d2,
                      gen_pt, gen_eta, gen_phi,
                      reco_jet_eta, reco_jet_phi):
    """
    For a single event, find the truth jet pairing from GenParticle H→bb.

    Strategy:
      1. Find all Higgs (PID=25) in the event
      2. Require exactly 2 Higgs
      3. For each Higgs, get its 2 daughter b-quarks (|PID|=5)
      4. Match each gen b-quark to the closest reco jet via ΔR
      5. Require all 4 matches are unique and ΔR < TRUTH_DR_MAX
      6. Determine which PAIRINGS index matches the truth assignment

    Returns:
        pairing_idx : int  — 0, 1, or 2 (matching PAIRINGS), or -1 if no match
        valid       : bool — whether truth matching succeeded
        match_drs   : ndarray (4,) — ΔR for each gen b-quark match (0 if unmatched)
    """
    match_drs = np.zeros(4, dtype=np.float32)

    # Find Higgs particles (PID=25)
    higgs_indices = [i for i, p in enumerate(gen_pid) if p == 25]
    if len(higgs_indices) != 2:
        return -1, False, match_drs

    # Collect b-quark daughters from each Higgs
    # h_bquarks[0] = list of (eta, phi) for H1's b-quarks
    # h_bquarks[1] = list of (eta, phi) for H2's b-quarks
    h_bquarks = [[], []]
    for hi, h_idx in enumerate(higgs_indices):
        d1_idx = gen_d1[h_idx]
        d2_idx = gen_d2[h_idx]
        if d1_idx < 0 or d2_idx < 0:
            return -1, False, match_drs
        for di in range(d1_idx, d2_idx + 1):
            if di < len(gen_pid) and abs(gen_pid[di]) == 5:
                h_bquarks[hi].append((gen_eta[di], gen_phi[di]))
        if len(h_bquarks[hi]) != 2:
            return -1, False, match_drs

    # Match each gen b-quark to closest reco jet
    # gen_b_jets[h_idx][b_idx] = reco jet index
    gen_b_jets = [[-1, -1], [-1, -1]]  # [higgs_idx][b_idx] → jet_idx
    all_matched_jets = []

    for hi in range(2):
        for bi in range(2):
            b_eta, b_phi = h_bquarks[hi][bi]
            drs = np.array([
                np.sqrt((b_eta - reco_jet_eta[j])**2 +
                        vec_dphi_scalar(b_phi, reco_jet_phi[j])**2)
                for j in range(N_JETS)
            ])
            best_j = np.argmin(drs)
            best_dr = drs[best_j]

            if best_dr > TRUTH_DR_MAX:
                return -1, False, match_drs

            gen_b_jets[hi][bi] = int(best_j)
            match_drs[hi * 2 + bi] = best_dr
            all_matched_jets.append(int(best_j))

    # Check all 4 matches are unique (4 b-quarks → 4 different jets)
    if len(set(all_matched_jets)) != 4:
        return -1, False, match_drs

    # Determine which pairing index matches
    # H1 jets = {gen_b_jets[0][0], gen_b_jets[0][1]}
    # H2 jets = {gen_b_jets[1][0], gen_b_jets[1][1]}
    h1_jets = frozenset(gen_b_jets[0])
    h2_jets = frozenset(gen_b_jets[1])

    for pi, (a, b, c, d) in enumerate(PAIRINGS):
        pair1 = frozenset([a, b])
        pair2 = frozenset([c, d])
        # Either H1↔pair1,H2↔pair2 or H1↔pair2,H2↔pair1 (H1↔H2 symmetry)
        if (h1_jets == pair1 and h2_jets == pair2) or \
           (h1_jets == pair2 and h2_jets == pair1):
            return pi, True, match_drs

    return -1, False, match_drs


def vec_dphi_scalar(phi1, phi2):
    """Scalar Δφ in [0, π]"""
    d = abs(phi1 - phi2)
    return d if d <= np.pi else 2*np.pi - d


def truth_match_batch(gen_arrays, reco_jet_eta, reco_jet_phi, n_events):
    """
    Run truth matching on a batch of events.

    Returns:
        truth_pairing : ndarray (N,) int8
        truth_valid   : ndarray (N,) bool
        truth_match_dr: ndarray (N, 4) float32
    """
    truth_pairing = np.full(n_events, -1, dtype=np.int8)
    truth_valid   = np.zeros(n_events, dtype=bool)
    truth_match_dr = np.zeros((n_events, 4), dtype=np.float32)

    pid_list = gen_arrays['pid']
    status_list = gen_arrays['status']
    d1_list = gen_arrays['d1']
    d2_list = gen_arrays['d2']
    pt_list = gen_arrays['pt']
    eta_list = gen_arrays['eta']
    phi_list = gen_arrays['phi']

    n_matched = 0
    for ev in range(n_events):
        pi, valid, drs = truth_match_event(
            ak.to_list(pid_list[ev]),
            ak.to_list(status_list[ev]),
            ak.to_list(d1_list[ev]),
            ak.to_list(d2_list[ev]),
            ak.to_list(pt_list[ev]),
            ak.to_list(eta_list[ev]),
            ak.to_list(phi_list[ev]),
            reco_jet_eta[ev],
            reco_jet_phi[ev],
        )
        truth_pairing[ev] = pi
        truth_valid[ev] = valid
        truth_match_dr[ev] = drs
        if valid:
            n_matched += 1

        if (ev + 1) % 100000 == 0:
            print(f"      Truth matching: {ev+1}/{n_events} "
                  f"(matched so far: {n_matched})", flush=True)

    print(f"    Truth matching complete: {n_matched}/{n_events} "
          f"({100*n_matched/max(n_events,1):.1f}%) matched", flush=True)

    return truth_pairing, truth_valid, truth_match_dr


# ═════════════════════════════════════════════════════════════════════════
# 4.  CORE: PROCESS ONE ROOT FILE
# ═════════════════════════════════════════════════════════════════════════
def process_one_root(root_path, label_sigbg, label_everytype,
                     kappa3_value=np.nan, diagram_label=-1,
                     is_hh_signal=False):
    """
    Read a single ROOT file and extract all features after selection cuts.

    Args:
        is_hh_signal : bool — if True, perform GenParticle truth matching
                       (only for HH signal events where 2 Higgs exist)
    Returns:
        result_dict with keys: hl_dict, ll_cloud, jets_array,
        truth_pairing, truth_valid, truth_match_dr, stats
    """
    t0 = time.time()
    print(f"  Opening {root_path} ...", flush=True)

    rf = uproot.open(root_path)
    tree = rf["Delphes;1"]
    arrays = tree.arrays(BRANCHES_TO_READ, library="ak")
    n_total = len(arrays)

    # ─── Selection cuts ───
    # 1. Lepton veto
    has_e = ak.any((arrays["Electron.PT"] > 10.0) &
                   (np.abs(arrays["Electron.Eta"]) < 2.5), axis=1)
    has_mu = ak.any((arrays["Muon.PT"] > 10.0) &
                    (np.abs(arrays["Muon.Eta"]) < 2.5), axis=1)
    lep_veto = ~(has_e | has_mu)

    # 2. Exactly 4 jets, each with pT > 20 GeV and |η| < 2.5.
    #    Delphes runs the Exclusive-4 jet algorithm, so every event has
    #    exactly 4 jets at the input.  `sum(j_mask) == 4` therefore means
    #    "ALL 4 jets pass the per-jet kinematic cut" — events with any
    #    jet failing pT or η are vetoed at the event level (no fallback to
    #    softer jets, since there are no extra jets to fall back to).
    j_mask = (arrays[f"{JET_BRANCH}.PT"] > 20.0) & \
             (np.abs(arrays[f"{JET_BRANCH}.Eta"]) < 2.5)
    jet_cut = ak.sum(j_mask, axis=1) == 4

    basic_mask = ak.to_numpy(lep_veto & jet_cut)
    arr = arrays[basic_mask]
    n_basic = len(arr)

    # ─── Jet 4-momenta ───
    j_pt   = ak.to_numpy(arr[f"{JET_BRANCH}.PT"][:, :4])
    j_eta  = ak.to_numpy(arr[f"{JET_BRANCH}.Eta"][:, :4])
    j_phi  = ak.to_numpy(arr[f"{JET_BRANCH}.Phi"][:, :4])
    j_mass = ak.to_numpy(arr[f"{JET_BRANCH}.Mass"][:, :4])
    j_btag_raw = ak.to_numpy(arr[f"{JET_BRANCH}.BTag"][:, :4])
    j_btag = (j_btag_raw.astype(int) >> BTAG_BIT) & 1

    # Jet substructure variables
    j_ncharged   = ak.to_numpy(arr[f"{JET_BRANCH}.NCharged"][:, :4]).astype(np.float32)
    j_nneutrals  = ak.to_numpy(arr[f"{JET_BRANCH}.NNeutrals"][:, :4]).astype(np.float32)
    j_chefrac    = ak.to_numpy(arr[f"{JET_BRANCH}.ChargedEnergyFraction"][:, :4]).astype(np.float32)
    j_ptd        = ak.to_numpy(arr[f"{JET_BRANCH}.PTD"][:, :4]).astype(np.float32)
    j_meansqdr   = ak.to_numpy(arr[f"{JET_BRANCH}.MeanSqDeltaR"][:, :4]).astype(np.float32)

    j_px = j_pt * np.cos(j_phi)
    j_py = j_pt * np.sin(j_phi)
    j_pz = j_pt * np.sinh(j_eta)
    j_E  = np.sqrt(j_mass**2 + j_px**2 + j_py**2 + j_pz**2)

    # ─── XHH Higgs reconstruction ───
    # ATLAS arXiv:2202.07288 resolved-channel definition:
    #   X_HH = sqrt( ((m_H1 - 120)/(0.1 m_H1))^2 + ((m_H2 - 110)/(0.1 m_H2))^2 )
    # The centres (120, 110) GeV are asymmetric between the two Higgs
    # candidates; H_1 / H_2 are pT-sorted (H_1 = leading p_T) before applying
    # the formula. We adopt the ATLAS resolved convention to harmonise with
    # the two-boosted-jet companion analysis.
    M_H1_CENTER = 120.0
    M_H2_CENTER = 110.0
    EPS_M       = 1e-3        # guards 0.1·m → 0
    best_xhh = np.full(n_basic, 1e9)
    best_pair = np.zeros(n_basic, dtype=int)

    for pi, (a, b, c, d) in enumerate(PAIRINGS):
        E1  = j_E[:, a] + j_E[:, b];  px1 = j_px[:, a] + j_px[:, b]
        py1 = j_py[:, a] + j_py[:, b]; pz1 = j_pz[:, a] + j_pz[:, b]
        m1  = np.sqrt(np.clip(E1**2 - px1**2 - py1**2 - pz1**2, 0, None))
        pt1 = np.sqrt(px1**2 + py1**2)

        E2  = j_E[:, c] + j_E[:, d];  px2 = j_px[:, c] + j_px[:, d]
        py2 = j_py[:, c] + j_py[:, d]; pz2 = j_pz[:, c] + j_pz[:, d]
        m2  = np.sqrt(np.clip(E2**2 - px2**2 - py2**2 - pz2**2, 0, None))
        pt2 = np.sqrt(px2**2 + py2**2)

        # pT-sort within this candidate pairing: H1 = leading pT
        lead1 = pt1 >= pt2
        mH1   = np.where(lead1, m1, m2)
        mH2   = np.where(lead1, m2, m1)
        d1    = (mH1 - M_H1_CENTER) / np.maximum(0.1 * mH1, EPS_M)
        d2    = (mH2 - M_H2_CENTER) / np.maximum(0.1 * mH2, EPS_M)
        xhh   = np.sqrt(d1*d1 + d2*d2)
        better = xhh < best_xhh
        best_xhh[better] = xhh[better]
        best_pair[better] = pi

    # Reconstruct best H1 (leading pT), H2
    H1_E  = np.zeros(n_basic); H1_px = np.zeros(n_basic)
    H1_py = np.zeros(n_basic); H1_pz = np.zeros(n_basic)
    H2_E  = np.zeros(n_basic); H2_px = np.zeros(n_basic)
    H2_py = np.zeros(n_basic); H2_pz = np.zeros(n_basic)
    H1_btag = np.zeros(n_basic, dtype=int)
    H2_btag = np.zeros(n_basic, dtype=int)

    for pi, (a, b, c, d) in enumerate(PAIRINGS):
        m = best_pair == pi
        if m.sum() == 0:
            continue
        E1 = j_E[m, a] + j_E[m, b]; px1 = j_px[m, a] + j_px[m, b]
        py1 = j_py[m, a] + j_py[m, b]; pz1 = j_pz[m, a] + j_pz[m, b]
        E2 = j_E[m, c] + j_E[m, d]; px2 = j_px[m, c] + j_px[m, d]
        py2 = j_py[m, c] + j_py[m, d]; pz2 = j_pz[m, c] + j_pz[m, d]

        pt1 = np.sqrt(px1**2 + py1**2)
        pt2 = np.sqrt(px2**2 + py2**2)
        swap = pt2 > pt1

        H1_E[m]  = np.where(swap, E2, E1)
        H1_px[m] = np.where(swap, px2, px1)
        H1_py[m] = np.where(swap, py2, py1)
        H1_pz[m] = np.where(swap, pz2, pz1)
        H2_E[m]  = np.where(swap, E1, E2)
        H2_px[m] = np.where(swap, px1, px2)
        H2_py[m] = np.where(swap, py1, py2)
        H2_pz[m] = np.where(swap, pz1, pz2)

        H1_btag[m] = np.where(swap, j_btag[m, c] + j_btag[m, d],
                                     j_btag[m, a] + j_btag[m, b])
        H2_btag[m] = np.where(swap, j_btag[m, a] + j_btag[m, b],
                                     j_btag[m, c] + j_btag[m, d])

    H1_pt, H1_eta, H1_phi, H1_m = vec_p4_kinematics(H1_E, H1_px, H1_py, H1_pz)
    H2_pt, H2_eta, H2_phi, H2_m = vec_p4_kinematics(H2_E, H2_px, H2_py, H2_pz)

    # ─── pT(H) < 200 GeV (resolved regime) ───
    ptH_mask = (H1_pt < _pc.PT_H_RESOLVED_MAX) & (H2_pt < _pc.PT_H_RESOLVED_MAX)
    n_final = int(ptH_mask.sum())

    if n_final == 0:
        print(f"  WARNING: No events survived cuts for {root_path}")
        return None

    # Apply ptH cut to all arrays
    basic_orig_indices = np.where(basic_mask)[0]
    final_orig_indices = basic_orig_indices[ptH_mask]

    j_pt = j_pt[ptH_mask]; j_eta = j_eta[ptH_mask]
    j_phi = j_phi[ptH_mask]; j_mass = j_mass[ptH_mask]
    j_btag = j_btag[ptH_mask]
    j_ncharged = j_ncharged[ptH_mask]; j_nneutrals = j_nneutrals[ptH_mask]
    j_chefrac = j_chefrac[ptH_mask]; j_ptd = j_ptd[ptH_mask]
    j_meansqdr = j_meansqdr[ptH_mask]
    j_E = j_E[ptH_mask]; j_px = j_px[ptH_mask]
    j_py = j_py[ptH_mask]; j_pz = j_pz[ptH_mask]
    best_xhh = best_xhh[ptH_mask]
    H1_pt = H1_pt[ptH_mask]; H1_eta = H1_eta[ptH_mask]
    H1_phi = H1_phi[ptH_mask]; H1_m = H1_m[ptH_mask]
    H2_pt = H2_pt[ptH_mask]; H2_eta = H2_eta[ptH_mask]
    H2_phi = H2_phi[ptH_mask]; H2_m = H2_m[ptH_mask]
    H1_E = H1_E[ptH_mask]; H1_px = H1_px[ptH_mask]
    H1_py = H1_py[ptH_mask]; H1_pz = H1_pz[ptH_mask]
    H2_E = H2_E[ptH_mask]; H2_px = H2_px[ptH_mask]
    H2_py = H2_py[ptH_mask]; H2_pz = H2_pz[ptH_mask]
    H1_btag = H1_btag[ptH_mask]; H2_btag = H2_btag[ptH_mask]

    arr_final = arrays[basic_mask][ptH_mask]

    # MET
    met     = ak.to_numpy(arr_final["MissingET.MET"][:, 0])
    met_phi = ak.to_numpy(arr_final["MissingET.Phi"][:, 0])

    # ─── Di-Higgs kinematics ───
    HH_E  = H1_E + H2_E;   HH_px = H1_px + H2_px
    HH_py = H1_py + H2_py; HH_pz = H1_pz + H2_pz
    HH_pt, HH_eta, HH_phi, HH_m = vec_p4_kinematics(HH_E, HH_px, HH_py, HH_pz)

    dphi_hh_met = vec_dphi(HH_phi, met_phi)
    mT_HH_met = np.sqrt(np.clip(2 * HH_pt * met * (1 - np.cos(dphi_hh_met)), 0, None))

    H1_p = np.sqrt(H1_px**2 + H1_py**2 + H1_pz**2)
    H2_p = np.sqrt(H2_px**2 + H2_py**2 + H2_pz**2)
    dot_p = H1_px * H2_px + H1_py * H2_py + H1_pz * H2_pz
    cos_HH_lab = np.where((H1_p > 0) & (H2_p > 0), dot_p / (H1_p * H2_p), 0.0)

    beta_z = HH_pz / (HH_E + 1e-9)   # zero-guard, matches recompute_hl_from_assignment
    gamma  = HH_E / np.sqrt(np.clip(HH_E**2 - HH_pz**2, 1e-9, None))
    H1_pz_cm = gamma * (H1_pz - beta_z * H1_E)
    H1_pt_cm = np.sqrt(H1_px**2 + H1_py**2)
    H1_p_cm  = np.sqrt(H1_pt_cm**2 + H1_pz_cm**2)
    cos_theta_hel = np.where(H1_p_cm > 0, H1_pz_cm / H1_p_cm, 0.0)

    scalar_sum_E = np.sum(j_E, axis=1)
    vec_sum_px = np.sum(j_px, axis=1)
    vec_sum_py = np.sum(j_py, axis=1)
    vec_sum_pz = np.sum(j_pz, axis=1)
    vec_sum_p  = np.sqrt(vec_sum_px**2 + vec_sum_py**2 + vec_sum_pz**2)

    # ─── Build HL feature dict ───
    hl = {
        'Event_ID':         np.arange(n_final),
        'target_sigbg':     np.full(n_final, label_sigbg, dtype=np.int8),
        'target_everytype': np.full(n_final, label_everytype, dtype=np.int8),
        'kappa3_value':     np.full(n_final, kappa3_value, dtype=np.float32),
        'diagram_label':    np.full(n_final, diagram_label, dtype=np.int8),
    }

    for j in range(N_JETS):
        hl[f'jet{j}_pt']   = j_pt[:, j]
        hl[f'jet{j}_mass'] = j_mass[:, j]
        hl[f'jet{j}_btag'] = j_btag[:, j].astype(np.int8)
        hl[f'dphi_jet{j}_met'] = vec_dphi(j_phi[:, j], met_phi)

    for j1 in range(N_JETS):
        for j2 in range(j1 + 1, N_JETS):
            hl[f'dR_{j1}{j2}'] = vec_dr(j_eta[:, j1], j_phi[:, j1],
                                        j_eta[:, j2], j_phi[:, j2])

    hl['H1_pt']    = H1_pt;   hl['H1_m']  = H1_m
    hl['H2_pt']    = H2_pt;   hl['H2_m']  = H2_m
    hl['H1_nbtag'] = H1_btag.astype(np.int8)
    hl['H2_nbtag'] = H2_btag.astype(np.int8)
    hl['dR_H1H2']  = vec_dr(H1_eta, H1_phi, H2_eta, H2_phi)

    hl['mHH']         = HH_m
    hl['pT_HH']       = HH_pt
    hl['XHH']         = best_xhh
    hl['dphi_HH_met'] = dphi_hh_met
    hl['mT_HH_met']   = mT_HH_met
    hl['cos_HH_lab']  = cos_HH_lab

    hl['cos_theta_hel'] = cos_theta_hel
    hl['n_btag_total']  = np.sum(j_btag, axis=1).astype(np.int8)
    hl['met']           = met
    hl['met_phi']       = met_phi    # raw MET φ — needed for SPANet HL recomputation
    hl['scalar_sum_E']  = scalar_sum_E
    hl['vec_sum_p']     = vec_sum_p

    # ─── TDA features ───
    print(f"    Computing TDA ({n_final} events)...", flush=True)
    tda_input = list(zip(
        arr_final["EFlow.Eta"].to_list(),
        arr_final["EFlow.Phi"].to_list(),
        arr_final["EFlow.ET"].to_list()
    ))
    with Pool(processes=NUM_CORES) as pool:
        tda_results = pool.map(compute_tda_single, tda_input)
    tda_arr = np.array(tda_results)
    hl['H_0']  = tda_arr[:, 0]
    hl['H_1']  = tda_arr[:, 1]
    hl['S_0']  = tda_arr[:, 2]
    hl['S_1']  = tda_arr[:, 3]
    hl['LB_1'] = tda_arr[:, 4]

    # ─── Jets array for SPANet: (N, 4, 10) ───
    jets_array = np.stack([
        j_pt, j_eta, j_phi, j_mass,
        j_btag.astype(np.float32),
        j_ncharged, j_nneutrals, j_chefrac, j_ptd, j_meansqdr,
    ], axis=-1).astype(np.float32)   # (N, 4, 10)

    # ─── LL particle cloud (40×4) ───  (mask channel dropped, see header docstring)
    print(f"    Building LL cloud ({n_final} events)...", flush=True)
    ll_cloud = np.zeros((n_final, N_PARTICLES, 4), dtype=np.float32)

    j_const_raw = tree[f"{JET_BRANCH}.Constituents"].array()
    if hasattr(j_const_raw, 'fields') and "refs" in j_const_raw.fields:
        j_const_ids = j_const_raw["refs"][final_orig_indices]
    else:
        j_const_ids = j_const_raw[final_orig_indices]

    trk_id  = tree["EFlowTrack.fUniqueID"].array()[final_orig_indices]
    trk_pt  = tree["EFlowTrack.PT"].array()[final_orig_indices]
    trk_eta = tree["EFlowTrack.Eta"].array()[final_orig_indices]
    trk_phi = tree["EFlowTrack.Phi"].array()[final_orig_indices]
    trk_pid = tree["EFlowTrack.PID"].array()[final_orig_indices]

    pho_id  = tree["EFlowPhoton.fUniqueID"].array()[final_orig_indices]
    pho_pt  = tree["EFlowPhoton.ET"].array()[final_orig_indices]
    pho_eta = tree["EFlowPhoton.Eta"].array()[final_orig_indices]
    pho_phi = tree["EFlowPhoton.Phi"].array()[final_orig_indices]

    nh_id  = tree["EFlowNeutralHadron.fUniqueID"].array()[final_orig_indices]
    nh_pt  = tree["EFlowNeutralHadron.ET"].array()[final_orig_indices]
    nh_eta = tree["EFlowNeutralHadron.Eta"].array()[final_orig_indices]
    nh_phi = tree["EFlowNeutralHadron.Phi"].array()[final_orig_indices]

    trk_pid_abs = abs(trk_pid)
    trk_type = ak.where(trk_pid_abs == 11, 1,
               ak.where(trk_pid_abs == 13, 2, 5))
    pho_type = ak.full_like(pho_pt, 3, dtype=int)
    nh_type  = ak.full_like(nh_pt, 4, dtype=int)

    for ev_i in range(n_final):
        all_id   = np.concatenate([ak.to_numpy(trk_id[ev_i]),  ak.to_numpy(pho_id[ev_i]),  ak.to_numpy(nh_id[ev_i])])
        all_pt   = np.concatenate([ak.to_numpy(trk_pt[ev_i]),  ak.to_numpy(pho_pt[ev_i]),  ak.to_numpy(nh_pt[ev_i])])
        all_eta  = np.concatenate([ak.to_numpy(trk_eta[ev_i]), ak.to_numpy(pho_eta[ev_i]), ak.to_numpy(nh_eta[ev_i])])
        all_phi  = np.concatenate([ak.to_numpy(trk_phi[ev_i]), ak.to_numpy(pho_phi[ev_i]), ak.to_numpy(nh_phi[ev_i])])
        all_type = np.concatenate([ak.to_numpy(trk_type[ev_i]), ak.to_numpy(pho_type[ev_i]), ak.to_numpy(nh_type[ev_i])])

        if len(all_id) == 0:
            continue

        for j_idx in range(N_JETS):
            jet_pt_val  = j_pt[ev_i, j_idx]
            jet_eta_val = j_eta[ev_i, j_idx]
            jet_phi_val = j_phi[ev_i, j_idx]

            jet_const = ak.to_numpy(j_const_ids[ev_i][j_idx]) \
                        if len(j_const_ids[ev_i]) > j_idx else np.array([])
            if len(jet_const) == 0:
                continue

            matched = np.isin(all_id, jet_const)
            c_pt   = all_pt[matched]
            c_eta  = all_eta[matched]
            c_phi  = all_phi[matched]
            c_type = all_type[matched]

            if len(c_pt) == 0:
                continue

            order = np.argsort(c_pt)[::-1]
            c_pt   = c_pt[order][:MAX_CONST]
            c_eta  = c_eta[order][:MAX_CONST]
            c_phi  = c_phi[order][:MAX_CONST]
            c_type = c_type[order][:MAX_CONST]
            n_c = len(c_pt)
            offset = j_idx * MAX_CONST

            ll_cloud[ev_i, offset:offset+n_c, 0] = c_pt / (jet_pt_val + 1e-9)
            dphi_c = c_phi - jet_phi_val
            dphi_c = np.where(dphi_c > np.pi, dphi_c - 2*np.pi, dphi_c)
            dphi_c = np.where(dphi_c < -np.pi, dphi_c + 2*np.pi, dphi_c)
            ll_cloud[ev_i, offset:offset+n_c, 1] = c_eta - jet_eta_val
            ll_cloud[ev_i, offset:offset+n_c, 2] = dphi_c
            ll_cloud[ev_i, offset:offset+n_c, 3] = c_type
            # NOTE: previous 5th channel (mask flag = 1.0) dropped to align
            # with the boosted analysis's no-mask convention. Padded slots
            # remain zero-vectors (0,0,0,0) — the network treats them as
            # implicit zero contribution.

        if (ev_i + 1) % 50000 == 0:
            print(f"      LL: {ev_i+1}/{n_final} events", flush=True)

    # ─── Truth matching (HH signal only) ───
    if is_hh_signal:
        print(f"    Running truth matching ({n_final} events)...", flush=True)
        gen_arrays_all = tree.arrays(GEN_BRANCHES, library="ak")
        # Apply same event selection
        gen_selected = gen_arrays_all[basic_mask][ptH_mask]
        gen_dict = {
            'pid':    gen_selected["Particle.PID"],
            'status': gen_selected["Particle.Status"],
            'd1':     gen_selected["Particle.D1"],
            'd2':     gen_selected["Particle.D2"],
            'pt':     gen_selected["Particle.PT"],
            'eta':    gen_selected["Particle.Eta"],
            'phi':    gen_selected["Particle.Phi"],
        }
        truth_pairing, truth_valid, truth_match_dr = truth_match_batch(
            gen_dict, j_eta, j_phi, n_final
        )
    else:
        truth_pairing = np.full(n_final, -1, dtype=np.int8)
        truth_valid   = np.zeros(n_final, dtype=bool)
        truth_match_dr = np.zeros((n_final, 4), dtype=np.float32)

    # ─── Statistics ───
    stats = {
        'n_total': n_total, 'n_basic': n_basic, 'n_final': n_final,
        'eff_basic': n_basic / n_total if n_total > 0 else 0,
        'eff_ptH':   n_final / n_basic if n_basic > 0 else 0,
        'eff_total': n_final / n_total if n_total > 0 else 0,
        'n_truth_matched': int(truth_valid.sum()),
    }

    elapsed = time.time() - t0
    print(f"    Done in {elapsed:.1f}s — {n_final} events "
          f"(ε={stats['eff_total']:.4f}, truth_matched={stats['n_truth_matched']})",
          flush=True)

    return {
        'hl_dict': hl,
        'll_cloud': ll_cloud,
        'jets_array': jets_array,
        'truth_pairing': truth_pairing,
        'truth_valid': truth_valid,
        'truth_match_dr': truth_match_dr,
        'stats': stats,
    }


# ═════════════════════════════════════════════════════════════════════════
# 5.  SAVE TO HDF5 (EXTENDED FORMAT)
# ═════════════════════════════════════════════════════════════════════════
def save_h5(output_path, hl_df, ll_cloud, jets_array,
            truth_pairing, truth_valid, truth_match_dr,
            meta_dict=None):
    """Save all data to HDF5 with extended structure."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"  Saving {output_path} ({len(hl_df)} events)...", flush=True)
    with h5py.File(output_path, 'w') as f:
        # HL features
        grp = f.create_group('hl')
        for col in hl_df.columns:
            grp.create_dataset(col, data=hl_df[col].values, compression='gzip')

        # LL particle cloud
        f.create_dataset('ll_cloud', data=ll_cloud, compression='gzip')

        # SPANet jet features
        ds_jets = f.create_dataset('jets', data=jets_array, compression='gzip')
        ds_jets.attrs['feature_names'] = JET_FEATURE_NAMES
        ds_jets.attrs['shape_desc'] = '(N_events, 4_jets, 10_features)'

        # Truth matching labels
        f.create_dataset('truth_pairing', data=truth_pairing, compression='gzip')
        f.create_dataset('truth_valid', data=truth_valid, compression='gzip')
        f.create_dataset('truth_match_dr', data=truth_match_dr, compression='gzip')

        # Metadata
        if meta_dict:
            meta = f.create_group('meta')
            for k, v in meta_dict.items():
                meta.attrs[k] = v

        # Store pairing definition for reference
        f.attrs['PAIRINGS'] = str(PAIRINGS)
        f.attrs['JET_FEATURE_NAMES'] = str(JET_FEATURE_NAMES)
        f.attrs['HL_FEATURES_45'] = str(HL_FEATURES_45)



# (Legacy stand-alone runners — run_sigbg, run_kappa_scan, run_diagram, main —
# were removed 2026-05-28.  Pipeline entry-point is 01_extract_features.py, which
# imports the helpers above; this module is library-only.)
