"""Per-event physics weights for the κ₃ analysis.

`w = xsec(pb) × BR_factor × 1000(fb/pb) × LUMI(fb⁻¹) / N_GEN`

A weight is the expected number of events at the analysis luminosity that this
MC event represents. Histograms summed with these weights give the expected
event yield in each bin.

This module is the single source of truth for weight construction: always
`from lib.weights import sigbg_weights, kappa_weights` — never re-implement
the formulas locally (subtle guards like `bkg_mask` are easy to drop).
"""
from __future__ import annotations
import numpy as np
from . import physics_constants as pc


def sigbg_weights(
    target_sigbg: np.ndarray,
    target_everytype: np.ndarray,
    n_btag_total: np.ndarray | None = None,
    apply_btag_cut: bool = False,
    n_gen_per_process: int | dict | None = None,
) -> np.ndarray:
    """Per-event weight for sigbg events.

    Parameters
    ----------
    target_sigbg : (N,) int — 1 for signal, 0 for background
    target_everytype : (N,) int — process id (0=signal, 1..7=bkg per PROC_ID_MAP)
    n_btag_total : (N,) int — sum of b-tag bits over leading-4 jets, or None
    apply_btag_cut : if True and n_btag_total provided, zero the weight where
                     `n_btag_total < BTAG_CUT` (weight-zeroing, not row-dropping)
    n_gen_per_process : int | dict[str, int] | None
        If int   : single N_gen used for signal AND every background process.
        If dict  : per-process map {process_name: N_gen}; missing entries fall
                   back to `pc.N_GEN_SIGBG_PER_PROCESS`.  Process names follow
                   `pc.PROC_ID_MAP` ('hqqvv', 'wwvv', ..., 'tt'); signal uses
                   the key 'signal'.
        If None  : (default, current behaviour) use the module constant
                   `pc.N_GEN_SIGBG_PER_PROCESS` for all processes — matches
                   the current production where every sigbg sample is 500k.
        Pass the config value through this argument so a 3rd-party MC set
        with a different generation budget is weighted correctly.

    Returns
    -------
    w : (N,) float64, with w >= 0.
        - signal events (target_sigbg == 1) use κ=1 cross-section
        - background events (target_sigbg == 0) use per-process xsec
          × BR(H→bb) (for hqqvv only) × LUMI × 1000 / N_GEN
    """
    target_sigbg     = np.asarray(target_sigbg, dtype=np.int8)
    target_everytype = np.asarray(target_everytype, dtype=np.int8)
    n = len(target_sigbg)
    if len(target_everytype) != n:
        raise ValueError(f'shape mismatch: target_sigbg {n} vs target_everytype {len(target_everytype)}')

    # Resolve the per-process N_gen lookup.
    if n_gen_per_process is None:
        N_gen_sig = pc.N_GEN_SIGBG_PER_PROCESS
        N_gen_bkg = {nm: pc.N_GEN_SIGBG_PER_PROCESS for nm in pc.PROC_ID_MAP.values()}
    elif isinstance(n_gen_per_process, int):
        N_gen_sig = int(n_gen_per_process)
        N_gen_bkg = {nm: int(n_gen_per_process) for nm in pc.PROC_ID_MAP.values()}
    elif isinstance(n_gen_per_process, dict):
        N_gen_sig = int(n_gen_per_process.get('signal', pc.N_GEN_SIGBG_PER_PROCESS))
        N_gen_bkg = {nm: int(n_gen_per_process.get(nm, pc.N_GEN_SIGBG_PER_PROCESS))
                     for nm in pc.PROC_ID_MAP.values()}
    else:
        raise TypeError(f'n_gen_per_process must be None, int, or dict; got {type(n_gen_per_process)}')

    sig_mask = (target_sigbg == 1)
    bkg_mask = (target_sigbg == 0)

    w = np.zeros(n, dtype=np.float64)

    # Signal first — κ=1 cross-section
    w_sig = (pc.KAPPA3_XSEC_PB[1.0] * pc.BR_HBB_SQ * 1000.0
             * pc.LUMI_FB_INV / N_gen_sig)
    w[sig_mask] = w_sig

    # Background, ONLY inside bkg_mask
    for pid, pname in pc.PROC_ID_MAP.items():
        proc_mask = bkg_mask & (target_everytype == pid)
        if proc_mask.any():
            w[proc_mask] = (
                pc.PROC_BKG_FB[pname] * pc.LUMI_FB_INV
                / N_gen_bkg[pname]
            )

    # BTAG cut (weight zeroing, not row drop, to preserve indexing alignment)
    if apply_btag_cut and n_btag_total is not None:
        n_btag = np.asarray(n_btag_total)
        if len(n_btag) != n:
            raise ValueError(f'n_btag_total length {len(n_btag)} != N {n}')
        w = np.where(n_btag >= pc.BTAG_CUT, w, 0.0)

    # Diagnostic invariant: no negative weight ever
    assert (w >= 0).all(), 'sigbg_weights produced negative weight'
    return w


def kappa_weights(
    k3_values: np.ndarray,
    n_btag_total: np.ndarray | None = None,
    apply_btag_cut: bool = False,
    k3_subset: list[float] | None = None,
    n_gen_per_kappa: int | None = None,
) -> np.ndarray:
    """Per-event weight for kappa_scan events (signal at various κ₃).

    Parameters
    ----------
    k3_values : (N,) float — per-event κ₃ value (matches stored kappa3_value)
    n_btag_total : (N,) int — optional BTAG count
    apply_btag_cut : zero weight when n_btag_total < BTAG_CUT
    k3_subset : optional list of κ values to include; events with other κ get
                w = 0. Useful for restricting evaluation grid. Default: all
                κ values in KAPPA3_XSEC_PB.
    n_gen_per_kappa : optional override for per-source generation statistics.
                Default uses module-level pc.N_GEN_KAPPA_PER_SLICE (100k), which
                matches kappa_scan_main / kappa_indep.  Pass 500_000 for the
                high-stat kappa_scan_500k source so its per-event weight is
                correctly normalised (else events get 5× over-weight).

    Returns
    -------
    w : (N,) float64. Events with κ not in the table get w = 0.
    """
    k3_values = np.asarray(k3_values, dtype=np.float64)
    n = len(k3_values)
    w = np.zeros(n, dtype=np.float64)
    N_gen = int(n_gen_per_kappa) if n_gen_per_kappa is not None else pc.N_GEN_KAPPA_PER_SLICE

    subset = (set(k3_subset) if k3_subset is not None
              else set(pc.KAPPA3_XSEC_PB.keys()))

    for k_nom in pc.KAPPA3_XSEC_PB:
        if k_nom not in subset:
            continue
        mask = pc.kappa_match(k3_values, k_nom)
        if mask.any():
            xsec_pb = pc.KAPPA3_XSEC_PB[k_nom]
            w[mask] = xsec_pb * pc.BR_HBB_SQ * 1000.0 * pc.LUMI_FB_INV / N_gen

    if apply_btag_cut and n_btag_total is not None:
        n_btag = np.asarray(n_btag_total)
        if len(n_btag) != n:
            raise ValueError(f'n_btag_total length {len(n_btag)} != N {n}')
        w = np.where(n_btag >= pc.BTAG_CUT, w, 0.0)

    assert (w >= 0).all(), 'kappa_weights produced negative weight'
    return w
