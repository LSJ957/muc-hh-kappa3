"""Per-event physics weights for the κ₃ analysis.

`w = σ(pb) × BR_factor × 1000(fb/pb) × L(fb⁻¹) / N_gen`

A weight is the expected number of events at the analysis luminosity that
this MC event represents; histograms summed with these weights give the
expected yield per bin.

All physics inputs (luminosity, cross sections, background process map)
come from the `physics:` block of config/<stage>.yaml — pass `cfg['physics']`
as the `phys` argument.  This module is the single source of truth for the
weight formulas: never re-implement them locally (subtle guards like
`bkg_mask` are easy to drop).
"""
from __future__ import annotations
import numpy as np
from . import physics_constants as pc


def sigbg_weights(
    target_sigbg: np.ndarray,
    target_everytype: np.ndarray,
    phys: dict,
    n_gen_per_process: int | dict,
) -> np.ndarray:
    """Per-event weight for the signal+background sample.

    Parameters
    ----------
    target_sigbg : (N,) int — 1 for signal, 0 for background
    target_everytype : (N,) int — process id (0=signal; background ids are the
        keys of phys['backgrounds'])
    phys : cfg['physics'] — lumi_fb_inv, kappa3_xsec_pb, backgrounds
    n_gen_per_process : int | dict[str, int]
        If int  : one N_gen for signal AND every background process.
        If dict : per-process map {process_name: N_gen, 'signal': N_gen};
                  every process present in the sample must have an entry.

    Returns
    -------
    w : (N,) float64, w >= 0.
        signal → σ(κ3=1)·BR(H→bb)²;  background → per-process σ
        (× one BR(H→bb) where phys marks apply_br_hbb).
    """
    target_sigbg     = np.asarray(target_sigbg, dtype=np.int8)
    target_everytype = np.asarray(target_everytype, dtype=np.int8)
    n = len(target_sigbg)
    if len(target_everytype) != n:
        raise ValueError(f'shape mismatch: target_sigbg {n} vs target_everytype {len(target_everytype)}')

    lumi = float(phys['lumi_fb_inv'])
    bkgs = {int(k): v for k, v in phys['backgrounds'].items()}

    if isinstance(n_gen_per_process, int):
        N_gen_sig = int(n_gen_per_process)
        N_gen_bkg = {pid: int(n_gen_per_process) for pid in bkgs}
    elif isinstance(n_gen_per_process, dict):
        N_gen_sig = int(n_gen_per_process['signal'])
        N_gen_bkg = {pid: int(n_gen_per_process[v['name']]) for pid, v in bkgs.items()}
    else:
        raise TypeError(f'n_gen_per_process must be int or dict; got {type(n_gen_per_process)}')

    sig_mask = (target_sigbg == 1)
    bkg_mask = (target_sigbg == 0)
    w = np.zeros(n, dtype=np.float64)

    # Signal — κ3=1 cross section
    w[sig_mask] = (float(phys['kappa3_xsec_pb'][1.0]) * pc.BR_HBB_SQ * 1000.0
                   * lumi / N_gen_sig)

    # Background, ONLY inside bkg_mask, per-process σ (+ BR(H→bb) if flagged)
    for pid, v in bkgs.items():
        proc_mask = bkg_mask & (target_everytype == pid)
        if proc_mask.any():
            br = pc.BR_HBB if v.get('apply_br_hbb', False) else 1.0
            w[proc_mask] = float(v['xsec_pb']) * br * 1000.0 * lumi / N_gen_bkg[pid]

    # A background event whose target_everytype is missing from
    # phys['backgrounds'] would keep w = 0 and silently vanish from every
    # yield and from the likelihood's background B — fail loud instead.
    _unmapped = bkg_mask & (w == 0.0)
    if _unmapped.any():
        bad = np.unique(target_everytype[_unmapped]).tolist()
        raise ValueError(
            f'{int(_unmapped.sum())} background events carry process ids {bad} '
            f'absent from phys["backgrounds"] — they would silently drop out of B')
    assert (w >= 0).all(), 'sigbg_weights produced negative weight'
    return w


def kappa_weights(
    k3_values: np.ndarray,
    phys: dict,
    n_gen_per_kappa: int,
    k3_subset: list[float] | None = None,
) -> np.ndarray:
    """Per-event weight for κ3-scan signal events.

    Parameters
    ----------
    k3_values : (N,) float — per-event κ3 value (stored kappa3_value)
    phys : cfg['physics']
    n_gen_per_kappa : generated events per κ3 slice for THIS sample
    k3_subset : optional list of κ3 values to include; events at other κ3 get
        w = 0.  Default: every κ3 in phys['kappa3_xsec_pb'].

    Returns
    -------
    w : (N,) float64.  Events whose κ3 is not in the table get w = 0.
    """
    k3_values = np.asarray(k3_values, dtype=np.float64)
    w = np.zeros(len(k3_values), dtype=np.float64)
    lumi  = float(phys['lumi_fb_inv'])
    N_gen = int(n_gen_per_kappa)
    table = {float(k): float(v) for k, v in phys['kappa3_xsec_pb'].items()}
    subset = set(k3_subset) if k3_subset is not None else set(table)

    for k_nom, xsec_pb in table.items():
        if k_nom not in subset:
            continue
        mask = pc.kappa_match(k3_values, k_nom)
        if mask.any():
            w[mask] = xsec_pb * pc.BR_HBB_SQ * 1000.0 * lumi / N_gen

    assert (w >= 0).all(), 'kappa_weights produced negative weight'
    return w
