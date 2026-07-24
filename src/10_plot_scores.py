#!/usr/bin/env python3
"""10_plot_scores.py — classifier score distributions (paper-style):
  left  : D_HH on the held-out test fold, signal vs per-process-weighted
          background (from ml1_scores.npz — no inference needed)
  right : D_κ3 evaluated on the independent κ3 slices (κ_low / κ_ref /
          κ_high from the config) — light GPU inference on that sample.

By default the background in the left panel is a single line (the cross-
section-weighted sum of all processes, normalized to unit area).  With
--split-bg it is instead drawn as a stack, one colour per background
process; the top edge of the stack equals the default total.
Output: analysis/<stage>/fig_scores.png  (…_splitbg.png with the flag)"""
import os, sys, argparse, json
_LIB = os.environ.get('HHML_CONDA_LIB')
if _LIB is None:
    print('WARNING: HHML_CONDA_LIB not set; skipping LD_LIBRARY_PATH injection.', flush=True)
elif _LIB not in os.environ.get('LD_LIBRARY_PATH', ''):
    os.environ['LD_LIBRARY_PATH'] = _LIB + ':' + os.environ.get('LD_LIBRARY_PATH', '')
    os.execv(sys.executable, [sys.executable] + sys.argv)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, 'lib'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lib.config_loader import load_config, resolve_paths
from lib.data_loader import load_input
from lib.weights import sigbg_weights
from lib.spanet_engine import recompute_hl_from_assignment
from lib.physics_constants import KAPPA_MATCH_TOL
from lib import ml_arch as MA


def norm_hist(vals, edges, w=None):
    """Fraction-per-bin normalization: bin heights sum to 1 (paper
    convention), NOT a probability density."""
    h, _ = np.histogram(vals, bins=edges, weights=w)
    s = h.sum()
    return h / (s if s > 0 else 1)


# display labels for the --split-bg legend (fallback: raw config name)
BG_LABELS = {
    'hqqvv': r'$Hq\bar q\,\nu\bar\nu$', 'wwvv': r'$W^+W^-\nu\bar\nu$',
    'zzvv':  r'$ZZ\,\nu\bar\nu$',       'ttvv': r'$t\bar t\,\nu\bar\nu$',
    'ww':    r'$W^+W^-$',               'zz':   r'$ZZ$',
    'tt':    r'$t\bar t$',
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--split-bg', action='store_true',
                    help='stack the background processes in individual colours '
                         '(cross-section weighted) instead of the single total line')
    args = ap.parse_args()
    cfg = load_config(args.config); cfg = resolve_paths(cfg, os.path.join(HERE, os.pardir))
    M = cfg['models_dir']

    # ── left panel: D_HH test-fold scores + per-process background weights ──
    z1 = np.load(os.path.join(M, 'ml1_scores.npz'))
    d1, y1, idx1 = z1['d_test'], z1['y_test'], z1['idx_test']
    sb_name = cfg['ml_usage']['ml1']['sigbg'][0]
    sb = load_input(cfg, sb_name, load_jets=False, load_truth=False, load_ll_cloud=False)
    n_gen = int(cfg['inputs'][sb_name]['n_gen_per_process'])
    w_all = sigbg_weights(sb['target_sigbg'], sb['target_everytype'],
                          cfg['physics'], n_gen)
    w1 = w_all[idx1]

    # ── right panel: D_κ3 on the independent κ3 slices ──
    from tensorflow.keras.models import load_model
    ml2_meta = json.load(open(os.path.join(M, 'ml2_best.json')))
    ml2 = load_model(os.path.join(M, 'ml2.keras'), compile=False, safe_mode=False)
    src = cfg['dll']['anchor']['source']
    d = load_input(cfg, src, load_jets=True, load_truth=False, load_ll_cloud=True)
    assign = np.load(os.path.join(M, f'assign_{src}.npy')).astype(np.int8)
    assert len(assign) == d['N'], 'assign file stale vs h5 — re-run 03'
    rec = recompute_hl_from_assignment(d['jets'], assign, d['hl']['met'], d['met_phi'])
    for k, v in rec.items():
        d['hl'][k] = v
    jc, jb = MA.build_jet_tokens(d['jets'], d['met_phi'], False)
    ht     = MA.build_higgs_tokens(d['jets'], assign, False)
    gnt    = MA.build_globals_non_tda(d['hl'], False, drop=ml2_meta['drop'])
    gtda   = MA.build_globals_tda(d['hl'])
    X = dict(jet_cont=jc, jet_btag=jb, higgs_tok=ht,
             globals_non_tda=gnt, globals_tda=gtda, ll_cloud=d['ll_cloud'])
    d2 = ml2.predict(X, batch_size=8192, verbose=0).ravel()

    k_lo  = float(cfg['ml_usage']['ml2']['kappa_low'])
    k_hi  = float(cfg['ml_usage']['ml2']['kappa_high'])
    k_ref = float(cfg['dll']['anchor']['kappa'])
    k3v   = d['kappa3_value']
    colors = {k_lo: '#d62728', k_ref: 'black', k_hi: '#ff7f0e'}

    fig, (aL, aR) = plt.subplots(1, 2, figsize=(9.5, 3.6))
    edges = np.linspace(0, 1, 41)
    if args.split_bg:
        # stacked per-process decomposition; the stack's top edge equals
        # the default single-line total (same weights, same normalization)
        tev1 = sb['target_everytype'][idx1]
        m_bg = y1 < 0.5
        bg_procs = sorted(cfg['physics']['backgrounds'].items())
        cmap = plt.get_cmap('tab10')
        hs = []
        for pid, meta in bg_procs:
            sel = m_bg & (tev1 == int(pid))
            h, _ = np.histogram(d1[sel], bins=edges, weights=w1[sel])
            hs.append(h)
        tot = np.sum(hs, axis=0).sum()
        tot = tot if tot > 0 else 1
        bottom = np.zeros(len(edges) - 1)
        for j, ((pid, meta), h) in enumerate(zip(bg_procs, hs)):
            hn = h / tot
            aL.fill_between(edges, np.r_[bottom, bottom[-1]],
                            np.r_[bottom + hn, (bottom + hn)[-1]],
                            step='post', color=cmap(j % 10), lw=0, alpha=0.8,
                            label=BG_LABELS.get(meta['name'], meta['name']),
                            zorder=1)
            bottom += hn
    else:
        aL.stairs(norm_hist(d1[y1 < 0.5], edges, w=w1[y1 < 0.5]), edges,
                  color='0.45', lw=1.5, label='background')
    h_sig = norm_hist(d1[y1 > 0.5], edges, w=w1[y1 > 0.5])
    aL.stairs(h_sig, edges, color='#1f77b4', lw=1.5, label='signal', zorder=3)
    aL.set_xlabel(r'$\mathcal{D}_{HH}$ score'); aL.set_yscale('log')
    if args.split_bg:
        # two decades of headroom so the 4-row legend clears every curve
        ymax = max(float(h_sig.max()), float(bottom.max()))
        aL.set_ylim(top=ymax * 10**2)
        aL.legend(frameon=False, fontsize=7, ncol=2, loc='upper left')
    else:
        aL.legend(frameon=False, fontsize=9)
    for kv in (k_lo, k_ref, k_hi):
        m = np.abs(k3v - kv) < KAPPA_MATCH_TOL
        if m.any():
            aR.stairs(norm_hist(d2[m], edges), edges, color=colors[kv],
                      lw=1.5, label=rf'$\kappa_3={kv}$')
    aR.set_xlabel(r'$\mathcal{D}_{\kappa_3}$ score')
    aR.legend(frameon=False, fontsize=9)
    for a in (aL, aR):
        a.set_ylabel('normalized events'); a.set_xlim(0, 1)
    lbl = {'3tev': '3 TeV', '10tev': '10 TeV'}.get(cfg['stage'], cfg['stage'])
    fig.suptitle(f'Muon Collider Simulation — √s = {lbl}, resolved', fontsize=11)
    fig.tight_layout()
    stem = 'fig_scores_splitbg' if args.split_bg else 'fig_scores'
    out = os.path.join(cfg['analysis_dir'], f'{stem}.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print(f'saved {out}')


if __name__ == '__main__':
    main()
