#!/usr/bin/env python3
"""09_plot_kinematics.py — kinematic distributions: κ3 signal slices vs the
per-process-weighted total background (paper-style, config-driven).

The κ3 slices are the shape classifier's two training points plus κ3 = 1
(read from ml_usage.ml2 and dll.anchor — change the config and the figure
follows).  Background events are weighted per process (physics block).

By default the background is drawn as a single grey band (the cross-
section-weighted sum of all processes, normalized to unit area).  With
--split-bg the same total is instead shown as a stack, one colour per
background process, so the per-process composition is visible; the top
edge of the stack equals the default band.
Output: analysis/<stage>/fig_kinematics.png  (…_splitbg.png with the flag)"""
import os, sys, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, 'lib'))
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lib.config_loader import load_config, resolve_paths
from lib.weights import sigbg_weights
from lib.physics_constants import KAPPA_MATCH_TOL

# These panels read the high-level quantities straight from the h5, so the
# pairing-dependent ones (mHH, dR_H1H2) reflect the X_HH pairing applied at
# extraction, not the SPANet re-pairing used for the classifiers. This is an
# illustrative selection-level figure; 10_plot_scores.py re-pairs via SPANet.
PANELS = [
    ('H_0',     r'$H_0$',             lambda f: f['hl/H_0'][:]),
    ('mHH',     r'$m_{HH}$  [GeV]',   lambda f: f['hl/mHH'][:]),
    ('pT_j1',   r'$p_T(j_1)$  [GeV]', lambda f: f['jets'][:, 0, 0]),
    ('dR_H1H2', r'$\Delta R(H_1,H_2)$', lambda f: f['hl/dR_H1H2'][:]),
]

# display labels for the --split-bg legend (fallback: raw config name)
BG_LABELS = {
    'hqqvv': r'$Hq\bar q\,\nu\bar\nu$', 'wwvv': r'$W^+W^-\nu\bar\nu$',
    'zzvv':  r'$ZZ\,\nu\bar\nu$',       'ttvv': r'$t\bar t\,\nu\bar\nu$',
    'ww':    r'$W^+W^-$',               'zz':   r'$ZZ$',
    'tt':    r'$t\bar t$',
}


def norm_hist(vals, edges, w=None):
    """Fraction-per-bin normalization: bin heights sum to 1 (paper
    convention), NOT a probability density."""
    h, _ = np.histogram(vals, bins=edges, weights=w)
    s = h.sum()
    return h / (s if s > 0 else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--split-bg', action='store_true',
                    help='stack the background processes in individual colours '
                         '(cross-section weighted) instead of the single grey band')
    args = ap.parse_args()
    cfg = load_config(args.config); cfg = resolve_paths(cfg, os.path.join(HERE, os.pardir))

    k_lo  = float(cfg['ml_usage']['ml2']['kappa_low'])
    k_hi  = float(cfg['ml_usage']['ml2']['kappa_high'])
    k_ref = float(cfg['dll']['anchor']['kappa'])
    sig_h5 = cfg['inputs'][cfg['dll']['anchor']['source']]['h5']
    sb_name = cfg['ml_usage']['ml1']['sigbg'][0]

    with h5py.File(sig_h5, 'r') as f:
        k3 = f['hl/kappa3_value'][:]
        obs_k = {key: fn(f) for key, _, fn in PANELS}
    with h5py.File(cfg['inputs'][sb_name]['h5'], 'r') as f:
        tsb = f['hl/target_sigbg'][:]
        tev = f['hl/target_everytype'][:]
        bg  = ~tsb.astype(bool)
        obs_bg = {key: fn(f)[bg] for key, _, fn in PANELS}
    n_gen = int(cfg['inputs'][sb_name]['n_gen_per_process'])
    w_bg = sigbg_weights(tsb, tev, cfg['physics'], n_gen)[bg]

    slices = [(k, np.abs(k3 - k) < KAPPA_MATCH_TOL) for k in (k_lo, k_ref, k_hi)]
    colors = {k_lo: '#d62728', k_ref: 'black', k_hi: '#ff7f0e'}
    bg_procs = sorted(cfg['physics']['backgrounds'].items())   # (pid, meta)
    tev_bg = tev[bg]
    cmap = plt.get_cmap('tab10')

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.6))
    for i_panel, ((key, xlbl, _), ax) in enumerate(zip(PANELS, axes.flat)):
        finite = np.isfinite(obs_bg[key])
        lo = np.quantile(obs_bg[key][finite], 0.001)
        hi = max(np.quantile(obs_k[key][m], 0.999) for _, m in slices)
        edges = np.linspace(lo, hi, 51)
        if args.split_bg:
            # stacked per-process decomposition; the stack's top edge equals
            # the default single-band total (same weights, same normalization)
            hs = []
            for pid, meta in bg_procs:
                sel = tev_bg == int(pid)
                h, _ = np.histogram(obs_bg[key][sel], bins=edges, weights=w_bg[sel])
                hs.append(h)
            tot = np.sum(hs, axis=0).sum()
            tot = tot if tot > 0 else 1
            bottom = np.zeros(len(edges) - 1)
            for j, ((pid, meta), h) in enumerate(zip(bg_procs, hs)):
                hn = h / tot
                lbl = (BG_LABELS.get(meta['name'], meta['name'])
                       if i_panel == 1 else '_nolegend_')
                ax.fill_between(edges, np.r_[bottom, bottom[-1]],
                                np.r_[bottom + hn, (bottom + hn)[-1]],
                                step='post', color=cmap(j % 10), lw=0,
                                alpha=0.8, label=lbl, zorder=1)
                bottom += hn
        else:
            h_bg = norm_hist(obs_bg[key], edges, w=w_bg)
            ax.fill_between(edges, np.r_[h_bg, h_bg[-1]], step='post',
                            color='0.7', alpha=0.55, lw=0, label='background', zorder=1)
        for kv, m in slices:
            h = norm_hist(obs_k[key][m], edges)
            ax.stairs(h, edges, color=colors[kv], lw=1.4,
                      label=rf'$\kappa_3={kv}$', zorder=3)
        ax.set_xlabel(xlbl); ax.set_ylabel('normalized events')
        ax.set_xlim(edges[0], edges[-1])
    if args.split_bg:
        # signal legend inside panel 0; background-process legend as a
        # figure-level strip between the suptitle and the panels, so it
        # never overlaps any histogram
        h0, l0 = axes.flat[0].get_legend_handles_labels()
        sig = [(h, l) for h, l in zip(h0, l0) if 'kappa' in l]
        axes.flat[0].legend([h for h, _ in sig], [l for _, l in sig],
                            frameon=False, fontsize=9)
        h1, l1 = axes.flat[1].get_legend_handles_labels()
        bgh = [(h, l) for h, l in zip(h1, l1) if 'kappa' not in l]
        fig.legend([h for h, _ in bgh], [l for _, l in bgh],
                   frameon=False, fontsize=9, ncol=4,
                   loc='upper center', bbox_to_anchor=(0.5, 0.955))
    else:
        axes.flat[0].legend(frameon=False, fontsize=9)
    lbl = {'3tev': '3 TeV', '10tev': '10 TeV'}.get(cfg['stage'], cfg['stage'])
    fig.suptitle(f'Muon Collider Simulation — √s = {lbl}, resolved', fontsize=11)
    if args.split_bg:
        # leave a horizontal strip for the figure-level process legend
        fig.tight_layout(rect=(0, 0, 1, 0.90))
    else:
        fig.tight_layout()
    stem = 'fig_kinematics_splitbg' if args.split_bg else 'fig_kinematics'
    out = os.path.join(cfg['analysis_dir'], f'{stem}.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print(f'saved {out}')


if __name__ == '__main__':
    main()
