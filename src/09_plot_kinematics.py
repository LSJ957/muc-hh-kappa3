#!/usr/bin/env python3
"""09_plot_kinematics.py — kinematic distributions: κ3 signal slices vs the
per-process-weighted total background (paper-style, config-driven).

The κ3 slices are the shape classifier's two training points plus κ3 = 1
(read from ml_usage.ml2 and dll.anchor — change the config and the figure
follows).  Background events are weighted per process (physics block).
Output: analysis/<stage>/fig_kinematics.png"""
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

PANELS = [
    ('H_0',     r'$H_0$',             lambda f: f['hl/H_0'][:]),
    ('mHH',     r'$m_{HH}$  [GeV]',   lambda f: f['hl/mHH'][:]),
    ('pT_j1',   r'$p_T(j_1)$  [GeV]', lambda f: f['jets'][:, 0, 0]),
    ('dR_H1H2', r'$\Delta R(H_1,H_2)$', lambda f: f['hl/dR_H1H2'][:]),
]


def norm_hist(vals, edges, w=None):
    h, _ = np.histogram(vals, bins=edges, weights=w)
    area = h.sum() * np.diff(edges)
    return h / np.where(area > 0, area, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
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

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.6))
    for (key, xlbl, _), ax in zip(PANELS, axes.flat):
        finite = np.isfinite(obs_bg[key])
        lo = np.quantile(obs_bg[key][finite], 0.001)
        hi = max(np.quantile(obs_k[key][m], 0.999) for _, m in slices)
        edges = np.linspace(lo, hi, 51)
        h_bg = norm_hist(obs_bg[key], edges, w=w_bg)
        ax.fill_between(edges, np.r_[h_bg, h_bg[-1]], step='post',
                        color='0.7', alpha=0.55, lw=0, label='background', zorder=1)
        for kv, m in slices:
            h = norm_hist(obs_k[key][m], edges)
            ax.stairs(h, edges, color=colors[kv], lw=1.4,
                      label=rf'$\kappa_3={kv}$', zorder=3)
        ax.set_xlabel(xlbl); ax.set_ylabel('normalised')
        ax.set_xlim(edges[0], edges[-1])
    axes.flat[0].legend(frameon=False, fontsize=9)
    lbl = {'3tev': '3 TeV', '10tev': '10 TeV'}.get(cfg['stage'], cfg['stage'])
    fig.suptitle(f'Muon Collider Simulation — √s = {lbl}, resolved', fontsize=11)
    fig.tight_layout()
    out = os.path.join(cfg['analysis_dir'], 'fig_kinematics.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print(f'saved {out}')


if __name__ == '__main__':
    main()
