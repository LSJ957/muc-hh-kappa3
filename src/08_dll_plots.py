#!/usr/bin/env python3
"""08_dll_plots.py — render DLL paper figures (PNG) from 07's npz:
    fig_dll_curve.png    morphing curve + raw per-κ scan ± bootstrap σ + w68 band
    fig_logSB.png        log10(S/B) on the 10×10 (ML1 × ML2) quantile grid"""
import os, sys, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, 'lib'))
import numpy as np
import matplotlib.pyplot as plt
from lib.config_loader import load_config, resolve_paths

C_BY_STAGE = {'3tev':  '#1f77b4', '10tev': '#d62728'}


def render_dll(npz_path, out_path, stage_label, color):
    R = np.load(npz_path, allow_pickle=True)
    kf = R['kfine']; df = R['dllB']; floor = float(df.min())
    sh = df - floor
    w = float(R['w68']); lo = float(R['w68_lo']); hi = float(R['w68_hi'])
    fit_grid = R['fit_grid']; raw = R['raw_vals'] - floor; sig = R['raw_sig']

    fig, a = plt.subplots(figsize=(6.6, 3.6))
    a.plot(kf, sh, '-', color=color, lw=2.0, zorder=4, label='morphing curve')
    a.errorbar(fit_grid, raw, yerr=sig, fmt='o', color=color, mec='k', mew=0.4, ms=5,
               ecolor=color, elinewidth=1.0, capsize=2.2, ls='none', zorder=5,
               label='raw per-κ DLL ± bootstrap σ')
    a.axhline(0.5, color='0.5', ls=':', lw=1.0)
    a.axvline(1.0, color='0.7', lw=0.7, alpha=0.6)
    a.axvspan(lo, hi, color=color, alpha=0.13)
    if stage_label == '3 TeV':
        a.set_xlim(0.2, 2.0); a.set_ylim(-0.15, 2.7)
    else:
        a.set_xlim(0.78, 1.22); a.set_ylim(-0.15, 3.0)
    a.set_xlabel(r'$\kappa_3 = \lambda_3/\lambda_3^{\rm SM}$')
    a.set_ylabel(r'$-\Delta\ln L$')
    a.text(0.97, 0.96, rf'$w_{{68}}={w:.2f}$' + '\n' + rf'$[{lo:.2f},\,{hi:.2f}]$',
           transform=a.transAxes, ha='right', va='top', fontsize=10,
           bbox=dict(boxstyle='round', fc='white', ec=color, lw=0.8))
    a.text(0.03, 0.96, f'Muon Collider Simulation\n√s = {stage_label}\nresolved',
           transform=a.transAxes, ha='left', va='top', fontsize=9, linespacing=1.5)
    a.legend(loc='lower center', fontsize=9, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {out_path}')


def render_logSB(npz_path, out_path, stage_label):
    R = np.load(npz_path, allow_pickle=True)
    # Reconstruct S(κ=1) from the morphing fit coefficients.
    # `morph_coef` has shape (3, Nbins) → α, β, γ per bin in
    #     S(κ) = α + β·(κ-1) + γ·(κ-1)²
    # At κ=1 the (κ-1)-basis reduces to just α — no need to evaluate β/γ
    # (the older `a + b*0 + c*0` form looked like a bug).
    a, _b, _c = R['morph_coef']
    S = np.clip(a, 0.0, None).reshape(10, 10)
    B = R['B'].reshape(10, 10)
    with np.errstate(divide='ignore', invalid='ignore'):
        log_sb = np.log10(np.where((S > 0) & (B > 0), S / np.where(B > 0, B, np.nan), np.nan))
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    cmap = plt.cm.viridis.copy(); cmap.set_bad('0.85')
    im = ax.imshow(np.ma.masked_invalid(log_sb).T, origin='lower', cmap=cmap,
                   vmin=-4, vmax=0, extent=[0, 10, 0, 10], aspect='equal')
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label(r'$\log_{10}(S/B)$  at  $\kappa_3=1$')
    ax.set_xlabel('ML1 bin'); ax.set_ylabel(r'ML2 bin (quantile)')
    ax.set_xticks(range(0, 11, 2)); ax.set_yticks(range(0, 11, 2))
    ax.set_title(f'2D template — Muon Collider Simulation, √s = {stage_label}', fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {out_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    cfg = load_config(args.config); cfg = resolve_paths(cfg, os.path.join(HERE, os.pardir))
    stage = cfg['stage']; lbl = {'3tev': '3 TeV', '10tev': '10 TeV'}[stage]
    color = C_BY_STAGE[stage]
    npz = os.path.join(cfg['dll_dir'], 'dll_morphing.npz')
    render_dll(npz,    os.path.join(cfg['dll_dir'], 'fig_dll_curve.png'), lbl, color)
    render_logSB(npz,  os.path.join(cfg['dll_dir'], 'fig_logSB.png'),     lbl)


if __name__ == '__main__':
    main()
