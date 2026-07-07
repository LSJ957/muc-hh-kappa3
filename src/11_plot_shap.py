#!/usr/bin/env python3
"""11_plot_shap.py — SHAP feature attributions for D_HH and D_κ3 (vertical
beeswarm, most important feature on the left).

Per-jet / per-Higgs streams are collapsed by summing SHAP over the token
axis (SHAP additivity) and colouring by the per-event mean feature value.
The low-level particle cloud is collapsed to one net entry per event; it
mixes different feature kinds, so its dots use the neutral colour instead
of a feature-value colour.

Requires the `shap` package (pip install shap).  GPU inference on a
subsample — a few minutes.  Output: analysis/<stage>/fig_shap.png"""
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
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from lib.config_loader import load_config, resolve_paths
from lib.data_loader import load_input
from lib.spanet_engine import recompute_hl_from_assignment
from lib import ml_arch as MA

N_BG, N_FG = 256, 2000        # SHAP background / foreground sample sizes
JET_FEATS   = ['jet_log_pt', 'jet_eta', 'jet_sin_phi', 'jet_cos_phi',
               'jet_log_m', 'jet_dphi_met']
HIGGS_FEATS = ['H_log_pt', 'H_eta', 'H_sin_phi', 'H_cos_phi', 'H_log_m',
               'H_nbtag', 'H_dR_jj']


def build_X(cfg, meta, input_name, n_events, seed=0):
    d = load_input(cfg, input_name, load_jets=True, load_truth=False, load_ll_cloud=True)
    assign = np.load(os.path.join(cfg['models_dir'], f'assign_{input_name}.npy')).astype(np.int8)
    assert len(assign) == d['N'], 'assign file stale vs h5 — re-run 03'
    rec = recompute_hl_from_assignment(d['jets'], assign, d['hl']['met'], d['met_phi'])
    for k, v in rec.items():
        d['hl'][k] = v
    rng = np.random.RandomState(seed)
    idx = rng.choice(d['N'], size=min(n_events, d['N']), replace=False)
    jc, jb = MA.build_jet_tokens(d['jets'], d['met_phi'], False)
    ht     = MA.build_higgs_tokens(d['jets'], assign, False)
    gnt    = MA.build_globals_non_tda(d['hl'], False, drop=meta['drop'])
    gtda   = MA.build_globals_tda(d['hl'])
    X = [jc[idx], ht[idx], gnt[idx], gtda[idx], d['ll_cloud'][idx]]
    jb_mean = jb.astype(np.float32).mean(axis=0).astype(np.int32)   # (4,)
    names_gnt  = [n for n in MA.GLOBALS_NON_TDA if n not in meta['drop']]
    names_gtda = list(MA.GLOBALS_TDA)
    return X, jb_mean, names_gnt, names_gtda


def collapse(sv, X, names_gnt, names_gtda):
    """dict[name] = (per-event signed SHAP, per-event colour value).
    Stream order matches the 5-input wrapper (jet_btag frozen — the integer
    embedding has no gradient, so it carries no SHAP entry here)."""
    out = {}
    for i, nm in enumerate(JET_FEATS):
        out[nm] = (sv[0][..., i].sum(axis=1), X[0][..., i].mean(axis=1))
    for i, nm in enumerate(HIGGS_FEATS):
        out[nm] = (sv[1][..., i].sum(axis=1), X[1][..., i].mean(axis=1))
    for i, nm in enumerate(names_gnt):
        out[nm] = (sv[2][:, i], X[2][:, i])
    for i, nm in enumerate(names_gtda):
        out[nm] = (sv[3][:, i], X[3][:, i])
    # particle cloud: mixes feature kinds → neutral colour (constant)
    out['particle cloud'] = (sv[4].sum(axis=(1, 2)), np.zeros(len(sv[4])))
    return out


def swarm_y(shaps, row_height=0.40, nbins=100):
    shaps = np.asarray(shaps); N = len(shaps)
    rng_ = np.max(shaps) - np.min(shaps)
    if N == 0 or rng_ < 1e-12:
        return np.zeros(N)
    quant = np.round(nbins * (shaps - np.min(shaps)) / rng_).astype(int)
    inds = np.argsort(quant + np.random.RandomState(0).randn(N) * 1e-6)
    layer, last_bin = 0, -1
    ys = np.zeros(N)
    for k in inds:
        if quant[k] != last_bin:
            layer = 0
        ys[k] = np.ceil(layer / 2) * ((-1) ** layer)
        layer += 1
        last_bin = quant[k]
    return ys * (0.9 * row_height / max(1.0, np.max(np.abs(ys))))


def draw(ax, streams, title, top_n=20):
    order = sorted(streams, key=lambda n: -float(np.mean(np.abs(streams[n][0]))))[:top_n]
    cmap = plt.get_cmap('coolwarm')
    for i, nm in enumerate(order):
        sv, val = streams[nm]
        q_lo, q_hi = np.quantile(val, [0.05, 0.95])
        colour = (np.full(len(val), 0.5) if q_hi - q_lo < 1e-12
                  else np.clip((val - q_lo) / (q_hi - q_lo), 0, 1))
        ax.scatter(i + swarm_y(sv), sv, c=colour, cmap=cmap, vmin=0, vmax=1,
                   s=6.0, alpha=0.75, edgecolors='none', rasterized=True)
    ax.axhline(0, color='0.4', lw=0.7)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=60, ha='right', rotation_mode='anchor', fontsize=8)
    ax.set_ylabel('SHAP value')
    ax.set_title(title, fontsize=10, loc='left')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    cfg = load_config(args.config); cfg = resolve_paths(cfg, os.path.join(HERE, os.pardir))
    M = cfg['models_dir']
    import shap
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import load_model

    fig, axes = plt.subplots(2, 1, figsize=(9.5, 8.0))
    for ax, head, input_name in (
            (axes[0], 'ml1', cfg['ml_usage']['ml1']['sigbg'][0]),
            (axes[1], 'ml2', cfg['dll']['anchor']['source'])):
        meta = json.load(open(os.path.join(M, f'{head}_best.json')))
        model = load_model(os.path.join(M, f'{head}.keras'), compile=False, safe_mode=False)
        X, jb_mean, names_gnt, names_gtda = build_X(cfg, meta, input_name, N_BG + N_FG)
        # 5-input wrapper: the integer b-tag stream feeds an Embedding, whose
        # gradient is undefined — freeze it to its per-jet mean inside the
        # graph and explain only the five float streams.
        jb_const = jb_mean.reshape(1, -1)
        inp = [keras.Input(shape=x.shape[1:], name=f'in{i}') for i, x in enumerate(X)]
        jb_frozen = keras.layers.Lambda(
            lambda t: tf.tile(tf.constant(jb_const), [tf.shape(t)[0], 1]))(inp[0])
        wrap = keras.Model(inp, model([inp[0], jb_frozen, inp[1], inp[2], inp[3], inp[4]]))
        Xbg = [x[:N_BG] for x in X]; Xfg = [x[N_BG:] for x in X]
        explainer = shap.GradientExplainer(wrap, Xbg)
        sv = explainer.shap_values(Xfg)
        sv = [np.asarray(s)[..., 0] if np.asarray(s).shape[-1] == 1 else np.asarray(s)
              for s in sv]
        streams = collapse(sv, Xfg, names_gnt, names_gtda)
        label = {'ml1': r'$\mathcal{D}_{HH}$', 'ml2': r'$\mathcal{D}_{\kappa_3}$'}[head]
        draw(ax, streams, f'{label} — {cfg["stage"]}')
        cb = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap='coolwarm'),
                          ax=ax, pad=0.01, aspect=30)
        cb.set_ticks([0, 1]); cb.set_ticklabels(['low', 'high'])
        cb.set_label('feature value', fontsize=8)
    fig.tight_layout()
    out = os.path.join(cfg['analysis_dir'], 'fig_shap.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print(f'saved {out}')


if __name__ == '__main__':
    main()
