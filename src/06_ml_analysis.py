#!/usr/bin/env python3
"""06_ml_analysis.py — per-stage ML diagnostics:
    train/val loss·AUC vs epoch (detect overfitting gap)
    permutation feature importance for ML1/ML2 on their held-out test fold
    permutation feature importance for SPANet on ALL truth-valid signal
      events (IN-SAMPLE: includes SPANet training events — fine for ranking
      features, do not quote the printed accuracy as out-of-sample)
    HL feature correlation heatmap
Outputs to analysis/<stage>/."""
import os, sys, argparse, json, time
_LIB = os.environ.get('HHML_CONDA_LIB')
if _LIB is None:
    print('WARNING: HHML_CONDA_LIB not set; skipping LD_LIBRARY_PATH injection. '
          'If TensorFlow/PyTorch fails to load shared libs, '
          'export HHML_CONDA_LIB=/path/to/conda/envs/<env>/lib first.', flush=True)
elif _LIB not in os.environ.get('LD_LIBRARY_PATH', ''):
    os.environ['LD_LIBRARY_PATH'] = _LIB + ':' + os.environ.get('LD_LIBRARY_PATH', '')
    os.execv(sys.executable, [sys.executable] + sys.argv)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, 'lib'))
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from tensorflow.keras.models import load_model
import torch
import h5py

from lib.config_loader import load_config, resolve_paths
from lib.data_loader   import load_concat
from lib import ml_arch as MA
from lib.spanet_engine import recompute_hl_from_assignment, SPANet, run_inference
from lib.jet_features  import transform_6


def log(m=''): print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)


# ───────────────────────────────────────────────
# 1) training history plots
# ───────────────────────────────────────────────
def plot_history(npz_path, png_path, title):
    H = np.load(npz_path)
    ep = np.arange(1, len(H[list(H)[0]]) + 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4))
    if 'loss' in H and 'val_loss' in H:
        a1.plot(ep, H['loss'], 'b-', label='train loss')
        a1.plot(ep, H['val_loss'], 'r-', label='val loss')
        a1.set_xlabel('epoch'); a1.set_ylabel('loss'); a1.legend(); a1.grid(alpha=0.3)
    if 'auc' in H and 'val_auc' in H:
        a2.plot(ep, H['auc'], 'b-', label='train AUC')
        a2.plot(ep, H['val_auc'], 'r-', label='val AUC')
        gap = H['auc'] - H['val_auc']
        a2.set_xlabel('epoch'); a2.set_ylabel('AUC')
        a2.legend(); a2.grid(alpha=0.3)
        a2.set_title(f'train/val gap final = {float(gap[-1]):+.3f}', fontsize=9)
    fig.suptitle(title, fontsize=10); fig.tight_layout()
    fig.savefig(png_path, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f'saved {png_path}')


# ───────────────────────────────────────────────
# 2) permutation importance + correlation
# ───────────────────────────────────────────────
def fi_nn(model, X_test, y_test, drop, nreps=3, label=''):
    base = float(roc_auc_score(y_test, model.predict(X_test, batch_size=8192, verbose=0).ravel()))
    rng = np.random.RandomState(0); N = len(y_test)
    kept = MA.kept_globals(drop)
    JC_NAMES = ['jet_log_pt', 'jet_eta', 'jet_sinphi', 'jet_cosphi', 'jet_log_m', 'jet_dphi_met']
    HT_NAMES = ['H_log_pt', 'H_eta', 'H_sin_phi', 'H_cos_phi', 'H_log_m', 'H_nbtag', 'H_dR_jj']

    def perm(modify):
        drops = []
        for _ in range(nreps):
            Xp = {k: v.copy() for k, v in X_test.items()}
            modify(Xp, rng.permutation(N))
            drops.append(base - float(roc_auc_score(y_test,
                model.predict(Xp, batch_size=8192, verbose=0).ravel())))
        return float(np.mean(drops)), float(np.std(drops))

    results = []
    for i, nm in enumerate(kept):
        results.append((nm, *perm(lambda Xp, p, i=i: Xp['globals_non_tda'].__setitem__(
            (slice(None), i), Xp['globals_non_tda'][p, i]))))
    for i, nm in enumerate(MA.GLOBALS_TDA):
        results.append((nm, *perm(lambda Xp, p, i=i: Xp['globals_tda'].__setitem__(
            (slice(None), i), Xp['globals_tda'][p, i]))))
    results.append(('jet_btag(all)', *perm(
        lambda Xp, p: Xp.__setitem__('jet_btag', Xp['jet_btag'][p]))))
    for i, nm in enumerate(JC_NAMES):
        results.append((nm, *perm(lambda Xp, p, i=i: Xp['jet_cont'].__setitem__(
            (slice(None), slice(None), i), Xp['jet_cont'][p][:, :, i]))))
    htN = X_test['higgs_tok'].shape[-1]; htn = HT_NAMES[:htN]
    for i, nm in enumerate(htn):
        results.append((nm, *perm(lambda Xp, p, i=i: Xp['higgs_tok'].__setitem__(
            (slice(None), slice(None), i), Xp['higgs_tok'][p][:, :, i]))))
    results.append(('ll_cloud(all)', *perm(lambda Xp, p: Xp.__setitem__('ll_cloud', Xp['ll_cloud'][p]))))
    results.sort(key=lambda r: r[1], reverse=True)
    print(f'\n[{label}] base AUC={base:.4f}')
    for nm, m, s in results: print(f'  {nm:<22} {m:+.4f} ± {s:.4f}')
    return base, results


def fi_spanet(pt_path, sig_h5, nreps=3):
    ck = torch.load(pt_path, map_location='cpu', weights_only=False)
    spcfg = ck['cfg']; jet_mean = ck['jet_mean']; jet_std = ck['jet_std']
    NF = int(jet_mean.shape[0])
    NAMES = ['log_pt', 'eta', 'sin_phi', 'cos_phi', 'log1p_m_over_M0', 'btag'][:NF]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    m = SPANet(spcfg).to(device); m.load_state_dict(ck['model_state_dict']); m.eval()
    with h5py.File(sig_h5, 'r') as f:
        tv = f['truth_valid'][:].astype(bool)
        jets_raw = f['jets'][tv].astype(np.float32)
        truth = f['truth_pairing'][tv].astype(np.int64)
    jets6 = transform_6(jets_raw)
    jets6n = ((jets6 - jet_mean) / jet_std).astype(np.float32)
    def acc(j):
        # `j` is already standardised — pass identity mean/std to run_inference
        # (which always applies `(j-mean)/std` internally) to avoid
        # double-normalisation in the permutation FI loop.
        _, assign, _ = run_inference(m, j, jet_mean=np.zeros_like(jet_mean),
                                     jet_std=np.ones_like(jet_std), device=device)
        return float((assign == truth).mean())
    base = acc(jets6n); rng = np.random.RandomState(0); N = len(jets6n)
    print(f'\n[SPANet] base pairing acc = {base:.4f} (truth_valid {N:,}; '
          f'in-sample — includes SPANet training events)')
    results = []
    for i, nm in enumerate(NAMES):
        drops = []
        for _ in range(nreps):
            j = jets6n.copy(); p = rng.permutation(N); j[:, :, i] = jets6n[p, :, i]
            drops.append(base - acc(j))
        m_v = float(np.mean(drops)); s_v = float(np.std(drops))
        results.append((nm, m_v, s_v))
        print(f'  {nm:<18} {m_v:+.4f} ± {s_v:.4f}')
    results.sort(key=lambda r: r[1], reverse=True)
    return base, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--nreps', type=int, default=3)
    args = ap.parse_args()
    cfg = load_config(args.config); cfg = resolve_paths(cfg, os.path.join(HERE, os.pardir))
    out = cfg['analysis_dir']; os.makedirs(out, exist_ok=True)
    stage = cfg['stage']
    M = cfg['models_dir']

    # ── training history plots ──
    plot_history(os.path.join(M, 'ml1_history.npz'),
                 os.path.join(out, 'history_ml1.png'), f'ML1 ({stage}) training history')
    plot_history(os.path.join(M, 'ml2_history.npz'),
                 os.path.join(out, 'history_ml2.png'), f'ML2 ({stage}) training history')

    # ── ML1 FI ──
    ml1 = load_model(os.path.join(M, 'ml1.keras'), compile=False, safe_mode=False)
    meta1 = json.load(open(os.path.join(M, 'ml1_best.json')))
    scores1 = np.load(os.path.join(M, 'ml1_scores.npz'))
    drop1 = list(meta1['drop'])
    # rebuild test X from scratch (cheaper than caching)
    sb = load_concat(cfg, cfg['ml_usage']['ml1']['sigbg'],
                     load_jets=True, load_truth=False, load_ll_cloud=True)
    assign = np.concatenate([np.load(os.path.join(M, f'assign_{nm}.npy'))
                             for nm in cfg['ml_usage']['ml1']['sigbg']]).astype(np.int8)
    assert len(assign) == sb['N'], 'assign files stale vs h5 — re-run 03'
    rec = recompute_hl_from_assignment(sb['jets'], assign, sb['hl']['met'], sb['met_phi'])
    for k, v in rec.items(): sb['hl'][k] = v
    jc, jb = MA.build_jet_tokens(sb['jets'], sb['met_phi'], False)
    ht     = MA.build_higgs_tokens(sb['jets'], assign, False)
    gnt    = MA.build_globals_non_tda(sb['hl'], False, drop=drop1)
    gtda   = MA.build_globals_tda(sb['hl'])
    llc    = sb['ll_cloud']
    idx_te = scores1['idx_test']
    X1 = {k: v[idx_te] for k, v in dict(jet_cont=jc, jet_btag=jb, higgs_tok=ht,
                                         globals_non_tda=gnt, globals_tda=gtda,
                                         ll_cloud=llc).items()}
    y1 = scores1['y_test']
    base1, results1 = fi_nn(ml1, X1, y1, drop1, nreps=args.nreps, label=f'ML1 {stage}')
    np.savez(os.path.join(out, 'fi_ml1.npz'),
             names=np.array([r[0] for r in results1], dtype=object),
             delta_auc=np.array([r[1] for r in results1]),
             std=np.array([r[2] for r in results1]), base_auc=base1)

    # ── ML2 FI ──
    ml2 = load_model(os.path.join(M, 'ml2.keras'), compile=False, safe_mode=False)
    meta2 = json.load(open(os.path.join(M, 'ml2_best.json')))
    scores2 = np.load(os.path.join(M, 'ml2_scores.npz'))
    drop2 = list(meta2['drop'])
    kp = load_concat(cfg, cfg['ml_usage']['ml2']['source'],
                     load_jets=True, load_truth=False, load_ll_cloud=True)
    assign_kp = np.concatenate([np.load(os.path.join(M, f'assign_{nm}.npy'))
                                for nm in cfg['ml_usage']['ml2']['source']]).astype(np.int8)
    rec = recompute_hl_from_assignment(kp['jets'], assign_kp, kp['hl']['met'], kp['met_phi'])
    for k, v in rec.items(): kp['hl'][k] = v
    jc2, jb2 = MA.build_jet_tokens(kp['jets'], kp['met_phi'], False)
    ht2      = MA.build_higgs_tokens(kp['jets'], assign_kp, False)
    gnt2     = MA.build_globals_non_tda(kp['hl'], False, drop=drop2)
    gtda2    = MA.build_globals_tda(kp['hl'])
    llc2     = kp['ll_cloud']
    idx_te2 = scores2['idx_test']
    X2 = {k: v[idx_te2] for k, v in dict(jet_cont=jc2, jet_btag=jb2, higgs_tok=ht2,
                                          globals_non_tda=gnt2, globals_tda=gtda2,
                                          ll_cloud=llc2).items()}
    y2 = scores2['y_test']
    base2, results2 = fi_nn(ml2, X2, y2, drop2, nreps=args.nreps, label=f'ML2 {stage}')
    np.savez(os.path.join(out, 'fi_ml2.npz'),
             names=np.array([r[0] for r in results2], dtype=object),
             delta_auc=np.array([r[1] for r in results2]),
             std=np.array([r[2] for r in results2]), base_auc=base2)

    # ── SPANet FI ──
    sig_h5 = cfg['inputs'][cfg['ml_usage']['ml1']['sigbg'][0]]['h5']
    pt = os.path.join(M, 'spanet.pt')
    base_sp, results_sp = fi_spanet(pt, sig_h5, nreps=args.nreps)
    np.savez(os.path.join(out, 'fi_spanet.npz'),
             names=np.array([r[0] for r in results_sp], dtype=object),
             delta_acc=np.array([r[1] for r in results_sp]),
             std=np.array([r[2] for r in results_sp]), base_acc=base_sp)

    # ── HL correlation (on ML1 test fold) ──
    kept = MA.kept_globals(drop1)
    corr_feats = list(kept) + ['H1_pt', 'H1_m', 'H2_pt', 'H2_m', 'H1_nbtag', 'H2_nbtag']
    M_arr = np.stack([sb['hl'][c].astype(np.float64) for c in corr_feats], axis=1)
    C = np.corrcoef(M_arr[idx_te].T)
    np.savez(os.path.join(out, 'corr_hl.npz'), feats=np.array(corr_feats, dtype=object), C=C)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(C, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr_feats))); ax.set_xticklabels(corr_feats, rotation=90, fontsize=8)
    ax.set_yticks(range(len(corr_feats))); ax.set_yticklabels(corr_feats, fontsize=8)
    for i in range(len(corr_feats)):
        for j in range(len(corr_feats)):
            ax.text(j, i, f'{C[i,j]:.2f}', ha='center', va='center', fontsize=6,
                    color='white' if abs(C[i, j]) > 0.5 else 'black')
    fig.colorbar(im, label='Pearson r', fraction=0.045)
    ax.set_title(f'HL feature correlation — {stage}')
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'corr_hl.png'), dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f'saved {os.path.join(out, "corr_hl.png")}')


if __name__ == '__main__':
    main()
