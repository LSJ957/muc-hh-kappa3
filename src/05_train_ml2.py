#!/usr/bin/env python3
"""05_train_ml2.py — ML2 (binary κ-low vs κ-high) train using HPs from
best.json.  Loads h5(s) listed in ml_usage.ml2.source, filters events by
kappa3_value ∈ {kappa_low, kappa_high} (config).  BTAG pool filter is
controlled by `training.ml2_btag_cut` in the config:
   -1 (default, current policy 2026-05-27) = no cut, train on the full pool;
   ≥0                                      = require n_btag_total ≥ that value.
Trains and saves ml2.keras + best.json + history + scores."""
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
import tensorflow as tf
from tensorflow.keras import optimizers, metrics
from sklearn.metrics import roc_auc_score

from lib.config_loader import load_config, resolve_paths
from lib.data_loader   import load_concat
from lib.splits        import make_split_70_15_15
from lib.sample_weights import ml2_sample_weights
from lib import ml_arch as MA
from lib.spanet_engine import recompute_hl_from_assignment
from lib.physics_constants import KAPPA_MATCH_TOL

# Default drop list — single source of truth in lib.physics_constants
from lib.physics_constants import DEFAULT_DROP_GLOBALS as _DEFAULT_DROP_GLOBALS
DEFAULT_DROP = list(_DEFAULT_DROP_GLOBALS)


def log(m=''): print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--from-best', default=None)
    ap.add_argument('--drop', default=','.join(DEFAULT_DROP))
    ap.add_argument('--epochs', type=int, default=None,
                    help='override training.epochs_final from config')
    args = ap.parse_args()

    cfg = load_config(args.config); cfg = resolve_paths(cfg, os.path.join(HERE, os.pardir))
    local_best = os.path.join(cfg['models_dir'], 'ml2_best.json')
    bp = args.from_best or (local_best if os.path.exists(local_best)
                            else cfg['optuna']['ml2']['from_best'])
    if not os.path.exists(bp):
        if bp == '__none__' or str(bp).strip().lower() == '__none__':
            sys.exit('ml2_best.json not found.  Run 05a_tune_ml2.py first '
                     '(or `bash run_all.sh <stage> --retune-ml`), or pass '
                     '`--from-best <path/to/ml2_best.json>`.')
        sys.exit(f'best.json not found: {bp}')
    bj = json.load(open(bp))
    hp = dict(bj['hp'])   # unified key across 02a/04a/05a
    drop = [d for d in args.drop.split(',') if d]
    kept = MA.kept_globals(drop)
    expected_kept = bj.get('kept_globals')
    if expected_kept is not None and sorted(expected_kept) != sorted(kept):
        sys.exit(
            f'kept_globals mismatch:\n'
            f'  best.json kept : {expected_kept}\n'
            f'  current drop   : {drop}  → kept {kept}\n'
            f'Re-run 05a with the new --drop, or pass --drop matching best.json.'
        )
    n_globals = len(kept)
    k_low  = float(cfg['ml_usage']['ml2']['kappa_low'])
    k_high = float(cfg['ml_usage']['ml2']['kappa_high'])
    # 2026-05-27: default to NO BTAG cut at training (κ-irrelevant feature; gives ×1.56
    # more data with no loss on BTAG≥3 inference subset). Current policy:
    # ml2_btag_cut: -1 in config; set it to ≥0 to restore the legacy BTAG cut.
    btag_cut = int(cfg['training'].get('ml2_btag_cut', -1))
    if btag_cut < 0:
        log(f'=== ML2 TRAIN  stage={cfg["stage"]}  binary κ {k_low} vs {k_high}  (no BTAG cut at training) ===')
    else:
        log(f'=== ML2 TRAIN  stage={cfg["stage"]}  binary κ {k_low} vs {k_high}  BTAG≥{btag_cut} ===')
    log(f'HP from {bp}: {hp}')
    log(f'dropping globals {drop} → kept {n_globals}/{len(MA.GLOBALS_NON_TDA)}')

    # ── data ──
    inputs = cfg['ml_usage']['ml2']['source']
    kp = load_concat(cfg, inputs, load_jets=True, load_truth=False, load_ll_cloud=True)
    assign_paths = [os.path.join(cfg['models_dir'], f'assign_{nm}.npy') for nm in inputs]
    assign_all = np.concatenate([np.load(p).astype(np.int8) for p in assign_paths])
    assert len(assign_all) == kp['N']
    rec = recompute_hl_from_assignment(kp['jets'], assign_all, kp['hl']['met'], kp['met_phi'])
    for k, v in rec.items():
        kp['hl'][k] = v
    kp['spanet_assignment'] = assign_all

    k3 = kp['kappa3_value'].astype(np.float64)
    m_lo = np.abs(k3 - k_low)  < KAPPA_MATCH_TOL
    m_hi = np.abs(k3 - k_high) < KAPPA_MATCH_TOL
    nbt = kp['n_btag_total']
    if btag_cut < 0:
        sel = (m_lo | m_hi)
    else:
        sel = (m_lo | m_hi) & (nbt >= btag_cut)
    if not sel.any():
        sys.exit(f'no events match κ_low={k_low} or κ_high={k_high}')
    idx = np.where(sel)[0]
    y_pool = np.where(m_hi[sel], 1.0, 0.0).astype(np.float32)
    cut_str = f'after BTAG≥{btag_cut}' if btag_cut >= 0 else '(no BTAG cut)'
    log(f'  pool: κ={k_low} {int(m_lo.sum()):,}  κ={k_high} {int(m_hi.sum()):,}  '
        f'{cut_str} → {len(idx):,}')

    sp = make_split_70_15_15(y_pool, seed=cfg['training']['seed'])
    apply_btag_feature_mask = False  # b-tag features visible
    jc, jb = MA.build_jet_tokens(kp["jets"], kp["met_phi"], apply_btag_feature_mask)
    ht     = MA.build_higgs_tokens(kp["jets"], assign_all, apply_btag_feature_mask)
    gnt    = MA.build_globals_non_tda(kp["hl"], apply_btag_feature_mask, drop=drop)
    gtda   = MA.build_globals_tda(kp['hl'])
    llc    = kp['ll_cloud']
    # subset to pool then split
    X_pool = dict(jet_cont=jc[idx], jet_btag=jb[idx], higgs_tok=ht[idx],
                  globals_non_tda=gnt[idx], globals_tda=gtda[idx], ll_cloud=llc[idx])
    Xtr = {k: v[sp['idx_train']] for k, v in X_pool.items()}
    Xva = {k: v[sp['idx_val']]   for k, v in X_pool.items()}
    Xte = {k: v[sp['idx_test']]  for k, v in X_pool.items()}
    ytr = y_pool[sp['idx_train']]; yva = y_pool[sp['idx_val']]; yte = y_pool[sp['idx_test']]
    log(f'  train/val/test = {len(ytr):,}/{len(yva):,}/{len(yte):,}  '
        f'pos_frac train={ytr.mean():.3f}')

    seed = cfg['training']['seed']
    epochs = args.epochs or cfg['training']['epochs_final']
    m = MA.build_tunable_model(jet_attn=True, hp=hp, seed=seed,
                               n_globals=n_globals, higgs_dim=7)
    # weighted_metrics keeps val_auc aligned with the sample_weighted loss
    #.  Same rationale as 04_train_ml1.
    m.compile(optimizer=optimizers.AdamW(learning_rate=float(hp['lr']),
                                          weight_decay=float(hp['wd'])),
              loss='binary_crossentropy',
              weighted_metrics=[metrics.AUC(name='auc')])
    n_pars = sum(int(tf.size(v).numpy()) for v in m.trainable_weights)
    log(f'  trainable params = {n_pars:,}  (train pool {len(ytr):,} → events/param {len(ytr)/max(n_pars,1):.2f}x)')
    if len(ytr) < n_pars:
        log(f'  ⚠️  OVER-PARAMETERISED ({len(ytr):,} events < {n_pars:,} params); reliable training relies on dropout/wd/early-stop')

    # ── per-event sample weights ──
    # Each ML2 class is a single κ slice, so the physics weight
    # σ(κ₃)·BR²·LUMI/N_gen is constant within a class and the per-class mean
    # normalisation in lib.sample_weights maps it to exactly 1: the recipe
    # reduces to plain class balancing.  It is kept for uniformity with ML1
    # (where the per-process weights DO survive the normalisation).
    k3_pool  = kp['kappa3_value'][idx]
    nbt_pool = kp['n_btag_total'][idx]
    sample_weight_tr, sample_weight_va, cw_ratio = ml2_sample_weights(
        k3_pool, nbt_pool, sp['idx_train'], sp['idx_val'], ytr, yva,
        k_low, k_high, apply_btag_cut=False)
    log(f'  sample_weight: per-class mean-normalised κ-weights × class_weight (ratio={cw_ratio:.2f})')
    log(f'    tr Σw(κ={k_high})={sample_weight_tr[ytr>0.5].sum():.2f}  '
        f'Σw(κ={k_low})={sample_weight_tr[ytr<0.5].sum():.2f}')

    # patience=15 lets training overshoot the val-AUC peak by ~15 epochs so
    # the cosine LR schedule can settle, then EarlyStopping cuts before the
    # train-only descent phase becomes visible overfitting in the curves.
    # restore_best_weights=True saves the best-val-AUC checkpoint.
    cbs = MA.make_callbacks(patience=15, cosine_warmup=True,
                             base_lr=float(hp['lr']), epochs=epochs,
                             warmup_frac=0.10)
    hist = m.fit(Xtr, ytr, validation_data=(Xva, yva, sample_weight_va),
                 epochs=epochs, batch_size=int(hp['batch']),
                 sample_weight=sample_weight_tr,
                 callbacks=cbs, verbose=2)

    out_keras = os.path.join(cfg['models_dir'], 'ml2.keras')
    out_hist  = os.path.join(cfg['models_dir'], 'ml2_history.npz')
    out_scores = os.path.join(cfg['models_dir'], 'ml2_scores.npz')
    out_meta  = os.path.join(cfg['models_dir'], 'ml2_best.json')
    m.save(out_keras)
    np.savez(out_hist, **{k: np.asarray(v) for k, v in hist.history.items()})
    pte = m.predict(Xte, batch_size=8192, verbose=0).ravel().astype(np.float32)
    pva = m.predict(Xva, batch_size=8192, verbose=0).ravel().astype(np.float32)
    auc_te = float(roc_auc_score(yte, pte)); auc_va = float(roc_auc_score(yva, pva))
    log(f'  test AUC = {auc_te:.4f}   val AUC = {auc_va:.4f}')
    np.savez(out_scores,
             d_test=pte, y_test=yte, idx_test=idx[sp['idx_test']],
             d_val=pva,  y_val=yva,  idx_val=idx[sp['idx_val']],
             kappa_low=k_low, kappa_high=k_high, btag_cut=btag_cut, drop=np.array(drop))
    with open(out_meta, 'w') as f:
        json.dump(dict(hp=hp, drop=drop, kept_globals=kept, epochs=epochs,
                       seed=seed, n_params=n_pars, test_auc=auc_te, val_auc=auc_va,
                       events_per_param=len(ytr) / max(n_pars, 1),
                       kappa_low=k_low, kappa_high=k_high, btag_cut=btag_cut,
                       stage=cfg['stage']), f, indent=2)
    log(f'saved → {out_keras}, {out_hist}, {out_scores}, {out_meta}')


if __name__ == '__main__':
    main()
