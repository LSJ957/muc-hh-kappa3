#!/usr/bin/env python3
"""04_train_ml1.py — ML1 (signal vs background) train using the tuned HPs from best.json.
Same code drives 3 / 10 TeV; only config differs.
Reads SPANet assignment from <models_dir>/assign_sigbg_main.npy (from 03)."""
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
from tensorflow.keras import optimizers, metrics, callbacks
from sklearn.metrics import roc_auc_score

from lib.config_loader import load_config, resolve_paths
from lib.data_loader   import load_concat
from lib.splits        import make_split_70_15_15
from lib.sample_weights import ml1_sample_weights
from lib import ml_arch as MA
from lib.spanet_engine import recompute_hl_from_assignment

# Default drop list — single source of truth in lib.physics_constants
from lib.physics_constants import DEFAULT_DROP_GLOBALS as _DEFAULT_DROP_GLOBALS
DEFAULT_DROP = list(_DEFAULT_DROP_GLOBALS)


def log(m=''): print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--from-best', default=None,
                    help='HP json (default: cfg.optuna.ml1.from_best)')
    ap.add_argument('--drop', default=','.join(DEFAULT_DROP))
    ap.add_argument('--epochs', type=int, default=None,
                    help='override training.epochs_final from config')
    args = ap.parse_args()

    cfg = load_config(args.config); cfg = resolve_paths(cfg, os.path.join(HERE, os.pardir))
    # prefer freshly-tuned best.json from 04a if present
    local_best = os.path.join(cfg['models_dir'], 'ml1_best.json')
    bp = args.from_best or (local_best if os.path.exists(local_best)
                            else cfg['optuna']['ml1']['from_best'])
    if not os.path.exists(bp):
        # If the placeholder sentinel `__none__` survives (no prior tuning run,
        # no override), give the actionable instruction instead of dumping a
        # confusing path.
        if bp == '__none__' or str(bp).strip().lower() == '__none__':
            sys.exit('ml1_best.json not found.  Run 04a_tune_ml1.py first '
                     '(or `bash run_all.sh <stage> --retune-ml`), or pass '
                     '`--from-best <path/to/ml1_best.json>`.')
        sys.exit(f'best.json not found: {bp}')
    bj = json.load(open(bp))
    hp = dict(bj['hp'])   # 02a/04a/05a all save under key 'hp'
    drop = [d for d in args.drop.split(',') if d]
    kept = MA.kept_globals(drop)
    # Reject silent input-dimension drift: if 04a stored a different DEFAULT_DROP
    # than the user requested at training time, the model in-dims won't match.
    expected_kept = bj.get('kept_globals')
    if expected_kept is not None and sorted(expected_kept) != sorted(kept):
        sys.exit(
            f'kept_globals mismatch:\n'
            f'  best.json kept : {expected_kept}\n'
            f'  current drop   : {drop}  → kept {kept}\n'
            f'Re-run 04a with the new --drop, or pass --drop matching best.json.'
        )
    n_globals = len(kept)
    log(f'=== ML1 TRAIN  stage={cfg["stage"]} ===')
    log(f'HP from {bp}: {hp}')
    log(f'dropping globals {drop} → kept {n_globals}/{len(MA.GLOBALS_NON_TDA)}')

    # ── data ──
    inputs = cfg['ml_usage']['ml1']['sigbg']
    # SPANET_SHARED_SPLIT needs truth_valid for canonical stratification; cost of loading
    # the bool/int8 truth arrays is tiny so we just always load them.
    sb = load_concat(cfg, inputs, load_jets=True, load_truth=True, load_ll_cloud=True)
    # apply SPANet-A pairing from 03_precompute_pairing
    assign_paths = [os.path.join(cfg['models_dir'], f'assign_{nm}.npy') for nm in inputs]
    assign = np.concatenate([np.load(p).astype(np.int8) for p in assign_paths])
    assert len(assign) == sb['N'], f'assign len {len(assign)} != N {sb["N"]}'
    rec = recompute_hl_from_assignment(sb['jets'], assign, sb['hl']['met'], sb['met_phi'])
    for k, v in rec.items():
        sb['hl'][k] = v
    sb['spanet_assignment'] = assign

    tgt = sb['target_sigbg']; nbt = sb['n_btag_total']
    # NOCUT: no pool filter
    apply_btag_feature_mask = False
    # SPANET_SHARED_SPLIT=1 → stratify on (sig_lab*2 + truth_valid) so 02_train_spanet
    # (with same env var set) produces an IDENTICAL 70/15/15 partition, making
    # idx_test truly blind to both SPANet and ML1.
    if os.environ.get('SPANET_SHARED_SPLIT', '0') == '1':
        from lib.splits import canonical_sigbg_strata
        _strata = canonical_sigbg_strata(tgt, sb['truth_valid'])
        sp = make_split_70_15_15(_strata, seed=cfg['training']['seed'])
        log(f'  [shared split] stratify=sig_lab*2+truth_valid (3 strata)')
    else:
        sp = make_split_70_15_15(tgt, seed=cfg['training']['seed'])
    log(f'  pool N={sb["N"]:,}  pos_frac={float(tgt.mean()):.4f}  '
        f'train/val/test = {len(sp["idx_train"]):,}/{len(sp["idx_val"]):,}/{len(sp["idx_test"]):,}')

    # build inputs
    jc, jb = MA.build_jet_tokens(sb["jets"], sb["met_phi"], apply_btag_feature_mask)
    ht     = MA.build_higgs_tokens(sb["jets"], assign, apply_btag_feature_mask)
    gnt    = MA.build_globals_non_tda(sb["hl"], apply_btag_feature_mask, drop=drop)
    gtda   = MA.build_globals_tda(sb['hl'])
    llc    = sb['ll_cloud']
    X_all = dict(jet_cont=jc, jet_btag=jb, higgs_tok=ht,
                 globals_non_tda=gnt, globals_tda=gtda, ll_cloud=llc)
    Xtr = {k: v[sp['idx_train']] for k, v in X_all.items()}
    Xva = {k: v[sp['idx_val']]   for k, v in X_all.items()}
    Xte = {k: v[sp['idx_test']]  for k, v in X_all.items()}
    ytr = tgt[sp['idx_train']].astype(np.float32)
    yva = tgt[sp['idx_val']].astype(np.float32)
    yte = tgt[sp['idx_test']].astype(np.float32)

    # ── model ──
    seed = cfg['training']['seed']
    epochs = args.epochs or cfg['training']['epochs_final']
    m = MA.build_tunable_model(jet_attn=True, hp=hp, seed=seed,
                               n_globals=n_globals, higgs_dim=7)
    # `weighted_metrics` (not `metrics`) so val_auc is computed with the same
    # sample_weight as the loss — EarlyStopping then monitors a quantity that
    # actually matches what we're optimising.  Using `metrics=`
    # alone would compute val_auc unweighted while the loss is weighted, so
    # the "best" checkpoint would be picked on a different criterion than the
    # training objective.
    m.compile(optimizer=optimizers.AdamW(learning_rate=float(hp['lr']),
                                          weight_decay=float(hp['wd'])),
              loss='binary_crossentropy',
              weighted_metrics=[metrics.AUC(name='auc')])
    log(f'  built model, params={m.count_params():,}')
    n_pars = sum(int(tf.size(v).numpy()) for v in m.trainable_weights)
    log(f'  trainable params = {n_pars:,}  (train pool {len(ytr):,} → events/param ratio {len(ytr)/max(n_pars,1):.2f}x)')

    # Per-event physics weights (xsec × BR × LUMI / N_gen, per-class normalised,
    # × class-ratio balance).  Single source: lib.sample_weights — same call
    # used by 04a_tune_ml1.py so Optuna and final-train see the same loss.
    # `n_gen_per_process` comes from the config so a 3rd-party MC set with a
    # different generation budget is weighted correctly.
    sigbg_input = cfg['ml_usage']['ml1']['sigbg'][0]
    n_gen_proc  = cfg['inputs'][sigbg_input].get('n_gen_per_process')
    sample_weight_tr, sample_weight_va, cw_ratio = ml1_sample_weights(
        tgt, sb['target_everytype'], nbt,
        sp['idx_train'], sp['idx_val'], ytr, yva,
        apply_btag_cut=False,
        n_gen_per_process=n_gen_proc)
    log(f'  sample_weight: per-class mean-normalised physics weights × class_weight (ratio={cw_ratio:.2f})')
    log(f'    tr Σw_sig = {sample_weight_tr[ytr>0.5].sum():.2f}  Σw_bg = {sample_weight_tr[ytr<0.5].sum():.2f}')

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

    # ── save ──
    out_keras = os.path.join(cfg['models_dir'], 'ml1.keras')
    out_hist  = os.path.join(cfg['models_dir'], 'ml1_history.npz')
    out_scores = os.path.join(cfg['models_dir'], 'ml1_scores.npz')
    out_meta  = os.path.join(cfg['models_dir'], 'ml1_best.json')
    m.save(out_keras)
    np.savez(out_hist, **{k: np.asarray(v) for k, v in hist.history.items()})
    # test-fold scores for downstream DLL
    pte = m.predict(Xte, batch_size=8192, verbose=0).ravel().astype(np.float32)
    pva = m.predict(Xva, batch_size=8192, verbose=0).ravel().astype(np.float32)
    auc_te = float(roc_auc_score(yte, pte))
    auc_va = float(roc_auc_score(yva, pva))
    log(f'  test AUC = {auc_te:.4f}   val AUC = {auc_va:.4f}')
    np.savez(out_scores,
             d_test=pte, y_test=yte, idx_test=sp['idx_test'],
             d_val=pva,  y_val=yva,  idx_val=sp['idx_val'],
             pool_size=sb['N'], drop=np.array(drop))
    with open(out_meta, 'w') as f:
        json.dump(dict(hp=hp, drop=drop, kept_globals=kept, epochs=epochs,
                       seed=seed, n_params=n_pars, test_auc=auc_te, val_auc=auc_va,
                       events_per_param=len(ytr) / max(n_pars, 1),
                       stage=cfg['stage']), f, indent=2)
    log(f'saved → {out_keras}, {out_hist}, {out_scores}, {out_meta}')


if __name__ == '__main__':
    main()
