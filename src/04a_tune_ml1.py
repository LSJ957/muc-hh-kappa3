#!/usr/bin/env python3
"""04a_tune_ml1.py — Optuna search for ML1 (sig-vs-bg) on the new pipeline data.
Same anti-overparameterisation trick as 02a: build candidate model, count
trainable params, prune if n_params × safety_factor > N_train.  Writes
<models_dir>/ml1_best.json consumed by 04_train_ml1.py."""
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
from tensorflow.keras import optimizers, callbacks, metrics
import optuna

from lib.config_loader import load_config, resolve_paths
from lib.data_loader   import load_concat
from lib.splits        import make_split_70_15_15
from lib import ml_arch as MA
from lib.spanet_engine import recompute_hl_from_assignment
from lib.sample_weights import ml1_sample_weights

# Default drop list — single source of truth in lib.physics_constants
from lib.physics_constants import DEFAULT_DROP_GLOBALS as _DEFAULT_DROP_GLOBALS
DEFAULT_DROP = list(_DEFAULT_DROP_GLOBALS)


def log(m=''): print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)


def sample_hp(trial: optuna.Trial) -> dict:
    d_token       = trial.suggest_categorical('d_token', [16, 32, 48, 64, 96])
    n_heads       = trial.suggest_categorical('n_heads', [2, 4, 8])
    n_jet_layers  = trial.suggest_int('n_jet_layers', 1, 3)
    n_ll_layers   = trial.suggest_int('n_ll_layers',  1, 2)
    ffn_mult      = trial.suggest_categorical('ffn_mult', [2, 4])
    dropout       = trial.suggest_float('dropout', 0.0, 0.30)
    lr            = trial.suggest_float('lr', 1e-4, 5e-3, log=True)
    wd            = trial.suggest_float('wd', 1e-6, 1e-3, log=True)
    head_dims     = trial.suggest_categorical('head_dims',
                       ['64_32_16', '128_64_32', '256_128_64', '384_192_96'])
    batch         = trial.suggest_categorical('batch', [512, 1024])
    if d_token % n_heads != 0:
        raise optuna.TrialPruned(f'd_token={d_token} % n_heads={n_heads} != 0')
    return dict(d_token=d_token, n_heads=n_heads, n_jet_layers=n_jet_layers,
                n_ll_layers=n_ll_layers, ffn_dim=d_token * ffn_mult,
                dropout=dropout, lr=lr, wd=wd, head_dims=head_dims, batch=batch)


def build_inputs_ml1(cfg):
    drop = DEFAULT_DROP
    n_globals = len(MA.kept_globals(drop))
    inputs = cfg['ml_usage']['ml1']['sigbg']
    # truth_valid is loaded for parity with 04 (small cost).
    sb = load_concat(cfg, inputs, load_jets=True, load_truth=True, load_ll_cloud=True)
    assign = np.concatenate([np.load(os.path.join(cfg['models_dir'], f'assign_{nm}.npy'))
                             for nm in inputs]).astype(np.int8)
    assert len(assign) == sb['N'], f'assign len {len(assign)} != N {sb["N"]}'
    rec = recompute_hl_from_assignment(sb['jets'], assign, sb['hl']['met'], sb['met_phi'])
    for k, v in rec.items(): sb['hl'][k] = v
    jc, jb = MA.build_jet_tokens(sb['jets'], sb['met_phi'], False)
    ht     = MA.build_higgs_tokens(sb['jets'], assign, False)
    gnt    = MA.build_globals_non_tda(sb['hl'], False, drop=drop)
    gtda   = MA.build_globals_tda(sb['hl'])
    llc    = sb['ll_cloud']
    y = sb['target_sigbg'].astype(np.float32)
    sp = make_split_70_15_15(y, seed=cfg['training']['seed'])
    X = dict(jet_cont=jc, jet_btag=jb, higgs_tok=ht,
             globals_non_tda=gnt, globals_tda=gtda, ll_cloud=llc)
    Xtr = {k: v[sp['idx_train']] for k, v in X.items()}
    Xva = {k: v[sp['idx_val']]   for k, v in X.items()}
    ytr = y[sp['idx_train']]; yva = y[sp['idx_val']]
    # tune/train loss-function parity: the same physics-
    # weight × class-weight recipe used by 04_train_ml1.py must drive Optuna.
    n_gen_proc = int(cfg['inputs'][inputs[0]]['n_gen_per_process'])
    sw_tr, sw_va, cw_ratio = ml1_sample_weights(
        sb['target_sigbg'], sb['target_everytype'],
        sp['idx_train'], sp['idx_val'], ytr, yva,
        cfg['physics'], n_gen_proc)
    return Xtr, ytr, Xva, yva, sw_tr, sw_va, cw_ratio, n_globals, drop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    cfg = load_config(args.config); cfg = resolve_paths(cfg, os.path.join(HERE, os.pardir))
    o = cfg['optuna']['ml1']
    n_trials   = int(o.get('n_trials', 20))
    epochs_t   = int(o.get('epochs_per_trial', 15))
    safety_fac = float(o.get('safety_factor', 2.0))

    log(f'=== ML1 OPTUNA  stage={cfg["stage"]}  n_trials={n_trials}  epochs/trial={epochs_t} ===')
    Xtr, ytr, Xva, yva, sw_tr, sw_va, cw_ratio, n_globals, drop = build_inputs_ml1(cfg)
    N_train = len(ytr)
    log(f'  train pool {N_train:,}  pos_frac={float(ytr.mean()):.4f}  '
        f'safety_factor={safety_fac} → max_params ≈ {int(N_train / safety_fac):,}')
    log(f'  sample_weight: per-class mean-normalised physics weights × class_weight (ratio={cw_ratio:.2f})')

    def objective(trial):
        hp = sample_hp(trial)
        m = MA.build_tunable_model(True, hp, seed=cfg['training']['seed'],
                                   n_globals=n_globals, higgs_dim=7)
        n_pars = sum(int(tf.size(v).numpy()) for v in m.trainable_weights)
        if n_pars * safety_fac > N_train:
            tf.keras.backend.clear_session(); del m
            raise optuna.TrialPruned(f'n_params={n_pars:,} × {safety_fac} > N_train={N_train:,}')
        # weighted_metrics so val_auc matches the sample_weighted loss surface
        # Without this, Optuna's `best_val_auc` would optimise an
        # unweighted metric while the final-train loss is weighted → tune/train
        # divergence on the model-selection criterion.
        m.compile(optimizer=optimizers.AdamW(learning_rate=hp['lr'], weight_decay=hp['wd']),
                  loss='binary_crossentropy',
                  weighted_metrics=[metrics.AUC(name='auc')])
        es = callbacks.EarlyStopping(monitor='val_auc', mode='max', patience=4,
                                     restore_best_weights=True, verbose=0)
        # sample_weight (not class_weight) so the tune sees the same loss surface
        # the final-train script does.
        hist = m.fit(Xtr, ytr, validation_data=(Xva, yva, sw_va), epochs=epochs_t,
                     batch_size=hp['batch'], sample_weight=sw_tr,
                     callbacks=[es], verbose=0)
        best_val_auc = float(max(hist.history['val_auc']))
        trial.set_user_attr('n_params', n_pars)
        tf.keras.backend.clear_session()
        return best_val_auc

    sampler = optuna.samplers.TPESampler(seed=cfg['training']['seed'])
    # NopPruner: this objective is a single Keras .fit with internal EarlyStopping;
    # there's no per-epoch trial.report()/should_prune() hook to wire MedianPruner to.
    # 02a uses MedianPruner because its loop reports val_assign_acc each epoch.
    pruner  = optuna.pruners.NopPruner()
    study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner)
    t0 = time.time()
    for i in range(n_trials):
        try:
            study.optimize(objective, n_trials=1, gc_after_trial=True, catch=())
        except optuna.TrialPruned as e:
            log(f'  trial {i+1}/{n_trials} pruned: {e}')
        ct = [t for t in study.trials if t.state.name == 'COMPLETE']
        if ct:
            log(f'  trial {i+1}/{n_trials}: best val_auc={study.best_value:.4f} '
                f'(n_params={study.best_trial.user_attrs.get("n_params","?")})')
    log(f'tuning done in {(time.time()-t0)/60:.1f} min')
    completed = [t for t in study.trials if t.state.name == 'COMPLETE']
    if not completed:
        sys.exit(f'No ML1 trial completed; relax safety_factor (={safety_fac}) or HP space.')

    # Reconstruct the PROCESSED hp dict (ffn_dim = d_token * ffn_mult etc.) —
    # study.best_params holds only RAW suggested params (ffn_mult, no ffn_dim).
    # ml_arch.build_tunable_model reads hp['ffn_dim'], so saving raw best_params
    # would KeyError downstream.
    best_hp = sample_hp(study.best_trial)
    out = os.path.join(cfg['models_dir'], 'ml1_best.json')
    with open(out, 'w') as f:
        # key 'hp' — same as 02a/05a.  04_train_ml1.py
        # reads bj['hp']; a KeyError is informative, no triple-fallback.
        json.dump(dict(hp=best_hp, best_val_auc=float(study.best_value),
                       n_params=int(study.best_trial.user_attrs.get('n_params', -1)),
                       safety_factor=safety_fac, n_train=N_train,
                       drop=drop, kept_globals=MA.kept_globals(drop),
                       n_trials_completed=len(completed), stage=cfg['stage']), f, indent=2)
    log(f'wrote {out}')


if __name__ == '__main__':
    main()
