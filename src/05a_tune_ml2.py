#!/usr/bin/env python3
"""05a_tune_ml2.py — Optuna search for ML2 (binary κ_low vs κ_high) with
anti-overparam pruning.  Same recipe as 04a but on the (much smaller) κ-binary
pool.  BTAG cut at training is governed by `training.ml2_btag_cut` in the
stage YAML: default -1 (no cut, current policy 2026-05-27); set ≥0 to enable."""
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
from lib.physics_constants import KAPPA_MATCH_TOL
from lib.sample_weights import ml2_sample_weights

# Default drop list — single source of truth in lib.physics_constants
from lib.physics_constants import DEFAULT_DROP_GLOBALS as _DEFAULT_DROP_GLOBALS
DEFAULT_DROP = list(_DEFAULT_DROP_GLOBALS)


def log(m=''): print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)


def sample_hp(trial: optuna.Trial) -> dict:
    # smaller HP space than ML1 — ML2 pool is ~10-20× smaller
    d_token       = trial.suggest_categorical('d_token', [8, 16, 24, 32, 48])
    n_heads       = trial.suggest_categorical('n_heads', [2, 4])
    n_jet_layers  = trial.suggest_int('n_jet_layers', 1, 2)
    n_ll_layers   = trial.suggest_int('n_ll_layers',  1, 2)
    ffn_mult      = trial.suggest_categorical('ffn_mult', [2, 3, 4])
    dropout       = trial.suggest_float('dropout', 0.05, 0.40)
    lr            = trial.suggest_float('lr', 5e-5, 5e-3, log=True)
    wd            = trial.suggest_float('wd', 1e-5, 5e-3, log=True)
    head_dims     = trial.suggest_categorical('head_dims',
                       ['32_16_8', '64_32_16', '128_64_32'])
    batch         = trial.suggest_categorical('batch', [256, 512, 1024])
    if d_token % n_heads != 0:
        raise optuna.TrialPruned(f'd_token={d_token} %% n_heads={n_heads} != 0')
    return dict(d_token=d_token, n_heads=n_heads, n_jet_layers=n_jet_layers,
                n_ll_layers=n_ll_layers, ffn_dim=d_token * ffn_mult,
                dropout=dropout, lr=lr, wd=wd, head_dims=head_dims, batch=batch)


def build_inputs_ml2(cfg):
    drop = DEFAULT_DROP
    n_globals = len(MA.kept_globals(drop))
    inputs = cfg['ml_usage']['ml2']['source']
    kp = load_concat(cfg, inputs, load_jets=True, load_truth=False, load_ll_cloud=True)
    assign_all = np.concatenate([np.load(os.path.join(cfg['models_dir'], f'assign_{nm}.npy'))
                                 for nm in inputs]).astype(np.int8)
    rec = recompute_hl_from_assignment(kp['jets'], assign_all, kp['hl']['met'], kp['met_phi'])
    for k, v in rec.items(): kp['hl'][k] = v
    k3 = kp['kappa3_value'].astype(np.float64); nbt = kp['n_btag_total']
    k_lo = float(cfg['ml_usage']['ml2']['kappa_low'])
    k_hi = float(cfg['ml_usage']['ml2']['kappa_high'])
    # default -1 = no BTAG cut at training (κ-irrelevant feature; gives ×1.56
    # more data with no loss on BTAG≥3 inference subset). Match 05_train_ml2's default.
    btag_cut = int(cfg['training'].get('ml2_btag_cut', -1))
    m_lo = np.abs(k3 - k_lo) < KAPPA_MATCH_TOL
    m_hi = np.abs(k3 - k_hi) < KAPPA_MATCH_TOL
    sel  = (m_lo | m_hi) if btag_cut < 0 else (m_lo | m_hi) & (nbt >= btag_cut)
    if not sel.any():
        cut_str = '(no BTAG cut)' if btag_cut < 0 else f'BTAG≥{btag_cut}'
        raise RuntimeError(f'ML2 pool empty for κ_low={k_lo}, κ_high={k_hi}, {cut_str}')
    idx = np.where(sel)[0]
    y = np.where(m_hi[sel], 1.0, 0.0).astype(np.float32)
    jc, jb = MA.build_jet_tokens(kp['jets'], kp['met_phi'], False)
    ht     = MA.build_higgs_tokens(kp['jets'], assign_all, False)
    gnt    = MA.build_globals_non_tda(kp['hl'], False, drop=drop)
    gtda   = MA.build_globals_tda(kp['hl'])
    llc    = kp['ll_cloud']
    X = dict(jet_cont=jc[idx], jet_btag=jb[idx], higgs_tok=ht[idx],
             globals_non_tda=gnt[idx], globals_tda=gtda[idx], ll_cloud=llc[idx])
    sp = make_split_70_15_15(y, seed=cfg['training']['seed'])
    Xtr = {k: v[sp['idx_train']] for k, v in X.items()}
    Xva = {k: v[sp['idx_val']]   for k, v in X.items()}
    ytr = y[sp['idx_train']]; yva = y[sp['idx_val']]
    # tune/train loss-function parity (N-7): same per-event physics × class
    # weight recipe as 05_train_ml2.py.
    k3_pool  = k3[idx]
    nbt_pool = nbt[idx]
    sw_tr, sw_va, cw_ratio = ml2_sample_weights(
        k3_pool, nbt_pool, sp['idx_train'], sp['idx_val'], ytr, yva,
        k_lo, k_hi, apply_btag_cut=False)
    return Xtr, ytr, Xva, yva, sw_tr, sw_va, cw_ratio, n_globals, drop, k_lo, k_hi, btag_cut


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    cfg = load_config(args.config); cfg = resolve_paths(cfg, os.path.join(HERE, os.pardir))
    o = cfg['optuna']['ml2']
    n_trials   = int(o.get('n_trials', 20))
    epochs_t   = int(o.get('epochs_per_trial', 25))
    safety_fac = float(o.get('safety_factor', 2.0))

    log(f'=== ML2 OPTUNA  stage={cfg["stage"]}  n_trials={n_trials}  epochs/trial={epochs_t} ===')
    Xtr, ytr, Xva, yva, sw_tr, sw_va, cw_ratio, n_globals, drop, klo, khi, btagc = build_inputs_ml2(cfg)
    N_train = len(ytr)
    log(f'  ML2 train pool {N_train:,}  (κ={klo}/{khi}, BTAG≥{btagc})  '
        f'safety_factor={safety_fac} → max_params ≈ {int(N_train / safety_fac):,}')
    log(f'  sample_weight: per-class mean-normalised κ-weights × class_weight (ratio={cw_ratio:.2f})')

    def objective(trial):
        hp = sample_hp(trial)
        m = MA.build_tunable_model(True, hp, seed=cfg['training']['seed'],
                                   n_globals=n_globals, higgs_dim=7)
        n_pars = sum(int(tf.size(v).numpy()) for v in m.trainable_weights)
        if n_pars * safety_fac > N_train:
            tf.keras.backend.clear_session(); del m
            raise optuna.TrialPruned(f'n_params={n_pars:,} × {safety_fac} > N_train={N_train:,}')
        # weighted_metrics so val_auc matches the sample_weighted loss
        #.  See 04a for the full rationale.
        m.compile(optimizer=optimizers.AdamW(learning_rate=hp['lr'], weight_decay=hp['wd']),
                  loss='binary_crossentropy',
                  weighted_metrics=[metrics.AUC(name='auc')])
        es = callbacks.EarlyStopping(monitor='val_auc', mode='max', patience=6,
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
    study = optuna.create_study(direction='maximize', sampler=sampler)
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
        sys.exit(f'No ML2 trial completed; relax safety_factor (={safety_fac}) or HP space.')

    # Reconstruct processed hp (with ffn_dim, not the raw ffn_mult). See E-1.
    best_hp = sample_hp(study.best_trial)
    out = os.path.join(cfg['models_dir'], 'ml2_best.json')
    with open(out, 'w') as f:
        # key 'hp' — same as 02a/04a.
        json.dump(dict(hp=best_hp, best_val_auc=float(study.best_value),
                       n_params=int(study.best_trial.user_attrs.get('n_params', -1)),
                       safety_factor=safety_fac, n_train=N_train,
                       drop=drop, kept_globals=MA.kept_globals(drop),
                       kappa_low=klo, kappa_high=khi, btag_cut=btagc,
                       n_trials_completed=len(completed), stage=cfg['stage']), f, indent=2)
    log(f'wrote {out}')


if __name__ == '__main__':
    main()
