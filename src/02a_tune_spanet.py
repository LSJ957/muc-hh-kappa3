#!/usr/bin/env python3
"""02a_tune_spanet.py — Optuna search for SPANet on 6 transformed jet features,
with an *anti-overparameterisation* constraint: trials whose model exceeds
N_train / safety_factor (configurable) parameters are pruned automatically,
letting Optuna self-restrict to architectures the dataset can support.

Writes models/<stage>/spanet_best.json (consumed by 02_train_spanet.py)."""
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
import torch
import optuna

from lib.config_loader import load_config, resolve_paths
from lib.jet_features  import compute_mean_std
import lib.spanet_engine as SE
sys.path.insert(0, os.path.dirname(__file__))
import importlib.util
spec = importlib.util.spec_from_file_location('m2', os.path.join(HERE, '02_train_spanet.py'))
M2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(M2)


def log(m=''): print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)


def sample_hp(trial: optuna.Trial) -> dict:
    """Wide HP space; the trainable-params cap prunes oversized combos."""
    jet_embed_dim = trial.suggest_categorical('jet_embed_dim', [16, 24, 32, 48, 64, 96, 128])
    n_attn_layers = trial.suggest_int('n_attn_layers', 1, 4)
    n_attn_heads  = trial.suggest_categorical('n_attn_heads', [2, 4, 8])
    dropout       = trial.suggest_float('dropout', 0.0, 0.30)
    lr            = trial.suggest_float('lr', 1e-4, 5e-3, log=True)
    wd            = trial.suggest_float('wd', 1e-6, 1e-3, log=True)
    # heads must divide embed_dim
    if jet_embed_dim % n_attn_heads != 0:
        raise optuna.TrialPruned(f'jet_embed_dim={jet_embed_dim} % n_attn_heads={n_attn_heads} != 0')
    return dict(jet_embed_dim=jet_embed_dim, n_attn_layers=n_attn_layers,
                n_attn_heads=n_attn_heads, dropout=dropout,
                lr=lr, wd=wd)


def objective_factory(cfg_yaml, n_train_global, safety_factor, epochs_trial):
    """Returns objective(trial) that:
       1) builds candidate SPANet, counts trainable params,
       2) prunes if  n_params * safety_factor > n_train_global,
       3) otherwise trains epochs_trial epochs and returns val_assign_acc."""
    cfg = load_config(cfg_yaml); cfg = resolve_paths(cfg, os.path.join(HERE, os.pardir))
    train_inputs = cfg['ml_usage']['spanet']['train']

    # build dataset once outside the trial loop (expensive) — share across trials
    jets6, ycls, yas, tv = M2.build_dataset(cfg, train_inputs)
    jet_mean, jet_std = compute_mean_std(jets6)
    jets6n = ((jets6 - jet_mean) / jet_std).astype(np.float32)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log(f'tuning dataset: N={len(jets6n):,}  truth_valid_frac={tv.mean():.3f}  device={device}')

    def objective(trial):
        hp = sample_hp(trial)
        spcfg = M2.build_cfg_from_hp(hp, epochs=epochs_trial)
        # model construction → param count check
        m = SE.SPANet(spcfg).to(device)
        n_pars = M2.count_trainable(m)
        if n_pars * float(safety_factor) > n_train_global:
            del m
            raise optuna.TrialPruned(
                f'n_params={n_pars:,} × {safety_factor} > N_train={n_train_global:,}'
            )
        # train epochs_trial epochs and report val assign_acc
        tr, va = M2.make_loaders(jets6n, ycls, yas, tv,
                                 batch=1024, seed=cfg['training']['seed'])
        crit = SE.SPANetLoss(spcfg).to(device)
        opt  = torch.optim.AdamW(m.parameters(), lr=spcfg['lr'], weight_decay=spcfg['weight_decay'])
        sch  = SE.get_lr_scheduler(opt, spcfg, n_steps_per_epoch=len(tr))
        best = -1.0
        for ep in range(1, epochs_trial + 1):
            SE.train_one_epoch(m, tr, crit, opt, sch, device)
            mv = SE.evaluate(m, va, crit, device)
            best = max(best, float(mv.get('assign_acc', 0)))
            trial.report(best, ep)
            if trial.should_prune():
                raise optuna.TrialPruned(f'patience prune @ ep{ep}, best={best:.4f}')
        # store n_params as a user attr for diagnostics
        trial.set_user_attr('n_params', n_pars)
        return best
    return objective, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--n-trials', type=int, default=None,
                    help='override cfg.optuna.spanet.n_trials')
    ap.add_argument('--out-suffix', default='',
                    help='append suffix to spanet_best.json filename '
                         "(e.g. '_methodB') to avoid clobbering")
    args = ap.parse_args()
    cfg = load_config(args.config); cfg = resolve_paths(cfg, os.path.join(HERE, os.pardir))
    o = cfg['optuna']['spanet']
    n_trials   = args.n_trials if args.n_trials is not None else int(o.get('n_trials', 30))
    ept        = int(o.get('epochs_per_trial', 20))
    safety_fac = float(o.get('safety_factor', 2.0))

    # The assignment task — which is what we ultimately score — only sees
    # truth_valid events (others get assignment-loss masked). So the
    # capacity-vs-data check MUST use the truth_valid subset, NOT total
    # sigbg.  Using total sigbg silently over-estimates the effective data
    # by ~25× (truth_valid fraction ≈ 3.8 %) and lets over-parameterised
    # models slip through.
    # NOTE: SPANet has its OWN 85/15 train/val split (no test fold) handled by
    # M2.make_loaders — independent of the ML1/ML2 70/15/15 split in
    # `cfg['training']['split']`.  Reading split[1]=0.15 here is therefore not
    # using the ML1/ML2 train fraction — it just happens to give the same
    # 0.15 that M2.make_loaders defaults to.  If you ever change make_loaders's
    # val_frac, change this in lockstep.
    val_frac = float(cfg['training']['split'][1])
    train_inputs = cfg['ml_usage']['spanet']['train']
    n_total = n_tv = 0
    import h5py
    for nm in train_inputs:
        with h5py.File(cfg['inputs'][nm]['h5'], 'r') as f:
            n_total += int(f['hl/target_sigbg'].shape[0])
            n_tv    += int(f['truth_valid'][:].sum())
    # truth_valid × (1 - SPANet val_frac); under SPANET_SHARED_SPLIT=1 the
    # actual train fraction is 0.70, making this cap ~20% loose (conservative
    # direction would require the shared-mode fraction here).
    n_train = int(round(n_tv * (1.0 - val_frac)))
    log(f'safety_factor={safety_fac} on n_train(truth_valid)={n_train:,}  '
        f'(total sigbg = {n_total:,}, truth_valid fraction = {100*n_tv/max(n_total,1):.1f}%)  '
        f'→ max_params ≈ {int(n_train / safety_fac):,}')

    objective, _ = objective_factory(args.config, n_train, safety_fac, ept)

    # Seed torch/numpy so per-trial weight init, dropout and loader shuffling
    # are reproducible (02_train_spanet seeds the final training the same way).
    import torch
    _seed = int(cfg['training']['seed'])
    torch.manual_seed(_seed); torch.cuda.manual_seed_all(_seed); np.random.seed(_seed)
    sampler = optuna.samplers.TPESampler(seed=_seed)
    pruner  = optuna.pruners.MedianPruner(n_warmup_steps=max(1, ept // 4))
    study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner)
    t0 = time.time()
    for i in range(n_trials):
        try:
            study.optimize(objective, n_trials=1, gc_after_trial=True, catch=())
        except optuna.TrialPruned as e:
            log(f'  trial {i+1}/{n_trials} pruned: {e}')
        # Optuna raises ValueError on .best_trial if no trial completed yet.
        try:
            log(f'  trial {i+1}/{n_trials}: best val_assign_acc={study.best_value:.4f} '
                f'(n_params={study.best_trial.user_attrs.get("n_params","?")})')
        except ValueError:
            pass
    log(f'tuning done in {(time.time()-t0)/60:.1f} min')
    completed = [t for t in study.trials if t.state.name == 'COMPLETE']
    if not completed:
        sys.exit(f'Optuna: no trial completed ({len(study.trials)} pruned/failed). '
                 f'Relax safety_factor (currently {safety_fac}) or widen HP space.')

    # save best
    os.makedirs(cfg['models_dir'], exist_ok=True)
    out = os.path.join(cfg['models_dir'], f'spanet_best{args.out_suffix}.json')
    with open(out, 'w') as f:
        json.dump(dict(hp=study.best_params,
                       val_assign_acc=float(study.best_value),
                       n_params=int(study.best_trial.user_attrs.get('n_params', -1)),
                       safety_factor=safety_fac,
                       n_train=n_train,
                       n_trials_total=len(study.trials),
                       n_trials_completed=sum(1 for t in study.trials
                                              if t.state.name == 'COMPLETE'),
                       stage=cfg['stage']), f, indent=2)
    log(f'wrote {out}')


if __name__ == '__main__':
    main()
