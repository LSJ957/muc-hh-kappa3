#!/usr/bin/env python3
"""02_train_spanet.py — train SPANet on 6 transformed jet features
   (log_pt, η, sin_φ, cos_φ, log1p(m/M0), btag).  Same code drives 3 / 10 TeV.

Reads HPs from `<models_dir>/spanet_best.json` (written by 02a_tune_spanet
or via `--from-best`) and retrains the final model from them.  Optuna HP
search is **not** invoked from here — run 02a separately, or use
`run_all.sh --retune-spanet`.  Configures Version A or Version B according
to the `ll_input` HP in the best.json (default A).

Outputs:
  models/<stage>/spanet.pt                    final weights + cfg + jet_mean/std
  models/<stage>/spanet_history.npz           per-epoch metrics (train+val)
  models/<stage>/spanet_best.json             HP used (for traceability)
"""
import os, sys, argparse, json, time, copy
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
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from lib.config_loader import load_config, resolve_paths
from lib.data_loader   import load_input
from lib.jet_features  import transform_6, compute_mean_std, FEATURE_NAMES
import lib.spanet_engine as SE


def log(m=''): print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)


def build_dataset(cfg, train_inputs, ll=False):
    """Concat all training inputs; transform jets to 6-feature; build HH4bDataset.
    `truth_valid` selects rows usable for the assignment loss."""
    parts = []
    for nm in train_inputs:
        parts.append(load_input(cfg, nm, load_jets=True, load_truth=True, load_ll_cloud=ll))
    jets    = np.concatenate([p['jets'] for p in parts]).astype(np.float32)
    sig_lab = np.concatenate([p['target_sigbg'] for p in parts]).astype(np.int64)
    tp      = np.concatenate([p['truth_pairing'] for p in parts]).astype(np.int64)
    tv      = np.concatenate([p['truth_valid']   for p in parts]).astype(bool)
    ll_c    = np.concatenate([p['ll_cloud']      for p in parts]).astype(np.float32) if ll else None
    jets6 = transform_6(jets)                                # (N,4,6)
    return jets6, sig_lab, tp, tv, ll_c


def make_loaders(jets6, sig_lab, tp, tv, ll_c, batch, seed=42, val_frac=0.15):
    """Stratified train/val split for SPANet.

    Two modes — selected at *call time* via the USE_V6_SPLIT env var:

    - **v5 default** (USE_V6_SPLIT not set / "0"):
        SPANet does its own 85/15 train/val split on its joint stratum
        (sig_lab × 2 + truth_valid).  85% train + 15% val, no test fold.
        This is fine on its own but **leaks** into ML1's 15% test fold,
        because ML1 chooses an INDEPENDENT 70/15/15 split: SPANet has seen
        ~85% of ML1's test events during its own training.

    - **v6** (USE_V6_SPLIT="1"):
        SPANet uses the SAME 70/15/15 partition as ML1.  It trains on the
        70% idx_train, validates on the 15% idx_val, and never touches the
        15% idx_test.  ML1's test fold is then blind to BOTH SPANet and
        ML1, eliminating the train→test leak.

    In both modes the stratum is `sig_lab × 2 + truth_valid` so the three
    relevant subpopulations (bg, signal-no-truth, signal-truth-valid) stay
    balanced across folds.
    """
    import os
    use_v6 = os.environ.get('USE_V6_SPLIT', '0') == '1'
    N = len(jets6)
    full = SE.HH4bDataset(jets6, sig_lab, tp, tv, ll_c)
    strata = sig_lab.astype(np.int64) * 2 + tv.astype(np.int64)
    if use_v6:
        # Shared 70/15/15 with ML1 (sigbg_main pool) — pick train+val portion,
        # reserve test as a true held-out fold for the whole pipeline.
        from lib.splits import make_split_70_15_15
        sp = make_split_70_15_15(strata, seed=seed)
        idx_train = sp['idx_train']
        idx_val   = sp['idx_val']
    else:
        idx_train, idx_val = train_test_split(
            np.arange(N), test_size=val_frac, random_state=seed, stratify=strata,
        )
    tr_ds = Subset(full, idx_train.tolist())
    va_ds = Subset(full, idx_val.tolist())
    return DataLoader(tr_ds, batch_size=batch, shuffle=True,  num_workers=0, drop_last=True), \
           DataLoader(va_ds, batch_size=batch, shuffle=False, num_workers=0)


def build_cfg_from_hp(hp: dict, version='A', n_jet=4, epochs=20) -> dict:
    """Translate a flat HP dict (jet_embed_dim, n_attn_heads, n_attn_layers,
    dropout, lr, wd, ll_input) into the cfg dict the engine's SPANet/SPANetLoss/
    get_lr_scheduler classes expect.  Defaults follow the existing SPANet-A."""
    embed = int(hp.get('jet_embed_dim', hp.get('embed_dim', 64)))
    return dict(
        version            = version,
        jet_input_dim      = 6,
        n_jets             = n_jet,
        # encoder
        jet_embed_dim      = embed,
        n_attn_heads       = int(hp.get('n_attn_heads', hp.get('n_heads', 4))),
        n_attn_layers      = int(hp.get('n_attn_layers', hp.get('n_layers', 3))),
        dropout            = float(hp.get('dropout', 0.10)),
        # heads (smaller default — fits under the truth_valid anti-overparam cap)
        assign_hidden      = hp.get('assign_hidden', [64, 32]),
        cls_hidden         = hp.get('cls_hidden', [64, 32]),
        # loss weights
        lambda_assign      = 1.0,
        lambda_cls         = 1.0,
        # mass loss disabled: _compute_true_masses interprets jets[..,0..3] as
        # (pT, η, φ, mass), but training feeds the normalized 6-feature jets
        # whose idx-2/3 are sin_φ/cos_φ — the resulting m_H target is garbage,
        # so its gradient would inject noise into the shared backbone.
        lambda_mass        = 0.0,
        # optimizer / scheduler
        lr                 = float(hp.get('lr', 1e-3)),
        weight_decay       = float(hp.get('wd', 1e-4)),
        lr_schedule        = 'cosine',
        warmup_epochs      = max(1, int(round(float(hp.get('warmup_frac', 0.10)) * epochs))),
        epochs             = epochs,
        batch_size         = int(hp.get('batch', 1024)),
        # LL stream (version B only)
        ll_input           = bool(hp.get('ll_input', False)),
        const_embed_dim    = int(hp.get('const_embed_dim', 32)),
        const_n_heads      = int(hp.get('const_n_heads', 2)),
        const_n_layers     = int(hp.get('const_n_layers', 2)),
        max_const_per_jet  = 10,
    )


def count_trainable(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_and_save(cfg_yaml, hp, out_dir, epochs, batch=1024, device=None):
    cfg = load_config(cfg_yaml); cfg = resolve_paths(cfg, os.path.join(HERE, os.pardir))
    train_inputs = cfg['ml_usage']['spanet']['train']
    log(f'building dataset from inputs: {train_inputs}')
    jets6, ycls, yas, tv, llc = build_dataset(cfg, train_inputs, ll=hp.get('ll_input', False))
    jet_mean, jet_std = compute_mean_std(jets6)
    log(f'jets6 shape={jets6.shape}  feature_names={FEATURE_NAMES}')
    log(f'jet_mean={jet_mean.round(3).tolist()}')
    log(f'jet_std ={jet_std.round(3).tolist()}')
    # standardize
    jets6n = ((jets6 - jet_mean) / jet_std).astype(np.float32)

    tr, va = make_loaders(jets6n, ycls, yas, tv, llc, batch=batch,
                          seed=cfg['training']['seed'])

    spcfg = build_cfg_from_hp(hp, epochs=epochs)
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model = SE.SPANet(spcfg).to(device)
    n_pars = count_trainable(model)
    log(f'SPANet built — jet_embed_dim={spcfg["jet_embed_dim"]}, '
        f'n_attn_layers={spcfg["n_attn_layers"]}, n_attn_heads={spcfg["n_attn_heads"]}, '
        f'dropout={spcfg["dropout"]:.3f}, trainable params={n_pars:,}, device={device}')

    crit = SE.SPANetLoss(spcfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=spcfg['lr'], weight_decay=spcfg['weight_decay'])
    sched = SE.get_lr_scheduler(opt, spcfg, n_steps_per_epoch=len(tr))

    history = []
    best_val_acc = -1.0
    best_state  = None
    best_epoch  = -1
    best_va_m   = None    # metrics dict from the best epoch (paired with best_state)
    for ep in range(1, epochs + 1):
        t0 = time.time()
        tr_m = SE.train_one_epoch(model, tr, crit, opt, sched, device)
        va_m = SE.evaluate(model, va, crit, device)
        history.append(dict(epoch=ep, **{f'tr_{k}': v for k, v in tr_m.items()},
                            **{f'va_{k}': v for k, v in va_m.items()}))
        # the engine returns 'total','cls','assign','mass','auc','assign_acc' — use 'total' (not 'loss')
        log(f'  epoch {ep:>3d}  tr_total={tr_m.get("total",0):.4f}  '
            f'va_total={va_m.get("total",0):.4f}  va_assign_acc={va_m.get("assign_acc",0):.4f}  '
            f'({time.time()-t0:.0f}s)')
        if va_m.get('assign_acc', 0) > best_val_acc:
            best_val_acc = va_m['assign_acc']
            best_epoch   = ep
            best_state   = copy.deepcopy(model.state_dict())
            best_va_m    = dict(va_m)    # snapshot to keep metrics paired with weights (W-7)

    # ── restore best weights before saving (rather than keeping the last epoch's) ──
    if best_state is not None:
        model.load_state_dict(best_state)
        log(f'  restored best weights from epoch {best_epoch} '
            f'(val_assign_acc={best_val_acc:.4f}) before save')

    # Pair val_metrics in the saved files with the best weights — saving the
    # last-epoch va_m next to best_state would be a metadata lie (review W-7).
    saved_val_metrics = best_va_m if best_va_m is not None else va_m

    # final save
    out_pt = os.path.join(out_dir, 'spanet.pt')
    out_hist = os.path.join(out_dir, 'spanet_history.npz')
    out_best = os.path.join(out_dir, 'spanet_best.json')
    os.makedirs(out_dir, exist_ok=True)
    torch.save(dict(model_state_dict=model.state_dict(),
                    cfg=spcfg, jet_mean=jet_mean, jet_std=jet_std,
                    val_metrics=saved_val_metrics, best_epoch=best_epoch, n_params=n_pars,
                    feature_names=FEATURE_NAMES), out_pt)
    np.savez(out_hist, **{k: np.array([h[k] for h in history]) for k in history[0]})
    with open(out_best, 'w') as f:
        json.dump(dict(hp=hp, val_metrics=saved_val_metrics, best_epoch=best_epoch,
                       n_params=n_pars, feature_names=FEATURE_NAMES,
                       stage=cfg['stage']), f, indent=2)
    log(f'saved → {out_pt}  ({n_pars:,} params, best_val_assign_acc={best_val_acc:.4f})')
    return out_pt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--from-best', default=None,
                    help='HP json (default: <models_dir>/spanet_best.json from 02a_tune)')
    ap.add_argument('--epochs', type=int, default=None,
                    help='override training.epochs_final')
    args = ap.parse_args()
    cfg = load_config(args.config); cfg = resolve_paths(cfg, os.path.join(HERE, os.pardir))

    best_path = args.from_best or os.path.join(cfg['models_dir'], 'spanet_best.json')
    if not os.path.exists(best_path):
        sys.exit(f'no HP file at {best_path}; run 02a_tune_spanet.py first or pass --from-best')
    bj = json.load(open(best_path))
    hp = bj['hp'] if 'hp' in bj else bj
    epochs = args.epochs or cfg['training']['epochs_final']
    log(f'=== SPANet TRAIN  stage={cfg["stage"]}  epochs={epochs} ===')
    log(f'HP from {best_path}: {hp}')
    train_and_save(args.config, hp, cfg['models_dir'], epochs)


if __name__ == '__main__':
    main()
