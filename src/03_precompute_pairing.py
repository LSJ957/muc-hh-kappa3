#!/usr/bin/env python3
"""03_precompute_pairing.py — run the trained SPANet on every h5 in the config
and save the chosen jet-pairing assignment as <models_dir>/assign_<input>.npy.
Subsequent ML training reads these to re-pair the assignment-dependent HL fields
in-memory (so re-pairing is fast and doesn't touch the h5)."""
import os, sys, argparse, time
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
import h5py
import torch
from lib.config_loader import load_config, resolve_paths
from lib.jet_features  import transform_6
import lib.spanet_engine as SE


def log(m=''): print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--pt', default=None,
                    help='trained spanet.pt path (default: <models_dir>/spanet.pt)')
    ap.add_argument('--only', default=None, help='restrict to one input name')
    args = ap.parse_args()

    cfg = load_config(args.config); cfg = resolve_paths(cfg, os.path.join(HERE, os.pardir))
    pt = args.pt or os.path.join(cfg['models_dir'], 'spanet.pt')
    if not os.path.exists(pt):
        sys.exit(f'SPANet weights not found at {pt} (run 02 first)')

    ck = torch.load(pt, map_location='cpu', weights_only=False)
    spcfg = ck['cfg']; jet_mean = ck['jet_mean']; jet_std = ck['jet_std']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SE.SPANet(spcfg).to(device); model.load_state_dict(ck['model_state_dict']); model.eval()
    log(f'loaded SPANet ({pt})  device={device}  jet_features={jet_mean.shape[0]}')

    is_B = spcfg.get('version') == 'B'
    targets = [args.only] if args.only else list(cfg['inputs'])
    for nm in targets:
        h5p = cfg['inputs'][nm]['h5']
        out = os.path.join(cfg['models_dir'], f'assign_{nm}.npy')
        if os.path.exists(out):
            # Recompute if the h5 was re-extracted or SPANet retrained after
            # this assignment was written — a same-length stale file would
            # otherwise desync silently (only a length assert guards it
            # downstream).
            out_m = os.path.getmtime(out)
            if out_m >= os.path.getmtime(h5p) and out_m >= os.path.getmtime(pt):
                log(f'  [skip] {nm}: {os.path.basename(out)} up to date')
                continue
            log(f'  [stale] {nm}: {os.path.basename(out)} older than h5/spanet.pt → recompute')
        # Version B (LL constituent cloud) requires loading ll_cloud per file
        # — previously dropped which would silently mis-pair.  Currently the
        # active pipeline is Version A so ll_cloud stays None (review N-2).
        with h5py.File(h5p, 'r') as f:
            jets_raw = f['jets'][:].astype(np.float32)
            ll_cloud = f['ll_cloud'][:].astype(np.float32) if is_B and 'll_cloud' in f else None
        jets6 = transform_6(jets_raw)
        log(f'  {nm}: N={len(jets6):,} → infer assignment'
            + (f'  (LL cloud: {ll_cloud.shape})' if ll_cloud is not None else ''))
        # run_inference normalises internally — pass raw 6-feature jets + the
        # checkpoint's true jet_mean/std (no zeros_like trick).
        _, assign, _ = SE.run_inference(model, jets6, jet_mean=jet_mean, jet_std=jet_std,
                                        device=device, ll_cloud=ll_cloud)
        np.save(out, assign.astype(np.int8))
        log(f'    saved {out}  (assignment hist: {np.bincount(assign, minlength=3).tolist()})')


if __name__ == '__main__':
    main()
