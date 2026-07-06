#!/usr/bin/env python3
"""01_extract_features.py — config-driven feature extraction (root → h5).

Reads the inputs block of config/{stage}.yaml.  For each named input it runs
`extract_engine.process_one_root` over every listed .root file (stamping
target_sigbg/target_everytype/kappa3/is_signal as specified), concatenates the
per-file results, and writes one h5 per input via `extract_engine.save_h5`.

Identical code drives 3 TeV and 10 TeV — only the config differs.
Per-root npz cache → re-running is cheap and resumable.  The cache is
validated against the current roots list and labels, but NOT against the
.root file content: if you regenerate a .root in place under the same
name, re-run with --force.

Usage
─────
  python3 src/01_extract_features.py --config config/3tev.yaml          # reuse if h5 exists
  python3 src/01_extract_features.py --config config/3tev.yaml --force  # full re-extract
  python3 src/01_extract_features.py --config config/3tev.yaml --only kappa_scan_main
"""
import os, sys, time, argparse
_LIB = os.environ.get('HHML_CONDA_LIB')
if _LIB is None:
    print('WARNING: HHML_CONDA_LIB not set; skipping LD_LIBRARY_PATH injection. '
          'If TensorFlow/PyTorch fails to load shared libs, '
          'export HHML_CONDA_LIB=/path/to/conda/envs/<env>/lib first.', flush=True)
elif _LIB not in os.environ.get('LD_LIBRARY_PATH', ''):
    os.environ['LD_LIBRARY_PATH'] = _LIB + ':' + os.environ.get('LD_LIBRARY_PATH', '')
    os.execv(sys.executable, [sys.executable] + sys.argv)

import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'lib'))
from lib.config_loader import load_config, resolve_paths
import lib.extract_engine as ee


def log(m=''): print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)


def normalise_list(v, n_roots, default=None, name=''):
    """Broadcast a scalar to length n_roots, or pass through a length-n list."""
    if v is None:
        return [default] * n_roots
    if isinstance(v, list):
        if len(v) != n_roots:
            raise ValueError(f'{name}: list length {len(v)} ≠ roots length {n_roots}')
        return list(v)
    return [v] * n_roots


def cache_path(cache_dir, input_name, root_idx):
    return os.path.join(cache_dir, f'{input_name}__r{root_idx:02d}.npz')


def extract_one_root(root_path, cache_file, label_sigbg, label_everytype, kappa3, is_signal):
    if os.path.exists(cache_file):
        # The cache is keyed by list POSITION (…__rNN.npz); if the roots list
        # was reordered/edited since extraction, the cached events belong to a
        # different file.  Catch that instead of silently mixing samples.
        cached_src = str(np.load(cache_file, allow_pickle=True)['source_root'])
        if cached_src != os.path.basename(root_path):
            raise RuntimeError(
                f'cache {os.path.basename(cache_file)} was extracted from "{cached_src}" '
                f'but the config now lists "{os.path.basename(root_path)}" at this position '
                f'— the roots list changed; re-run with --force')
        log(f'    [cache] {os.path.basename(cache_file)}')
        return
    if not os.path.exists(root_path):
        raise FileNotFoundError(root_path)
    log(f'    process_one_root  {os.path.basename(root_path)}')
    k3val = float(kappa3) if kappa3 is not None else float('nan')
    res = ee.process_one_root(
        root_path,
        label_sigbg=label_sigbg,
        label_everytype=label_everytype,
        kappa3_value=k3val,
        diagram_label=-1,
        is_hh_signal=bool(is_signal),
    )
    if res is None or len(res['hl_dict'][list(res['hl_dict'])[0]]) == 0:
        raise RuntimeError(f'no events survived cuts: {root_path}')
    hl = pd.DataFrame(res['hl_dict'])
    np.savez_compressed(
        cache_file,
        hl_cols       = np.array(list(hl.columns)),
        hl_vals       = hl.values.astype(np.float64),
        ll_cloud      = res['ll_cloud'].astype(np.float32),
        jets          = res['jets_array'].astype(np.float32),
        truth_pairing = res['truth_pairing'].astype(np.int8),
        truth_valid   = res['truth_valid'].astype(bool),
        truth_match_dr= res['truth_match_dr'].astype(np.float32),
        source_root   = np.array(os.path.basename(root_path)),
    )
    n = len(hl); nv = int(res['truth_valid'].sum())
    log(f'      → N={n:,}  truth_valid={nv:,}  ({100*nv/max(n,1):.1f}%)  cached')


def combine_input(input_name, spec, cache_dir, stage, sqrts_TeV,
                  k3_list, sigbg_list, everytype_list):
    out = spec['h5']
    if os.path.exists(out):
        log(f'  [skip combine] {os.path.basename(out)} exists (use --force to overwrite)')
        return
    roots = spec['roots']
    hls, lls, jts, tps, tvs, tdrs, src = [], [], [], [], [], [], []
    cols = None
    for ri, _ in enumerate(roots):
        cf = cache_path(cache_dir, input_name, ri)
        d = np.load(cf, allow_pickle=True)
        cols = list(d['hl_cols'])
        hl_i = pd.DataFrame(d['hl_vals'], columns=cols)
        # Labels (κ3 / process ids) were baked into the cache when it was
        # written; if the config was edited since (without --force), the
        # stamped values disagree with the current registry → corrupted
        # templates downstream.  Validate every file before combining.
        src_i = str(d['source_root'])
        if src_i != os.path.basename(roots[ri]):
            raise RuntimeError(
                f'{input_name} root #{ri}: cache holds "{src_i}" but config lists '
                f'"{os.path.basename(roots[ri])}" — roots list changed; re-run with --force')
        got_k3 = hl_i['kappa3_value'].to_numpy()
        ok_k3 = (np.isnan(got_k3).all() if k3_list[ri] is None
                 else np.allclose(got_k3, float(k3_list[ri]), atol=1e-4))
        ok_sb = bool((hl_i['target_sigbg'].to_numpy() == int(sigbg_list[ri])).all())
        ok_et = bool((hl_i['target_everytype'].to_numpy() == int(everytype_list[ri])).all())
        if not (ok_k3 and ok_sb and ok_et):
            raise RuntimeError(
                f'{input_name} root #{ri} ({src_i}): cached labels disagree with the current '
                f'config (kappa3 ok={ok_k3}, target_sigbg ok={ok_sb}, target_everytype '
                f'ok={ok_et}) — config changed since extraction; re-run with --force')
        hls.append(hl_i)
        lls.append(d['ll_cloud']); jts.append(d['jets'])
        tps.append(d['truth_pairing']); tvs.append(d['truth_valid']); tdrs.append(d['truth_match_dr'])
        src.append(src_i)
    hl_c = pd.concat(hls, ignore_index=True)
    ll_c = np.concatenate(lls); jt_c = np.concatenate(jts)
    tp_c = np.concatenate(tps); tv_c = np.concatenate(tvs); tdr_c = np.concatenate(tdrs)
    meta = {
        'dataset':       input_name,
        'stage':         stage,
        'sqrts_TeV':     sqrts_TeV,
        'source_roots':  '|'.join(src),
        'n_events':      len(hl_c),
        'n_truth_valid': int(tv_c.sum()),
        'schema_version':'pipeline-v1',
    }
    log(f'  combine → {os.path.basename(out)}: {len(hl_c):,} events  (truth_valid={int(tv_c.sum()):,})')
    ee.save_h5(out, hl_c, ll_c, jt_c, tp_c, tv_c, tdr_c, meta_dict=meta)
    log(f'    saved {out}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--force', action='store_true',
                    help='delete any existing per-root cache + output h5 first')
    ap.add_argument('--only', default=None,
                    help='restrict to this single input name (otherwise process all)')
    args = ap.parse_args()

    pipeline_root = os.path.abspath(os.path.join(HERE, os.pardir))
    cfg = load_config(args.config)
    cfg = resolve_paths(cfg, pipeline_root)
    stage = cfg['stage']; sqrts = cfg['sqrts_TeV']
    cache_dir = os.path.join(cfg['data_dir'], '_cache_extract')
    os.makedirs(cache_dir, exist_ok=True)

    targets = [args.only] if args.only else list(cfg['inputs'])
    for nm in targets:
        if nm not in cfg['inputs']:
            sys.exit(f'unknown input "{nm}" — declared: {list(cfg["inputs"])}')

    log(f'=== EXTRACT  stage={stage}  cache={cache_dir} ===')
    for input_name in targets:
        spec = cfg['inputs'][input_name]
        out_h5 = spec['h5']
        if args.force:
            if os.path.exists(out_h5):
                os.remove(out_h5); log(f'[--force] removed {out_h5}')
            for ri in range(len(spec['roots'])):
                cf = cache_path(cache_dir, input_name, ri)
                if os.path.exists(cf):
                    os.remove(cf)
        elif os.path.exists(out_h5):
            log(f'[skip] {os.path.basename(out_h5)} exists  (use --force to re-extract)')
            continue

        log(f'\n>>> {input_name}  ({len(spec["roots"])} root file(s) → {os.path.basename(out_h5)})')
        n_r = len(spec['roots'])
        k3   = normalise_list(spec.get('kappa3'),           n_r, None,  f'{input_name}.kappa3')
        ls   = normalise_list(spec.get('target_sigbg'),     n_r, 0,     f'{input_name}.target_sigbg')
        lev  = normalise_list(spec.get('target_everytype'), n_r, 0,     f'{input_name}.target_everytype')
        sig  = normalise_list(spec.get('is_signal'),        n_r, False, f'{input_name}.is_signal')
        for ri, rp in enumerate(spec['roots']):
            extract_one_root(rp, cache_path(cache_dir, input_name, ri),
                             label_sigbg=int(ls[ri]),
                             label_everytype=int(lev[ri]),
                             kappa3=k3[ri],
                             is_signal=bool(sig[ri]))
        combine_input(input_name, spec, cache_dir, stage, sqrts,
                      k3_list=k3, sigbg_list=ls, everytype_list=lev)

    log('\n=== EXTRACT DONE ===')


if __name__ == '__main__':
    main()
