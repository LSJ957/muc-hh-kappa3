"""Config-driven h5 loader.  Takes an input NAME (declared in config.inputs)
plus the resolved-paths config; returns the canonical event dict used by
SPANet, ML1, ML2 and DLL code.  Same loader for 3 TeV and 10 TeV — only the
h5 path changes (resolved from config).

All h5 files written by 01_extract_features.py carry the same schema:
  /hl/<feature>                    1-D arrays of length N
  /jets                            (N, 4, 10) raw jet features
  /ll_cloud                        (N, 40, 4) particle-flow cloud  [pT_frac, Δη, Δφ, type]
  /truth_pairing  /truth_valid     int8 / bool
  /truth_match_dr                  (N, 4) float32
  /meta/                           dataset metadata (HDF5 group)
"""
from __future__ import annotations
import time
import h5py
import numpy as np

# Re-export HL_FEATURES_45 from physics_constants — single source of truth.
from . import physics_constants as _pc
HL_FEATURES_45 = list(_pc.HL_FEATURES_45)



def load_h5(h5_path: str, *, cols_hl=None, load_jets=True, load_truth=True, load_ll_cloud=True) -> dict:
    """Load one h5 produced by 01_extract_features.py."""
    t0 = time.time(); out: dict = {}
    cols_hl = cols_hl or HL_FEATURES_45
    with h5py.File(h5_path, 'r') as f:
        for c in cols_hl:
            out.setdefault('hl', {})[c] = f[f'hl/{c}'][:]
        out['target_everytype'] = f['hl/target_everytype'][:].astype(np.int8)
        out['n_btag_total']     = f['hl/n_btag_total'][:].astype(np.int8)
        out['met_phi']          = f['hl/met_phi'][:].astype(np.float32)
        out['kappa3_value']     = f['hl/kappa3_value'][:].astype(np.float32)
        out['target_sigbg']     = f['hl/target_sigbg'][:].astype(np.int8) \
                                  if 'target_sigbg' in f['hl'] \
                                  else (out['target_everytype'] == 0).astype(np.int8)
        N = len(out['target_everytype'])
        if load_jets:
            out['jets'] = f['jets'][:].astype(np.float32)
        if load_truth:
            out['truth_pairing'] = f['truth_pairing'][:].astype(np.int8)
            out['truth_valid']   = f['truth_valid'][:].astype(bool)
        if load_ll_cloud and 'll_cloud' in f:
            out['ll_cloud'] = f['ll_cloud'][:].astype(np.float32)
    out['N'] = N
    print(f'[load_h5] {h5_path.split("/")[-1]}: N={N:,} in {time.time()-t0:.2f}s')
    return out


def load_input(cfg: dict, input_name: str, **kwargs) -> dict:
    """Load by config name (cfg must be resolve_paths-ed)."""
    if input_name not in cfg['inputs']:
        raise KeyError(f'input "{input_name}" not in config (have: {list(cfg["inputs"])})')
    return load_h5(cfg['inputs'][input_name]['h5'], **kwargs)


def load_concat(cfg: dict, input_names: list[str], **kwargs) -> dict:
    """Load multiple inputs and event-concatenate them into one dict (matching keys)."""
    if isinstance(input_names, str):
        input_names = [input_names]
    parts = [load_input(cfg, n, **kwargs) for n in input_names]
    if len(parts) == 1:
        return parts[0]
    out = {}; keys_arr = ('target_everytype', 'n_btag_total', 'met_phi',
                          'kappa3_value', 'target_sigbg')
    for k in keys_arr:
        out[k] = np.concatenate([p[k] for p in parts])
    for k in ('jets', 'truth_pairing', 'truth_valid', 'll_cloud'):
        if all(k in p for p in parts):
            out[k] = np.concatenate([p[k] for p in parts])
    out['hl'] = {}
    for c in parts[0]['hl']:
        out['hl'][c] = np.concatenate([p['hl'][c] for p in parts])
    out['N'] = sum(p['N'] for p in parts)
    return out

