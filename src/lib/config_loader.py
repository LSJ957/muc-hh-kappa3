"""YAML config loader + validator.

Config schema (per stage)
─────────────────────────
stage:           '3tev' or '10tev'
sqrts_TeV:       3.0 or 10.0
data_dir:        relative/absolute path where extracted h5s go
models_dir:      relative/absolute path for model weights
analysis_dir:    relative/absolute path for FI/correlation/history outputs
dll_dir:         relative/absolute path for DLL outputs

inputs:                              # named input registry
  <name>:
    h5:        path/to/output.h5     # produced by 01_extract_features
    roots:                           # list of explicitly-named .root files
      - /abs/path/to/file_1.root
      - /abs/path/to/file_2.root
    kappa3:    null | float | list   # ground-truth κ3 to stamp per file
                                       null → use root meta; float → all events get this value;
                                       list of floats (len==len(roots)) → per-file value
    target_sigbg:    1 | 0 | per-file list  # signal=1, bg=0
    target_everytype: 0..7 | per-file list  # subprocess id
    is_signal: bool | per-file list

ml_usage:
  ml1:
    sigbg: [<input names>]           # h5(s) used for ML1 (sig vs bg)
  ml2:
    kappa_low:  <float>              # single κ value for class 0
    kappa_high: <float>              # single κ value for class 1
    source: [<input names>]          # h5(s) to draw events from (κ value filter applied at runtime)
  spanet:
    train: [<input names>]           # h5(s) for SPANet training (signal events w/ truth_valid)

dll:
  fit_kappa_grid: [<float list>]     # κ values scanned for the −ΔlnL(κ3) curve
  template_sources: [<input names>]  # h5(s) providing the per-κ signal templates
                                     # (priority order: first listed wins per κ)
  fit_window: [lo, hi]               # OPTIONAL: restrict the poly4 fit to this
                                     # κ sub-range (points outside are scanned
                                     # and plotted but not fitted)
  anchor:
    source: <input name>             # h5 providing the independent κ=1 reference n_A
    kappa: <float>                   # κ value of the reference (usually 1.0)

optuna:
  spanet:
    enabled: true|false
    n_trials: int
    epochs_per_trial: int
    safety_factor: float             # prune trial if n_params × k > N_truth_valid_train
                                     # (uses truth-valid subset, NOT total sigbg). Default 2.0.
  ml1:
    enabled: false                   # default skip (use existing best.json)
    from_best: path/to/best.json     # path used when enabled=false (load HP)
    n_trials: int
    safety_factor: float
  ml2: ... same as ml1

training:
  seed: 42
  split: [0.70, 0.15, 0.15]          # train/val/test
  epochs_final: int                  # both stages
  batch_choices: [int, int]
  ml2_btag_cut: int                  # n_btag_total >= this for ML2 pool
"""
import os
import yaml


def load_config(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f'config not found: {path}')
    with open(path) as f:
        cfg = yaml.safe_load(f)
    _validate(cfg, path)
    # lib.physics_constants selects LUMI and the κ-xsec tables from the
    # PIPELINE_STAGE env var at import time (default '3tev').  A stage/config
    # mismatch silently mis-weights every yield and skips κ points absent from
    # the wrong table — fail loud here instead.
    try:
        from . import physics_constants as _pc
    except ImportError:                       # lib/ placed on sys.path directly
        import physics_constants as _pc
    if cfg['stage'] != _pc._STAGE:
        raise RuntimeError(
            f"config stage '{cfg['stage']}' != PIPELINE_STAGE '{_pc._STAGE}' — "
            f"export PIPELINE_STAGE={cfg['stage']} before running "
            f"(run_all.sh does this automatically)")
    return cfg


def _validate(cfg: dict, path: str):
    req = ['stage', 'sqrts_TeV', 'data_dir', 'models_dir', 'analysis_dir', 'dll_dir',
           'inputs', 'ml_usage', 'dll', 'optuna', 'training']
    miss = [k for k in req if k not in cfg]
    if miss:
        raise ValueError(f'config {path}: missing top-level keys: {miss}')
    # inputs
    for name, spec in cfg['inputs'].items():
        if 'h5' not in spec or 'roots' not in spec:
            raise ValueError(f'inputs.{name}: needs "h5" and "roots"')
        if not isinstance(spec['roots'], list) or not spec['roots']:
            raise ValueError(f'inputs.{name}.roots must be a non-empty list')
    # ml_usage — top-level and per-head required sub-fields (silent KeyErrors
    # at runtime are user-hostile; fail loud here).
    for key in ('ml1', 'ml2', 'spanet'):
        if key not in cfg['ml_usage']:
            raise ValueError(f'ml_usage.{key} missing')
    _required = {
        'ml1':    ('sigbg',),
        'ml2':    ('source', 'kappa_low', 'kappa_high'),
        'spanet': ('train',),
    }
    for head, fields in _required.items():
        for f in fields:
            if f not in cfg['ml_usage'][head]:
                raise ValueError(f'ml_usage.{head}.{f} missing')
    # dll
    for key in ('fit_kappa_grid', 'template_sources', 'anchor'):
        if key not in cfg['dll']:
            raise ValueError(f'dll.{key} missing')
    for key in ('source', 'kappa'):
        if key not in cfg['dll']['anchor']:
            raise ValueError(f'dll.anchor.{key} missing')
    # name references must exist in inputs
    declared = set(cfg['inputs'].keys())
    for ml_name, ml_spec in cfg['ml_usage'].items():
        for sub in ('sigbg', 'source', 'train'):
            if sub in ml_spec:
                for n in ml_spec[sub]:
                    if n not in declared:
                        raise ValueError(f'ml_usage.{ml_name}.{sub}: "{n}" not in inputs')
    for n in cfg['dll']['template_sources']:
        if n not in declared:
            raise ValueError(f'dll.template_sources: "{n}" not in inputs')
    if cfg['dll']['anchor']['source'] not in declared:
        raise ValueError(f'dll.anchor.source: "{cfg["dll"]["anchor"]["source"]}" not in inputs')


def resolve_paths(cfg: dict, pipeline_root: str) -> dict:
    """Make all output paths absolute, rooted at pipeline_root unless already absolute.
    Also expands ``${data_root}`` in every input's ``roots`` list:
      • value from env var ``HHML_DATA_ROOT``  (preferred)
      • else value of top-level ``data_root`` key in the yaml
      • else leave the placeholder in place and let the file-open fail later
        with a clear "missing data_root" message.
    """
    for key in ('data_dir', 'models_dir', 'analysis_dir', 'dll_dir'):
        p = cfg[key]
        if not os.path.isabs(p):
            cfg[key] = os.path.join(pipeline_root, p)
        os.makedirs(cfg[key], exist_ok=True)

    # ─── data_root substitution for .root paths ───
    env_root = os.environ.get('HHML_DATA_ROOT')
    cfg_root = cfg.get('data_root')
    data_root = env_root or cfg_root
    if data_root:
        for name, spec in cfg['inputs'].items():
            spec['roots'] = [r.replace('${data_root}', data_root) for r in spec['roots']]

    # input h5 paths
    for name, spec in cfg['inputs'].items():
        if not os.path.isabs(spec['h5']):
            spec['h5'] = os.path.join(pipeline_root, spec['h5'])
        os.makedirs(os.path.dirname(spec['h5']), exist_ok=True)
    return cfg
