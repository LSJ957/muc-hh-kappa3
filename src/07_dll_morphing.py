#!/usr/bin/env python3
"""07_dll_morphing.py — config-driven P3 leak-free DLL with per-bin κ³-quadratic
morphing.  Reads ML1/ML2 scores (from 04/05), builds the 10×10 (ML1 uniform ×
ML2 quantile) template per κ available in `dll.morph_sources`, fits the per-bin
quadratic, evaluates Asimov DLL on a fine κ grid against `dll.anchor`, also
records raw P3 scatter with MC-bootstrap σ and a per-κ DLL table."""
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
from tensorflow.keras.models import load_model

from lib.config_loader import load_config, resolve_paths
from lib.data_loader   import load_input
from lib.splits        import make_split_70_15_15
from lib.weights       import sigbg_weights, kappa_weights
from lib.quantile      import weighted_quantile_edges, uniform_edges
from lib.histograms    import hist2d
from lib.dll           import asimov_dll, connected_w68_on_fine
from lib import ml_arch as MA
from lib.morphing      import fit_per_bin_quadratic, evaluate as morph_eval
from lib.spanet_engine import recompute_hl_from_assignment
from lib.physics_constants import KAPPA_MATCH_TOL
from lib import physics_constants as pc


def log(m=''): print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)


def _kappa_w_evt(xsec_pb, ngen, leak_resc=1.0):
    """Per-event weight for a κ-scan signal template at analysis luminosity.

    w = σ(κ)[pb] · BR(H→bb)² · 1000 [fb/pb] · LUMI[fb⁻¹] / N_gen · leak_resc.

    Single source of truth for the per-κ template weight (review B-2):
    the inline formula previously duplicated at two sites in this file is
    now derived from this helper; lib.weights.kappa_weights computes the
    same quantity, just on an event array instead of a scalar.
    """
    return xsec_pb * pc.BR_HBB_SQ * 1e3 * pc.LUMI_FB_INV / ngen * leak_resc


def predict_scores(model, hl, jets, assign, met_phi, ll_cloud, drop):
    jc, jb = MA.build_jet_tokens(jets, met_phi, apply_btag_feature_mask=False)
    ht     = MA.build_higgs_tokens(jets, assign, apply_btag_feature_mask=False)
    gnt    = MA.build_globals_non_tda(hl, apply_btag_feature_mask=False, drop=drop)
    gtda   = MA.build_globals_tda(hl)
    X = dict(jet_cont=jc, jet_btag=jb, higgs_tok=ht,
             globals_non_tda=gnt, globals_tda=gtda, ll_cloud=ll_cloud)
    return model.predict(X, batch_size=8192, verbose=0).ravel().astype(np.float32)


def load_and_repair(cfg, input_name):
    """Load h5 + apply SPANet pairing (from 03) + recompute assignment-dependent HL."""
    d = load_input(cfg, input_name, load_jets=True, load_truth=False, load_ll_cloud=True)
    assign = np.load(os.path.join(cfg['models_dir'], f'assign_{input_name}.npy')).astype(np.int8)
    assert len(assign) == d['N']
    rec = recompute_hl_from_assignment(d['jets'], assign, d['hl']['met'], d['met_phi'])
    for k, v in rec.items():
        d['hl'][k] = v
    d['spanet_assignment'] = assign
    return d, assign


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--nboot', type=int, default=300,
                    help='bootstrap reps for raw P3 scatter σ')
    ap.add_argument('--ml2-model', default=None,
                    help='override ml2.keras filename (e.g. ml2_btag_legacy.keras, ml2_nobtag_active.keras)')
    ap.add_argument('--ml2-best',  default=None,
                    help='override ml2_best.json filename (auto-derived from --ml2-model if not given)')
    ap.add_argument('--anchor-source', default=None,
                    help='override cfg.dll.anchor.source (e.g. kappa_scan_main for P1)')
    ap.add_argument('--morph-sources', default=None,
                    help='comma-separated override for cfg.dll.morph_sources')
    ap.add_argument('--apply-btag-cut', choices=['true', 'false'], default=None,
                    help='override the analysis-stage BTAG≥3 cut (default uses the value baked in code)')
    ap.add_argument('--out-suffix', default='',
                    help="append suffix to dll_morphing.npz / dll_per_kappa.md filenames")
    args = ap.parse_args()
    cfg = load_config(args.config); cfg = resolve_paths(cfg, os.path.join(HERE, os.pardir))
    stage = cfg['stage']
    out_dir = cfg['dll_dir']; os.makedirs(out_dir, exist_ok=True)

    # ── load ML1 + ML2 models + their drop sets ──
    ml2_model_file = args.ml2_model or 'ml2.keras'
    ml2_best_file  = args.ml2_best  or (ml2_model_file.replace('.keras', '_best.json')
                                        if ml2_model_file != 'ml2.keras' else 'ml2_best.json')
    ml1_meta = json.load(open(os.path.join(cfg['models_dir'], 'ml1_best.json')))
    ml2_meta = json.load(open(os.path.join(cfg['models_dir'], ml2_best_file)))
    drop_ml1 = ml1_meta['drop']; drop_ml2 = ml2_meta['drop']
    log(f'=== DLL morphing  stage={stage} ===')
    log(f'  ML2 model = {ml2_model_file}  ML2 best = {ml2_best_file}')
    log(f'  ML1: drop={drop_ml1}; ML2: drop={drop_ml2}')

    ml1 = load_model(os.path.join(cfg['models_dir'], 'ml1.keras'), compile=False, safe_mode=False)
    ml2 = load_model(os.path.join(cfg['models_dir'], ml2_model_file), compile=False, safe_mode=False)

    # ── CLI overrides for sources / cut ──
    if args.anchor_source:
        cfg['dll']['anchor']['source'] = args.anchor_source
        log(f'  CLI override: anchor.source = {args.anchor_source}')
    if args.morph_sources:
        cfg['dll']['morph_sources'] = [s.strip() for s in args.morph_sources.split(',')]
        log(f'  CLI override: morph_sources = {cfg["dll"]["morph_sources"]}')
    if args.apply_btag_cut is not None:
        cfg['dll']['_apply_btag_cut_override'] = (args.apply_btag_cut == 'true')
        log(f'  CLI override: apply_btag_cut = {cfg["dll"]["_apply_btag_cut_override"]}')

    # ── 1) Build B (background bin yields from sigbg test fold ×6.67) ──
    sb, _ = load_and_repair(cfg, cfg['ml_usage']['ml1']['sigbg'][0])  # first sigbg input
    tgt = sb['target_sigbg']; nbt = sb['n_btag_total']
    pid = sb['target_everytype']
    sp = make_split_70_15_15(tgt, seed=cfg['training']['seed'])
    test_mask = np.zeros(sb['N'], dtype=bool); test_mask[sp['idx_test']] = True
    resc = sb['N'] / max(int(test_mask.sum()), 1)            # test-fold rescale (~6.67)
    # apply-btag-cut at analysis (default False — adopted setup is "no analysis-stage
    # BTAG cut" since ML2 is now trained without BTAG cut, so an additional analysis-
    # stage cut hurts statistics with no compensating gain.  CLI flag --apply-btag-cut
    # true forces the legacy behaviour for diagnostic runs).
    _apply_btag = cfg['dll'].get('_apply_btag_cut_override', False)
    # Honour config n_gen_per_process so a non-default MC budget is correctly
    # normalised in B (review V-5).
    _sigbg_in = cfg['ml_usage']['ml1']['sigbg'][0]
    _ngen     = cfg['inputs'][_sigbg_in].get('n_gen_per_process')
    w_sb = sigbg_weights(tgt, pid, nbt, _apply_btag, n_gen_per_process=_ngen) * test_mask * resc
    bkg  = test_mask & (tgt == 0)
    log(f'  sigbg N={sb["N"]:,}  test={int(test_mask.sum()):,}  bkg-test={int(bkg.sum()):,}  resc×{resc:.2f}')

    d1_sb = predict_scores(ml1, sb['hl'], sb['jets'], sb['spanet_assignment'],
                           sb['met_phi'], sb['ll_cloud'], drop_ml1)
    d2_sb = predict_scores(ml2, sb['hl'], sb['jets'], sb['spanet_assignment'],
                           sb['met_phi'], sb['ll_cloud'], drop_ml2)

    # binning: ML1 uniform 10; ML2 quantile 10 using bkg+anchor_set events
    anchor_name = cfg['dll']['anchor']['source']; k_anchor = float(cfg['dll']['anchor']['kappa'])
    anc, _ = load_and_repair(cfg, anchor_name)
    sm_mask = np.abs(anc['kappa3_value'] - k_anchor) < KAPPA_MATCH_TOL
    if not sm_mask.any():
        sys.exit(f'anchor "{anchor_name}" has no κ={k_anchor} events')
    # Use anchor source's actual N_gen (kappa_set2: 100k, kappa_scan_500k: 500k, ...)
    anchor_ngen = int(cfg['inputs'][anchor_name].get('n_gen_per_kappa', pc.N_GEN_KAPPA_PER_SLICE))
    w_anc = kappa_weights(anc['kappa3_value'], anc['n_btag_total'], _apply_btag,
                          n_gen_per_kappa=anchor_ngen)
    d1_anc = predict_scores(ml1, anc['hl'], anc['jets'], anc['spanet_assignment'],
                            anc['met_phi'], anc['ll_cloud'], drop_ml1)
    d2_anc = predict_scores(ml2, anc['hl'], anc['jets'], anc['spanet_assignment'],
                            anc['met_phi'], anc['ll_cloud'], drop_ml2)
    e1 = uniform_edges(10)
    e2 = weighted_quantile_edges(np.r_[d2_sb[bkg], d2_anc[sm_mask]],
                                 np.r_[w_sb[bkg],  w_anc[sm_mask]], 10)
    B  = hist2d(d1_sb[bkg], d2_sb[bkg], w_sb[bkg], e1, e2).ravel()
    nA_anchor = hist2d(d1_anc[sm_mask], d2_anc[sm_mask], w_anc[sm_mask], e1, e2).ravel()
    log(f'  B sum = {B.sum():.1f}    anchor (κ={k_anchor}) sum = {nA_anchor.sum():.1f}')

    # ── 2) Build per-κ templates from morph_sources (PRIORITY order: first
    #       listed source wins when multiple sources share a κ value).
    #       Iterate over CANONICAL κ keys from pc.KAPPA3_XSEC_PB to avoid
    #       float32-vs-float64 dict-lookup mismatches. ──
    chosen_src = {}     # canonical κ → src_name (priority pick)
    src_cache = {}      # src_name → dict(d1, d2, k3, nbt)
    src_ngen  = {nm: int(cfg['inputs'][nm].get('n_gen_per_kappa', pc.N_GEN_KAPPA_PER_SLICE))
                 for nm in cfg['dll']['morph_sources']}

    for src_name in cfg['dll']['morph_sources']:
        d, _ = load_and_repair(cfg, src_name)
        d1 = predict_scores(ml1, d['hl'], d['jets'], d['spanet_assignment'],
                            d['met_phi'], d['ll_cloud'], drop_ml1)
        d2 = predict_scores(ml2, d['hl'], d['jets'], d['spanet_assignment'],
                            d['met_phi'], d['ll_cloud'], drop_ml2)
        src_cache[src_name] = dict(d1=d1, d2=d2, k3=d['kappa3_value'],
                                   nbt=d['n_btag_total'])
        # For each canonical κ in xsec table, check if this source has events
        for kf in pc.KAPPA3_XSEC_PB:
            # P3 leak-free: skip the anchor source at the anchor's κ value
            # (otherwise raw template = anchor events → DLL=0 by self-comparison)
            if src_name == anchor_name and abs(kf - k_anchor) < KAPPA_MATCH_TOL:
                continue
            m = pc.kappa_match(d['kappa3_value'], kf)
            if int(m.sum()) < 100:
                continue
            if kf not in chosen_src:    # priority: first source listed wins
                chosen_src[kf] = src_name

    # ── ML2 test-fold per-source row masks (review D-1 leak fix) ─────────
    # ML2 trained on κ ∈ {kappa_low, kappa_high} from ml_usage.ml2.source.
    # A template at those κ values from any ml2.source has data leakage —
    # ML2 has memorised the train/val portion → artificially sharper scores.
    # Recover the exact ML2 test fold (same seed, same concat order, same κ
    # filter, same btag-cut), then restrict ONLY those leaky (src,κ) pairs.
    ml2_sources = list(cfg['ml_usage']['ml2']['source'])
    ml2_klow = float(cfg['ml_usage']['ml2']['kappa_low'])
    ml2_khi  = float(cfg['ml_usage']['ml2']['kappa_high'])
    ml2_train_kappas = (ml2_klow, ml2_khi)
    ml2_btag_cut = int(cfg['training'].get('ml2_btag_cut', -1))
    per_src_test_mask = {nm: None for nm in ml2_sources}
    _offsets = [0]
    _concat_k3, _concat_nbt = [], []
    for nm in ml2_sources:
        if nm in src_cache:
            k3_arr  = src_cache[nm]['k3']
            nbt_arr = src_cache[nm]['nbt']
        else:
            d, _ = load_and_repair(cfg, nm)
            k3_arr  = d['kappa3_value']
            nbt_arr = d['n_btag_total']
        _concat_k3.append(k3_arr); _concat_nbt.append(nbt_arr)
        _offsets.append(_offsets[-1] + len(k3_arr))
        per_src_test_mask[nm] = np.zeros(len(k3_arr), dtype=bool)
    concat_k3  = np.concatenate(_concat_k3)
    concat_nbt = np.concatenate(_concat_nbt)
    m_lo_concat = pc.kappa_match(concat_k3, ml2_klow)
    m_hi_concat = pc.kappa_match(concat_k3, ml2_khi)
    sel_concat = (m_lo_concat | m_hi_concat)
    if ml2_btag_cut >= 0:
        sel_concat = sel_concat & (concat_nbt >= ml2_btag_cut)
    idx_pool = np.where(sel_concat)[0]
    y_pool = np.where(m_hi_concat[sel_concat], 1.0, 0.0).astype(np.float32)
    sp_ml2 = make_split_70_15_15(y_pool, seed=cfg['training']['seed'])
    test_rows_concat = idx_pool[sp_ml2['idx_test']]
    for i, nm in enumerate(ml2_sources):
        lo, hi = _offsets[i], _offsets[i + 1]
        in_range = (test_rows_concat >= lo) & (test_rows_concat < hi)
        per_src_test_mask[nm][test_rows_concat[in_range] - lo] = True
    log(f'  ML2 test-fold recovered: pool N={len(y_pool):,}  test N={len(test_rows_concat):,}'
        f'  (split seed={cfg["training"]["seed"]})')

    # Build per-κ template using chosen source, with per-source N_gen weighting
    uniq_k = sorted(chosen_src)
    Hmat = np.zeros((len(uniq_k), 100))
    for i, kf in enumerate(uniq_k):
        src = chosen_src[kf]
        S = src_cache[src]
        m = pc.kappa_match(S['k3'], kf)
        if _apply_btag:
            m = m & (S['nbt'] >= pc.BTAG_CUT)
        leak_resc = 1.0
        leaked = (src in ml2_sources and
                  any(abs(kf - kk) < KAPPA_MATCH_TOL for kk in ml2_train_kappas))
        if leaked:
            n_full = int(m.sum())
            m_test = m & per_src_test_mask[src]
            n_test = int(m_test.sum())
            if n_test < 100:
                log(f'    ⚠️  κ={kf:.3f} src={src}: only {n_test} ML2 test events; '
                    f'falling back to full pool (template will retain a small leak)')
            else:
                leak_resc = n_full / n_test    # ≈ 6.67 (test fold ≈ 15%)
                m = m_test
        xsec_pb = pc.KAPPA3_XSEC_PB[kf]
        w_evt = _kappa_w_evt(xsec_pb, src_ngen[src], leak_resc=leak_resc)
        wt = np.full(int(m.sum()), w_evt, dtype=np.float64)
        Hmat[i] = hist2d(S['d1'][m], S['d2'][m], wt, e1, e2).ravel()
        tag = f'  test-only×{leak_resc:.2f}' if leaked and leak_resc > 1.0 else ''
        log(f'    κ={kf:.3f}  src={src}  N(BTAG≥{pc.BTAG_CUT})={int(m.sum()):,}  '
            f'N_gen={src_ngen[src]:,}  w·N={Hmat[i].sum():.2f}{tag}')
    uniq_k = np.array(uniq_k, dtype=float)
    log(f'  morphing input: {len(uniq_k)} unique κ points')

    fit = fit_per_bin_quadratic(uniq_k, Hmat)
    log(f'  morphing R² (template fit) = {fit["R2"]:.5f}')

    # ── 3) DLL on fine grid ──
    kfine = np.linspace(uniq_k.min(), uniq_k.max(), 2001)
    dllB = np.array([asimov_dll(nA_anchor + B, morph_eval(fit, k) + B) for k in kfine])
    # restrict w68 read to the user's chosen fit grid (default subset)
    fit_grid = np.array(cfg['dll']['fit_kappa_grid'], dtype=float)
    # Single source of truth: lib.dll.connected_w68_on_fine (formerly inline).
    _w = connected_w68_on_fine(kfine, dllB,
                               fit_range=(fit_grid.min(), fit_grid.max()))
    w68_val = _w['w68_connected']
    lo, hi  = _w['k3_lo'], _w['k3_hi']
    kmin    = _w['k3_min']
    log(f'  w68 = {w68_val:.4f}  κ ∈ [{lo:.3f}, {hi:.3f}]  κ̂={kmin:.3f}  '
        f'DLL_B(κ=1) raw = {dllB[np.argmin(np.abs(kfine-1.0))]:.4f}')

    # ── 4) Raw P3 scatter on fit_grid with bootstrap σ ──
    # For each requested κ, use the SAME priority-chosen source as the
    # morphing template (so the raw DLL is consistent with the smooth curve).
    #
    # NOTE on the bootstrap σ saved below: it resamples the κ-template events
    # only — the anchor template `nA_anchor` and the background `B` are held
    # fixed.  The resulting σ therefore reflects ONLY the per-κ MC-statistics
    # variance of the scanned signal template; it does NOT include the
    # anchor-MC statistics floor, nor any experimental (data-Poisson)
    # uncertainty.  For absolute experimental uncertainty on w_{68}, run
    # Asimov toys instead.  This σ_bootstrap is appropriate for diagnosing
    # MC-stat noise of the raw P3 scatter relative to the smooth morphed curve.
    raw_vals = np.full(len(fit_grid), np.nan)
    raw_sig  = np.full(len(fit_grid), np.nan)
    rng = np.random.default_rng(0)

    def _canon(k):
        """Map a near-κ to its canonical KAPPA3_XSEC_PB key (within tol)."""
        for ck in pc.KAPPA3_XSEC_PB:
            if abs(float(k) - ck) < KAPPA_MATCH_TOL:
                return ck
        return None

    def hist_chosen(k):
        kf = _canon(k)
        if kf is None or kf not in chosen_src:
            return None
        src = chosen_src[kf]
        S = src_cache[src]
        m = pc.kappa_match(S['k3'], kf)
        if _apply_btag:
            m = m & (S['nbt'] >= pc.BTAG_CUT)
        # D-1 leak guard, identical recipe as the per-κ template build
        leak_resc = 1.0
        leaked = (src in ml2_sources and
                  any(abs(kf - kk) < KAPPA_MATCH_TOL for kk in ml2_train_kappas))
        if leaked:
            n_full = int(m.sum())
            m_test = m & per_src_test_mask[src]
            n_test = int(m_test.sum())
            if n_test >= 100:
                leak_resc = n_full / n_test
                m = m_test
        if int(m.sum()) < 100: return None
        xsec_pb = pc.KAPPA3_XSEC_PB[kf]
        w_evt = _kappa_w_evt(xsec_pb, src_ngen[src], leak_resc=leak_resc)
        wt = np.full(int(m.sum()), w_evt, dtype=np.float64)
        return S['d1'][m], S['d2'][m], wt

    for ki, k in enumerate(fit_grid):
        ev = hist_chosen(k)
        if ev is None:
            log(f'    raw scatter: κ={k} unavailable → skip')
            continue
        a, b, w = ev
        h = hist2d(a, b, w, e1, e2).ravel()
        raw_vals[ki] = float(asimov_dll(nA_anchor + B, h + B))
        # bootstrap
        boots = np.zeros(args.nboot)
        N = len(a)
        for bi in range(args.nboot):
            j = rng.integers(0, N, N)
            hb = hist2d(a[j], b[j], w[j], e1, e2).ravel()
            boots[bi] = asimov_dll(nA_anchor + B, hb + B)
        raw_sig[ki] = float(boots.std())

    # ── 5) Save everything ──
    out_npz = os.path.join(out_dir, f'dll_morphing{args.out_suffix}.npz')
    np.savez(out_npz,
             stage=stage, e1=e1, e2=e2, B=B, nA_anchor=nA_anchor,
             morph_coef=fit['coef'], morph_R2=fit['R2'], morph_kappas=uniq_k,
             morph_basis=fit['basis'],   # tag the (κ-1)^[0,1,2] convention (review V-6)
             kfine=kfine, dllB=dllB,
             fit_grid=fit_grid, raw_vals=raw_vals, raw_sig=raw_sig,
             w68=w68_val, w68_lo=lo, w68_hi=hi, kmin=kmin)
    log(f'  saved {out_npz}')

    # ── 6) per-κ table (markdown) ──
    md = os.path.join(out_dir, f'dll_per_kappa{args.out_suffix}.md')
    with open(md, 'w') as f:
        f.write(f'# DLL per κ — stage={stage}\n\n')
        f.write(f'P3 anchor: source=`{anchor_name}`, κ_anchor={k_anchor}\n')
        f.write(f'morph_R² (template fit) = {fit["R2"]:.5f}\n\n')
        f.write(f'**w68 = {w68_val:.4f}**   κ ∈ [{lo:.3f}, {hi:.3f}]   κ̂ = {kmin:.3f}\n')
        f.write(f'DLL_B(κ=1) raw (un-shifted) = {dllB[np.argmin(np.abs(kfine-1.0))]:.4f}\n\n')
        f.write('| κ | DLL_morphing(κ) | raw P3 DLL | ±σ_bootstrap |\n')
        f.write('|---:|---:|---:|---:|\n')
        for ki, k in enumerate(fit_grid):
            dm = float(np.interp(k, kfine, dllB))
            rv = raw_vals[ki]; rs = raw_sig[ki]
            f.write(f'| {k:.3f} | {dm:.4f} | {rv:.4f} | {rs:.4f} |\n')
    log(f'  saved {md}')


if __name__ == '__main__':
    main()
