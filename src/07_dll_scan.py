#!/usr/bin/env python3
"""07_dll_scan.py — config-driven Asimov −ΔlnL(κ3) scan and CL extraction.

For each κ3 on `dll.fit_kappa_grid`, builds the 10×10 (ML1 uniform × ML2
quantile) signal template from `dll.template_sources`, adds the background B
(ML1 test fold, rescaled to the full yield), and evaluates the Asimov −ΔlnL
against the κ3=1 reference spectrum from `dll.anchor`.  The reference is a
statistically independent κ3=1 sample, so the likelihood is never evaluated
against the same events that built the templates.

The scan is then shifted by its κ3=1 value (−ΔlnL(κ3=1)=0 by convention; the
constant shift does not affect the intervals), fitted with a fourth-order
polynomial, and the 68%/95% CL intervals are read as the connected regions
below 0.5/1.92 referenced to the fitted minimum (single-parameter convention).
`dll.fit_window` optionally restricts the polynomial fit to a κ3 sub-range
(e.g. [0.8, 1.2] at 10 TeV, where the refined grid resolves the likelihood
well); points outside the window are still scanned and plotted."""
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
from lib.histograms    import hist2d, fraction_out_of_range
from lib.dll           import asimov_dll, poly4_w68
from lib import ml_arch as MA
from lib.spanet_engine import recompute_hl_from_assignment
from lib.physics_constants import KAPPA_MATCH_TOL
from lib import physics_constants as pc


def log(m=''): print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)


def _kappa_w_evt(xsec_pb, ngen, lumi, resc=1.0):
    """Per-event weight for a κ-scan signal template at analysis luminosity:
    w = σ(κ)[pb] · BR(H→bb)² · 1000 [fb/pb] · L[fb⁻¹] / N_gen · resc."""
    return xsec_pb * pc.BR_HBB_SQ * 1e3 * lumi / ngen * resc


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


def connected_region(coef, k_lo, k_hi, thresh, n_fine=5000):
    """Connected region around the polynomial minimum at a given threshold,
    referenced to the fitted minimum (same algorithm as lib.dll.poly4_w68's
    internal loop, threshold parameterised for the 95% CL re-use)."""
    kf = np.linspace(k_lo, k_hi, n_fine)
    df = np.polyval(coef, kf)
    sh = np.maximum(df - df.min(), 0.0)
    mask = sh < thresh
    if not mask.any():
        return float('nan'), float('nan'), float('nan'), False
    imin = int(sh.argmin())
    lo, hi = imin, imin
    while lo > 0 and mask[lo - 1]: lo -= 1
    while hi < len(mask) - 1 and mask[hi + 1]: hi += 1
    touches = bool(lo == 0 or hi == len(mask) - 1)
    return float(kf[hi] - kf[lo]), float(kf[lo]), float(kf[hi]), touches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    cfg = load_config(args.config); cfg = resolve_paths(cfg, os.path.join(HERE, os.pardir))
    stage = cfg['stage']
    out_dir = cfg['dll_dir']; os.makedirs(out_dir, exist_ok=True)

    # ── load ML1 + ML2 models + their drop sets ──
    ml1_meta = json.load(open(os.path.join(cfg['models_dir'], 'ml1_best.json')))
    ml2_meta = json.load(open(os.path.join(cfg['models_dir'], 'ml2_best.json')))
    drop_ml1 = ml1_meta['drop']; drop_ml2 = ml2_meta['drop']
    log(f'=== DLL scan  stage={stage} ===')
    log(f'  ML1: drop={drop_ml1}; ML2: drop={drop_ml2}')
    ml1 = load_model(os.path.join(cfg['models_dir'], 'ml1.keras'), compile=False, safe_mode=False)
    ml2 = load_model(os.path.join(cfg['models_dir'], 'ml2.keras'), compile=False, safe_mode=False)

    # ── 1) Background B from the ML1 test fold, rescaled to the full yield ──
    # NOTE: assumes a single ml_usage.ml1.sigbg input (as shipped); 04 uses
    # load_concat over the list, so a multi-input setup needs the same here.
    sb, _ = load_and_repair(cfg, cfg['ml_usage']['ml1']['sigbg'][0])
    tgt = sb['target_sigbg']
    pid = sb['target_everytype']
    # Reconstruct the ML1 split with the SAME recipe as 04
    sp = make_split_70_15_15(tgt, seed=cfg['training']['seed'])
    # Cross-check the re-derived ML1 test fold against the indices 04 actually
    # used (saved in ml1_scores.npz).  If the split recipe in 04 ever drifts,
    # B would silently include ML1 training events — fail loud instead.
    _ml1_scores_p = os.path.join(cfg['models_dir'], 'ml1_scores.npz')
    if os.path.exists(_ml1_scores_p):
        _saved = np.load(_ml1_scores_p)['idx_test']
        if not np.array_equal(np.sort(_saved), np.sort(sp['idx_test'])):
            sys.exit('ML1 test-fold reconstruction disagrees with ml1_scores.npz idx_test — '
                     'the 04/07 split recipes have drifted; fix before trusting B')
        log('  ML1 test-fold cross-check vs ml1_scores.npz: OK')
    test_mask = np.zeros(sb['N'], dtype=bool); test_mask[sp['idx_test']] = True
    resc = sb['N'] / max(int(test_mask.sum()), 1)            # test-fold rescale (~6.67)
    _sigbg_in = cfg['ml_usage']['ml1']['sigbg'][0]
    _ngen     = int(cfg['inputs'][_sigbg_in]['n_gen_per_process'])
    w_sb = sigbg_weights(tgt, pid, cfg['physics'], _ngen) * test_mask * resc
    bkg  = test_mask & (tgt == 0)
    log(f'  sigbg N={sb["N"]:,}  test={int(test_mask.sum()):,}  bkg-test={int(bkg.sum()):,}  resc×{resc:.2f}')

    d1_sb = predict_scores(ml1, sb['hl'], sb['jets'], sb['spanet_assignment'],
                           sb['met_phi'], sb['ll_cloud'], drop_ml1)
    d2_sb = predict_scores(ml2, sb['hl'], sb['jets'], sb['spanet_assignment'],
                           sb['met_phi'], sb['ll_cloud'], drop_ml2)

    # ── 2) κ3=1 reference spectrum from the independent sample + binning ──
    anchor_name = cfg['dll']['anchor']['source']; k_anchor = float(cfg['dll']['anchor']['kappa'])
    anc, _ = load_and_repair(cfg, anchor_name)
    sm_mask = np.abs(anc['kappa3_value'] - k_anchor) < KAPPA_MATCH_TOL
    if not sm_mask.any():
        sys.exit(f'reference "{anchor_name}" has no κ={k_anchor} events')
    anchor_ngen = int(cfg['inputs'][anchor_name]['n_gen_per_kappa'])
    w_anc = kappa_weights(anc['kappa3_value'], cfg['physics'], anchor_ngen)
    d1_anc = predict_scores(ml1, anc['hl'], anc['jets'], anc['spanet_assignment'],
                            anc['met_phi'], anc['ll_cloud'], drop_ml1)
    d2_anc = predict_scores(ml2, anc['hl'], anc['jets'], anc['spanet_assignment'],
                            anc['met_phi'], anc['ll_cloud'], drop_ml2)
    e1 = uniform_edges(10)
    e2 = weighted_quantile_edges(np.r_[d2_sb[bkg], d2_anc[sm_mask]],
                                 np.r_[w_sb[bkg],  w_anc[sm_mask]], 10)
    B  = hist2d(d1_sb[bkg], d2_sb[bkg], w_sb[bkg], e1, e2).ravel()
    nA_anchor = hist2d(d1_anc[sm_mask], d2_anc[sm_mask], w_anc[sm_mask], e1, e2).ravel()
    log(f'  B sum = {B.sum():.1f}    κ={k_anchor} reference sum = {nA_anchor.sum():.1f}')

    # ── 3) Score the template sources (PRIORITY order: the first listed
    #       source wins when several sources cover the same κ value). ──
    chosen_src = {}     # canonical κ → src_name (priority pick)
    src_cache = {}      # src_name → dict(d1, d2, k3)
    KXS = {float(k): float(v) for k, v in cfg['physics']['kappa3_xsec_pb'].items()}
    src_ngen  = {nm: int(cfg['inputs'][nm]['n_gen_per_kappa'])
                 for nm in cfg['dll']['template_sources']}
    for src_name in cfg['dll']['template_sources']:
        d, _ = load_and_repair(cfg, src_name)
        d1 = predict_scores(ml1, d['hl'], d['jets'], d['spanet_assignment'],
                            d['met_phi'], d['ll_cloud'], drop_ml1)
        d2 = predict_scores(ml2, d['hl'], d['jets'], d['spanet_assignment'],
                            d['met_phi'], d['ll_cloud'], drop_ml2)
        src_cache[src_name] = dict(d1=d1, d2=d2, k3=d['kappa3_value'])
        for kf in KXS:
            # Keep the κ=1 reference independent: skip the reference source at
            # its own κ (otherwise template = reference events → DLL=0 by
            # self-comparison).
            if src_name == anchor_name and abs(kf - k_anchor) < KAPPA_MATCH_TOL:
                continue
            m = pc.kappa_match(d['kappa3_value'], kf)
            if int(m.sum()) < 100:
                continue
            if kf not in chosen_src:    # priority: first source listed wins
                chosen_src[kf] = src_name

    # ── 4) ML2 test-fold per-source row masks (leakage guard) ──
    # ML2 was trained on κ ∈ {kappa_low, kappa_high} drawn from
    # ml_usage.ml2.source, so a template at those κ values would be evaluated
    # partly on ML2's own training events (artificially sharp scores).
    # Recover the exact ML2 test fold (same seed, same concat order, same κ
    # filter) and restrict ONLY those (src, κ) templates.
    ml2_sources = list(cfg['ml_usage']['ml2']['source'])
    ml2_klow = float(cfg['ml_usage']['ml2']['kappa_low'])
    ml2_khi  = float(cfg['ml_usage']['ml2']['kappa_high'])
    ml2_train_kappas = (ml2_klow, ml2_khi)
    per_src_test_mask = {nm: None for nm in ml2_sources}
    _offsets = [0]
    _concat_k3 = []
    for nm in ml2_sources:
        if nm in src_cache:
            k3_arr = src_cache[nm]['k3']
        else:
            d, _ = load_and_repair(cfg, nm)
            k3_arr = d['kappa3_value']
        _concat_k3.append(k3_arr)
        _offsets.append(_offsets[-1] + len(k3_arr))
        per_src_test_mask[nm] = np.zeros(len(k3_arr), dtype=bool)
    concat_k3 = np.concatenate(_concat_k3)
    m_lo_concat = pc.kappa_match(concat_k3, ml2_klow)
    m_hi_concat = pc.kappa_match(concat_k3, ml2_khi)
    sel_concat = (m_lo_concat | m_hi_concat)
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
    # Cross-check the recipe-duplicated fold against the indices 05 actually
    # used (saved in ml2_scores.npz); recipe drift between 05 and this
    # reconstruction would silently re-open the template leakage.
    _ml2_scores_p = os.path.join(cfg['models_dir'], 'ml2_scores.npz')
    if os.path.exists(_ml2_scores_p):
        _saved = np.load(_ml2_scores_p)['idx_test']
        if not np.array_equal(np.sort(_saved), np.sort(test_rows_concat)):
            sys.exit('ML2 test-fold reconstruction disagrees with ml2_scores.npz idx_test — '
                     'the 05/07 pool recipes have drifted; fix before trusting templates')
        log('  ML2 test-fold cross-check vs ml2_scores.npz: OK')

    # ── 5) Raw per-κ DLL scan on fit_kappa_grid ──
    def _canon(k):
        """Map a near-κ to its canonical KAPPA3_XSEC_PB key (within tol)."""
        for ck in KXS:
            if abs(float(k) - ck) < KAPPA_MATCH_TOL:
                return ck
        return None

    def template(k):
        kf = _canon(k)
        if kf is None or kf not in chosen_src:
            return None
        src = chosen_src[kf]
        S = src_cache[src]
        m = pc.kappa_match(S['k3'], kf)
        leak_resc = 1.0
        leaked = (src in ml2_sources and
                  any(abs(kf - kk) < KAPPA_MATCH_TOL for kk in ml2_train_kappas))
        if leaked:
            n_full = int(m.sum())
            m_test = m & per_src_test_mask[src]
            n_test = int(m_test.sum())
            if n_test < 100:
                log(f'    ⚠️  κ={kf:.3f} src={src}: only {n_test} ML2 test events; '
                    f'falling back to the full pool (template retains a small leak)')
            else:
                # test fold is a uniform random subsample of the κ slice →
                # scaling it to the full-slice yield is unbiased
                leak_resc = n_full / n_test    # ≈ 6.7 (test fold ≈ 15%)
                m = m_test
        if int(m.sum()) < 100:
            return None
        w_evt = _kappa_w_evt(KXS[kf], src_ngen[src],
                             float(cfg['physics']['lumi_fb_inv']), resc=leak_resc)
        wt = np.full(int(m.sum()), w_evt, dtype=np.float64)
        # The d2 quantile edges were calibrated on bkg-test + reference only;
        # extreme-κ template scores can fall outside and hist2d drops them
        # silently — surface any κ-dependent yield loss.
        _foor = fraction_out_of_range(S['d1'][m], S['d2'][m], e1, e2, wt)
        if _foor > 1e-6:
            log(f'    ⚠️  κ={kf:.3f}: {100*_foor:.4f}% of template weight falls outside '
                f'the (d1,d2) bin range and is dropped')
        h = hist2d(S['d1'][m], S['d2'][m], wt, e1, e2).ravel()
        tag = f'  test-only×{leak_resc:.2f}' if leaked and leak_resc > 1.0 else ''
        log(f'    κ={kf:.3f}  src={src}  N={int(m.sum()):,}  '
            f'N_gen={src_ngen[src]:,}  w·N={h.sum():.2f}{tag}')
        return h

    fit_grid = np.array(cfg['dll']['fit_kappa_grid'], dtype=float)
    raw_vals = np.full(len(fit_grid), np.nan)
    S_k1 = None
    for ki, k in enumerate(fit_grid):
        h = template(k)
        if h is None:
            log(f'    κ={k}: no template available → skip')
            continue
        raw_vals[ki] = float(asimov_dll(nA_anchor + B, h + B))
        if abs(float(k) - 1.0) < KAPPA_MATCH_TOL:
            S_k1 = h

    # ── 6) Shift at κ3=1 + fourth-order polynomial fit + CL intervals ──
    i_k1 = int(np.argmin(np.abs(fit_grid - 1.0)))
    if abs(float(fit_grid[i_k1]) - 1.0) >= KAPPA_MATCH_TOL:
        sys.exit('fit_kappa_grid does not contain κ=1 — cannot anchor the scan')
    if not np.isfinite(raw_vals[i_k1]):
        sys.exit('no κ=1 scan value — cannot anchor the scan')
    k1_raw = float(raw_vals[i_k1])
    raw_shifted = raw_vals - k1_raw
    log(f'  raw −ΔlnL(κ=1) = {k1_raw:.4f}  (subtracted from all points)')

    fwin = cfg['dll'].get('fit_window')
    if fwin:
        wlo, whi = float(fwin[0]), float(fwin[1])
        mfit = (fit_grid >= wlo - 1e-9) & (fit_grid <= whi + 1e-9) & np.isfinite(raw_shifted)
        log(f'  fit window κ ∈ [{wlo}, {whi}]  →  {int(mfit.sum())} points enter the fit')
    else:
        mfit = np.isfinite(raw_shifted)
    res  = poly4_w68(fit_grid[mfit], raw_shifted[mfit])
    coef = np.asarray(res['poly_coef'])
    w95, lo95, hi95, open95 = connected_region(coef, fit_grid[mfit].min(),
                                               fit_grid[mfit].max(), 1.92)
    _edge = 'fit-window edge' if fwin else 'scan edge'
    log(f'  w68 = {res["w68_connected"]:.4f}  κ ∈ [{res["k3_lo"]:.3f}, {res["k3_hi"]:.3f}]'
        f'{f"  (interval reaches the {_edge} → open on that side)" if res["touches_boundary"] else ""}')
    log(f'  w95 = {w95:.4f}  κ ∈ [{lo95:.3f}, {hi95:.3f}]'
        f'{f"  (interval reaches the {_edge} → open on that side)" if open95 else ""}')
    log(f'  κ̂ = {res["k3_min"]:.3f}   poly4 R² = {res["r2"]:.4f}')

    # ── 7) Save ──
    out_npz = os.path.join(out_dir, 'dll_scan.npz')
    np.savez(out_npz,
             stage=stage, e1=e1, e2=e2, B=B, nA_anchor=nA_anchor,
             S_k1=(S_k1 if S_k1 is not None else np.full_like(B, np.nan)),
             fit_grid=fit_grid, raw_vals=raw_vals, raw_shifted=raw_shifted,
             k1_raw=k1_raw, fit_window=(np.array(fwin, dtype=float) if fwin else np.array([])),
             poly_coef=coef,
             w68=res['w68_connected'], w68_lo=res['k3_lo'], w68_hi=res['k3_hi'],
             w68_open=res['touches_boundary'],
             w95=w95, w95_lo=lo95, w95_hi=hi95, w95_open=open95,
             kmin=res['k3_min'], r2=res['r2'])
    log(f'  saved {out_npz}')

    md = os.path.join(out_dir, 'dll_per_kappa.md')
    with open(md, 'w') as f:
        f.write(f'# −ΔlnL(κ3) scan — stage={stage}\n\n')
        f.write(f'κ3=1 reference spectrum: source=`{anchor_name}`, κ={k_anchor}\n')
        f.write(f'raw −ΔlnL(κ=1) before the shift = {k1_raw:.4f}\n')
        if fwin:
            f.write(f'polynomial fit window: κ3 ∈ [{fwin[0]}, {fwin[1]}]\n')
        f.write(f'\n**68% CL: {res["k3_lo"]:.3f} < κ3 < {res["k3_hi"]:.3f}**'
                f'   (w68 = {res["w68_connected"]:.4f})\n')
        f.write(f'**95% CL: {lo95:.3f} < κ3 < {hi95:.3f}**'
                f'{"  (interval reaches the fitted range edge → open on that side)" if open95 else ""}'
                f'   (w95 = {w95:.4f})\n\n')
        f.write('| κ3 | −ΔlnL (raw) | −ΔlnL (shifted) | in fit |\n')
        f.write('|---:|---:|---:|:---:|\n')
        for ki, k in enumerate(fit_grid):
            f.write(f'| {k:.3f} | {raw_vals[ki]:.4f} | {raw_shifted[ki]:.4f} '
                    f'| {"yes" if mfit[ki] else "no"} |\n')
    log(f'  saved {md}')


if __name__ == '__main__':
    main()
