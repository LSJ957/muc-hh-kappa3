"""ML1/ML2 architecture + input builders.  Single source of truth used by both
stages; HPs supplied via dict from best.json or Optuna."""
from __future__ import annotations
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks

from . import physics_constants as _pc

# 12 non-TDA event-level globals (config drops 4 by default, leaving 8 for ML)
GLOBALS_NON_TDA = ['mHH', 'pT_HH', 'XHH', 'dphi_HH_met', 'mT_HH_met', 'dR_H1H2',
                   'cos_HH_lab', 'cos_theta_hel', 'n_btag_total', 'met',
                   'scalar_sum_E', 'vec_sum_p']
GLOBALS_TDA     = ['H_0', 'H_1', 'S_0', 'S_1', 'LB_1']
# Default-drop list (kinematically degenerate / numerically unstable globals).
# Single source of truth in physics_constants; mirrored here for convenience.
DEFAULT_DROP_GLOBALS = list(_pc.DEFAULT_DROP_GLOBALS)


def kept_globals(drop=()) -> list:
    drop = set(drop or ())
    return [c for c in GLOBALS_NON_TDA if c not in drop]


# ───────────────────────────────────────────────────────────────────────
# Input builders (operate on lib.data_loader output dicts: data['hl'], data['jets'])
# ───────────────────────────────────────────────────────────────────────
def _reconstruct_higgs(jets: np.ndarray, assign: np.ndarray) -> dict:
    """Returns dict of (N,2) arrays per recombined Higgs candidate, **pT-sorted**:
    slot [:,0] = leading-pT Higgs, [:,1] = subleading.  This matches the convention
    used by lib.spanet_engine.recompute_hl_from_assignment so the Higgs-token
    kinematics line up with hl['H1_*'] / hl['H2_*'] (especially H1/H2_nbtag).

    Includes 'nbtag' (number of b-tagged jets within each Higgs pair, 0/1/2) so
    callers can use a single coherent source of pT-ordered Higgs features.
    """
    PAIRS = _pc.PAIRINGS_PAIRS                     # single source of truth
    N = len(jets)
    pt = jets[:, :, 0]; eta = jets[:, :, 1]; phi = jets[:, :, 2]; m = np.maximum(jets[:, :, 3], 0.0)
    btag = np.rint(np.clip(jets[:, :, 4], 0, 1)).astype(np.int32)
    px = pt * np.cos(phi); py = pt * np.sin(phi); pz = pt * np.sinh(eta)
    E = np.sqrt(px*px + py*py + pz*pz + m*m)
    out_pt    = np.zeros((N, 2), np.float32)
    out_eta   = np.zeros((N, 2), np.float32)
    out_phi   = np.zeros((N, 2), np.float32)
    out_m     = np.zeros((N, 2), np.float32)
    out_drjj  = np.zeros((N, 2), np.float32)
    out_nbtag = np.zeros((N, 2), np.int32)
    for ai, ((i1, i2), (j1, j2)) in enumerate(PAIRS):
        mask = (assign == ai)
        if not mask.any(): continue
        for k, (a, b) in enumerate([(i1, i2), (j1, j2)]):
            Px = px[mask, a] + px[mask, b]; Py = py[mask, a] + py[mask, b]
            Pz = pz[mask, a] + pz[mask, b]; Ee = E[mask, a] + E[mask, b]
            ptH  = np.sqrt(Px*Px + Py*Py)
            etaH = np.arcsinh(Pz / np.maximum(ptH, 1e-6))
            phiH = np.arctan2(Py, Px)
            mH   = np.sqrt(np.maximum(Ee*Ee - Px*Px - Py*Py - Pz*Pz, 0.0))
            out_pt[mask, k]    = ptH
            out_eta[mask, k]   = etaH
            out_phi[mask, k]   = phiH
            out_m[mask, k]     = mH
            out_nbtag[mask, k] = btag[mask, a] + btag[mask, b]
            de = eta[mask, a] - eta[mask, b]
            dp = (phi[mask, a] - phi[mask, b] + np.pi) % (2*np.pi) - np.pi
            out_drjj[mask, k]  = np.sqrt(de*de + dp*dp)
    # pT-sort per event: slot 0 = higher pT, slot 1 = lower pT
    order = np.argsort(-out_pt, axis=1)
    rows  = np.arange(N)[:, None]
    return dict(
        pt    = out_pt[rows, order],
        eta   = out_eta[rows, order],
        phi   = out_phi[rows, order],
        m     = out_m[rows, order],
        drjj  = out_drjj[rows, order],
        nbtag = out_nbtag[rows, order],
    )


def build_jet_tokens(jets, met_phi, apply_btag_feature_mask=False):
    """Returns jet_cont (N,4,6) [log_pt, eta, sin_phi, cos_phi, log1p(m/5),
       |dphi_jet_met|]  and  jet_btag (N,4) int.
    The Δφ(jet, MET) is **unsigned** (|Δφ| ∈ [0, π]) to match the convention
    used in `extract_engine.vec_dphi` and the stored HL features
    `dphi_jet{j}_met`.  Sign of Δφ contains no extra physical information that
    isn't already in `sin_phi`/`cos_phi`."""
    pt = jets[:, :, 0].astype(np.float32); eta = jets[:, :, 1].astype(np.float32)
    phi = jets[:, :, 2].astype(np.float32); m = np.maximum(jets[:, :, 3], 0.0).astype(np.float32)
    btag = np.rint(np.clip(jets[:, :, 4], 0, 1)).astype(np.int32)
    log_pt = np.log(np.maximum(pt, 1.0))
    sphi = np.sin(phi); cphi = np.cos(phi)
    logm = np.log1p(m / 5.0)
    dphi_jm = np.abs((phi - met_phi[:, None] + np.pi) % (2*np.pi) - np.pi)
    jet_cont = np.stack([log_pt, eta, sphi, cphi, logm, dphi_jm], axis=-1).astype(np.float32)
    if apply_btag_feature_mask:
        btag = np.zeros_like(btag)
    return jet_cont, btag


def build_higgs_tokens(jets, assign, apply_btag_feature_mask=False):
    """Width-7 token per Higgs: [log_pt, eta, sin_phi, cos_phi, log_m, n_btag, dR_jj].
    All 7 features are pT-sorted (slot 0 = leading-pT Higgs) — coherent with
    hl['H1_*'] / hl['H2_*'] from recompute_hl_from_assignment.  nbtag now comes
    from `_reconstruct_higgs` itself (single source of truth, no slot-mismatch)."""
    rec = _reconstruct_higgs(jets, assign)
    log_pt = np.log(np.maximum(rec['pt'], 1.0))
    log_m  = np.log1p(np.maximum(rec['m'], 0.0))
    eta    = rec['eta']
    sphi   = np.sin(rec['phi']); cphi = np.cos(rec['phi'])
    drjj   = rec['drjj']
    nbt    = rec['nbtag'].astype(np.float32)
    if apply_btag_feature_mask:
        nbt = np.zeros_like(nbt)
    return np.stack([log_pt, eta, sphi, cphi, log_m, nbt, drjj], axis=-1).astype(np.float32)


def build_globals_non_tda(hl, apply_btag_feature_mask=False, drop=()):
    cols = kept_globals(drop)
    arr = np.stack([hl[c].astype(np.float32) for c in cols], axis=1)
    if apply_btag_feature_mask and 'n_btag_total' in cols:
        arr[:, cols.index('n_btag_total')] = 0.0
    return arr


def build_globals_tda(hl):
    return np.stack([hl[c].astype(np.float32) for c in GLOBALS_TDA], axis=1)


# ───────────────────────────────────────────────────────────────────────
# Model
# ───────────────────────────────────────────────────────────────────────
def transformer_block(x, d, n_heads, ffn_dim, dropout, wd, name):
    attn = layers.MultiHeadAttention(num_heads=n_heads, key_dim=max(d // n_heads, 1),
                                     dropout=dropout, name=f'{name}_mha')(x, x)
    x = layers.LayerNormalization(name=f'{name}_ln1')(layers.Add()([x, attn]))
    ffn = layers.Dense(ffn_dim, activation='gelu',
                       kernel_regularizer=regularizers.l2(wd))(x)
    ffn = layers.Dense(d, kernel_regularizer=regularizers.l2(wd))(ffn)
    ffn = layers.Dropout(dropout)(ffn)
    x = layers.LayerNormalization(name=f'{name}_ln2')(layers.Add()([x, ffn]))
    return x


def build_tunable_model(jet_attn: bool, hp: dict, seed=42, n_globals=12, higgs_dim=7):
    """jet on/off; Higgs+TDA flatten-dense; LL Particle-Cloud transformer always on.
    HPs from best.json."""
    keras.utils.set_random_seed(seed)
    d = int(hp['d_token']); nh = int(hp['n_heads']); ffn = int(hp['ffn_dim'])
    drop = float(hp['dropout']); wd = float(hp['wd'])
    head_dims = hp['head_dims']
    if isinstance(head_dims, str):
        head_dims = [int(x) for x in head_dims.split('_')]
    n_jet_layers = int(hp.get('n_jet_layers', 2))
    n_ll_layers  = int(hp.get('n_ll_layers', 1))

    inp_jc   = keras.Input(shape=(4, 6),  name='jet_cont')
    inp_jb   = keras.Input(shape=(4,),    name='jet_btag', dtype='int32')
    inp_ht   = keras.Input(shape=(2, higgs_dim), name='higgs_tok')
    inp_gnt  = keras.Input(shape=(n_globals,),   name='globals_non_tda')
    inp_gtda = keras.Input(shape=(5,),    name='globals_tda')
    inp_llc  = keras.Input(shape=(40, 4), name='ll_cloud')  # mask channel dropped 2026-06-02

    # ── jet branch ──
    if jet_attn:
        jc = layers.BatchNormalization()(inp_jc)
        jc = layers.Dense(d)(jc)
        bt = layers.Dense(d)(layers.Embedding(2, 8)(inp_jb))
        jt = layers.LayerNormalization()(layers.Add()([jc, bt]))
        for i in range(n_jet_layers):
            jt = transformer_block(jt, d, nh, ffn, drop, wd, name=f'jet_tr{i}')
        jet_repr = layers.GlobalAveragePooling1D()(jt)
    else:
        jf = layers.Flatten()(inp_jc)
        jbf = layers.Lambda(lambda t: tf.cast(t, tf.float32), output_shape=(4,))(inp_jb)
        z = layers.BatchNormalization()(layers.Concatenate()([jf, jbf]))
        jet_repr = layers.Dense(d, activation='gelu', kernel_regularizer=regularizers.l2(wd))(z)

    hf = layers.BatchNormalization()(layers.Flatten()(inp_ht))
    higgs_repr = layers.Dense(d, activation='gelu', kernel_regularizer=regularizers.l2(wd))(hf)

    g_full = layers.BatchNormalization()(layers.Concatenate()([inp_gnt, inp_gtda]))

    # ll_cloud now has 4 channels [pT_frac, Δη, Δφ, type]; the type is index 3.
    ll_cont = layers.Lambda(lambda x: tf.gather(x, [0, 1, 2], axis=-1),
                            output_shape=(40, 3))(inp_llc)
    ll_type = layers.Lambda(lambda x: tf.cast(x[..., 3], tf.int32), output_shape=(40,))(inp_llc)
    ll_te = layers.Embedding(6, 4)(ll_type)
    ll = layers.LayerNormalization()(layers.Dense(d)(
            layers.BatchNormalization()(layers.Concatenate(axis=-1)([ll_cont, ll_te]))))
    for i in range(n_ll_layers):
        ll = transformer_block(ll, d, nh, ffn, drop, wd, name=f'll_tr{i}')
    ll_repr = layers.GlobalAveragePooling1D()(ll)

    z = layers.BatchNormalization()(layers.Concatenate()([jet_repr, higgs_repr, g_full, ll_repr]))
    for hd in head_dims:
        z = layers.Dropout(drop)(layers.Dense(int(hd), activation='gelu',
                                              kernel_regularizer=regularizers.l2(wd))(z))
    out = layers.Dense(1, activation='sigmoid')(z)
    return keras.Model([inp_jc, inp_jb, inp_ht, inp_gnt, inp_gtda, inp_llc], out)


def make_callbacks(patience=10, monitor='val_auc', mode='max',
                   *, cosine_warmup=False, base_lr=None, epochs=None,
                   warmup_frac=0.10, min_lr_ratio=0.0):
    """Return a list of keras callbacks for ML1 / ML2 training.

    Always includes `EarlyStopping(restore_best_weights=True)`.

    Optionally includes a per-epoch learning-rate schedule that
    mirrors the SPANet trainer in `lib/spanet_engine.get_lr_scheduler`:
    linear warm-up for the first `warmup_frac * epochs` epochs, then
    a cosine decay from `base_lr` down to `min_lr_ratio * base_lr`.
    Pass `cosine_warmup=True` along with `base_lr` and `epochs`."""
    cbs = [callbacks.EarlyStopping(
            monitor=monitor, mode=mode, patience=patience,
            restore_best_weights=True, verbose=1)]
    if cosine_warmup:
        assert base_lr is not None and epochs is not None, (
            'cosine_warmup requires base_lr and epochs')
        warmup_epochs = max(1, int(round(warmup_frac * epochs)))
        base_lr = float(base_lr)
        min_lr = min_lr_ratio * base_lr

        def lr_schedule(epoch, _current_lr):
            if epoch < warmup_epochs:
                # linear warm-up from 0 to base_lr (matches SPANet 0 → base_lr)
                return base_lr * (epoch + 1) / warmup_epochs
            progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
            cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
            return min_lr + (base_lr - min_lr) * cosine

        cbs.append(callbacks.LearningRateScheduler(lr_schedule, verbose=0))
    return cbs
