"""Shared jet-feature transformation: raw (pT, η, φ, m, btag, ...) → 6 features
(log_pt, η, sin_φ, cos_φ, log1p(m/M0), btag) used identically by SPANet AND the
ML1/ML2 jet-token stream.  This guarantees the same numerical pre-processing
across the entire analysis chain."""
import numpy as np

M0 = 5.0   # log1p(m/M0) reference mass [GeV]


def transform_6(jets_raw: np.ndarray) -> np.ndarray:
    """jets_raw : (..., 10) raw jet array as stored in the h5 (`/jets`).
       Columns assumed: 0 pt, 1 eta, 2 phi, 3 mass, 4 btag, 5.. extra (ignored).
       Returns : (..., 6) float32 with [log_pt, eta, sin_phi, cos_phi, log1p(m/M0), btag]."""
    j = np.asarray(jets_raw, dtype=np.float32)
    pt   = np.maximum(j[..., 0], 1.0)              # avoid log(0); pT in GeV
    eta  =            j[..., 1]
    phi  =            j[..., 2]
    m    = np.maximum(j[..., 3], 0.0)
    btag = np.rint(np.clip(j[..., 4], 0.0, 1.0))
    out = np.stack([
        np.log(pt),
        eta,
        np.sin(phi),
        np.cos(phi),
        np.log1p(m / M0),
        btag,
    ], axis=-1).astype(np.float32)
    return out


FEATURE_NAMES = ['log_pt', 'eta', 'sin_phi', 'cos_phi', 'log1p_m_over_M0', 'btag']


def compute_mean_std(jets_transformed: np.ndarray) -> tuple:
    """Per-feature mean / std over (events, jets). btag is left un-normalised."""
    flat = jets_transformed.reshape(-1, jets_transformed.shape[-1])
    mean = flat.mean(axis=0).astype(np.float32)
    std  = flat.std(axis=0).astype(np.float32)
    # don't normalise btag (already ∈ {0,1})
    btag_idx = FEATURE_NAMES.index('btag')
    mean[btag_idx] = 0.0; std[btag_idx] = 1.0
    std[std < 1e-6] = 1.0
    return mean, std
