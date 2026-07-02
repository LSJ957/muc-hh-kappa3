#!/usr/bin/env python3
"""SPANet engine: model classes (Symmetry-Preserving Attention Network for
HH→4b jet pairing + sig/bg classification + Higgs-mass regression aux task),
training/eval helpers, and post-training HL recompute / inference utilities.
This is a **library module** consumed by `02_train_spanet.py`,
`03_precompute_pairing.py`, `04_train_ml1.py`, `05_train_ml2.py`, `06_ml_analysis.py`
and `07_dll_morphing.py`.  No stand-alone runner — the legacy `train_spanet()`,
`apply_spanet_and_save()`, `make_plots()`, `main()` were removed 2026-05-28.

Two architecture versions are supported via cfg['version']:
  • Version A: jet self-attention only (4 jets × jet_input_dim features)
  • Version B: also conditions on a constituent particle cloud
                (4 jets × max_const_per_jet × 4 features —
                 [pT_frac, Δη, Δφ, type], mask channel dropped 2026-06-02)

In the active pipeline 02_train_spanet sets `jet_input_dim=6` (the
`transform_6` output: log_pt, η, sin_φ, cos_φ, log1p(m/M0), btag).
"""

import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.metrics import roc_auc_score

from . import physics_constants as pc

# ═════════════════════════════════════════════════════════════════════════
# 0.  CONFIGURATION — single source of truth from physics_constants
# ═════════════════════════════════════════════════════════════════════════
PAIRINGS                  = list(pc.PAIRINGS_FLAT)
N_PAIRINGS                = pc.N_PAIRINGS
M_HIGGS                   = pc.M_HIGGS_GEV
ASSIGNMENT_DEPENDENT_HL   = list(pc.ASSIGNMENT_DEPENDENT_HL)
HL_FEATURES_45            = list(pc.HL_FEATURES_45)


# ═════════════════════════════════════════════════════════════════════════
# 1.  DATA LOADING
# ═════════════════════════════════════════════════════════════════════════
class HH4bDataset(Dataset):
    """
    Dataset for SPANet training.

    Returns:
        jets          : (4, jet_input_dim) float32 — jet features
        label_cls     : int — 0=background, 1=signal
        label_assign  : int — pairing index (0,1,2) or -1
        truth_valid   : bool — whether assignment label is usable
        met           : float32 — for auxiliary features
        ll_cloud      : (40, 4) float32 [Version B only]
    """
    def __init__(self, jets, labels_cls, labels_assign, truth_valid, ll_cloud=None):
        # NB: met / met_phi are intentionally NOT stored — they were unused by
        # SPANet.forward / SPANetLoss.forward; carrying them transferred the
        # tensors to the GPU every batch for no reason (review B-3).
        self.jets = torch.from_numpy(jets).float()
        self.labels_cls = torch.from_numpy(labels_cls).long()
        self.labels_assign = torch.from_numpy(labels_assign).long()
        self.truth_valid = torch.from_numpy(truth_valid).bool()
        self.ll_cloud = torch.from_numpy(ll_cloud).float() if ll_cloud is not None else None

    def __len__(self):
        return len(self.jets)

    def __getitem__(self, idx):
        item = {
            'jets': self.jets[idx],
            'label_cls': self.labels_cls[idx],
            'label_assign': self.labels_assign[idx],
            'truth_valid': self.truth_valid[idx],
        }
        if self.ll_cloud is not None:
            item['ll_cloud'] = self.ll_cloud[idx]
        return item


# ═════════════════════════════════════════════════════════════════════════
# 2.  MODEL: SPANet ARCHITECTURE
# ═════════════════════════════════════════════════════════════════════════

class ConstituentEncoder(nn.Module):
    """
    Mini Transformer for per-jet constituent encoding (Version B).
    Input: (batch, n_const, 4) per jet → (batch, const_embed_dim) per jet
    (Channels: [pT_frac, Δη, Δφ, type]; mask channel dropped 2026-06-02.)
    """
    def __init__(self, cfg):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(4, cfg['const_embed_dim']),
            nn.GELU(),
            nn.LayerNorm(cfg['const_embed_dim']),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg['const_embed_dim'],
            nhead=cfg['const_n_heads'],
            dim_feedforward=cfg['const_embed_dim'] * 4,
            dropout=cfg['dropout'],
            activation='gelu',
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=cfg['const_n_layers']
        )

    def forward(self, x, mask):
        """
        x    : (batch, n_const, 4)
        mask : (batch, n_const) bool — True where padded (no particle)
        """
        h = self.embed(x)                     # (B, n_const, D)
        h = self.encoder(h, src_key_padding_mask=mask)  # (B, n_const, D)
        # Mean-pool over non-padded constituents
        valid = (~mask).unsqueeze(-1).float()  # (B, n_const, 1)
        n_valid = valid.sum(dim=1).clamp(min=1)
        pooled = (h * valid).sum(dim=1) / n_valid  # (B, D)
        return pooled


class JetEncoder(nn.Module):
    """
    Per-jet feature embedding → self-attention across jets.
    Input: (batch, 4, jet_input_dim) → (batch, 4, embed_dim)
    """
    def __init__(self, jet_input_dim, cfg):
        super().__init__()
        D = cfg['jet_embed_dim']

        # Per-jet embedding MLP
        self.jet_embed = nn.Sequential(
            nn.Linear(jet_input_dim, D),
            nn.GELU(),
            nn.LayerNorm(D),
            nn.Dropout(cfg['dropout']),
            nn.Linear(D, D),
            nn.GELU(),
            nn.LayerNorm(D),
        )

        # No positional encoding → the encoder is permutation-equivariant over
        # the four jets (pos_embed removed: ablation showed it is performance-
        # neutral, and dropping it restores input-permutation invariance).

        # Self-attention layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D,
            nhead=cfg['n_attn_heads'],
            dim_feedforward=D * 4,
            dropout=cfg['dropout'],
            activation='gelu',
            batch_first=True,
        )
        self.attn = nn.TransformerEncoder(
            encoder_layer, num_layers=cfg['n_attn_layers']
        )

    def forward(self, x):
        """x: (batch, 4, jet_input_dim) → (batch, 4, D)"""
        h = self.jet_embed(x)           # (B, 4, D)
        h = self.attn(h)                 # (B, 4, D)  — permutation-equivariant (no pos_embed)
        return h


class AssignmentHead(nn.Module):
    """
    Predicts pairing probability: which of 3 pairings is correct.

    For each pairing (a,b,c,d):
      - Pool H1 = mean(jet_a, jet_b), H2 = mean(jet_c, jet_d)
      - Compute a score from the pair representation
    Output: (batch, 3) logits over pairings

    This is symmetry-preserving: the score for a pairing only depends on
    the jets in that pairing, regardless of label ordering.
    """
    def __init__(self, embed_dim, cfg):
        super().__init__()
        D = embed_dim
        hidden = cfg['assign_hidden']

        # Pair scorer (symmetrised): takes (H1_pool + H2_pool || |H1_pool - H2_pool|)
        # → 2D concatenation → MLP → scalar score.  Both terms are invariant
        # under the H1<->H2 exchange, so the pairing score is now manifestly
        # symmetric under swapping the two Higgs candidates (the mean pool already
        # makes it symmetric under the intra-candidate jet swap).
        layers = []
        in_dim = D * 2  # symmetric sum + absolute difference
        for h in hidden:
            layers.extend([nn.Linear(in_dim, h), nn.GELU(), nn.Dropout(cfg['dropout'])])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.scorer = nn.Sequential(*layers)

    def forward(self, jet_embeddings):
        """
        jet_embeddings: (batch, 4, D)
        Returns: (batch, 3) logits
        """
        scores = []
        for a, b, c, d in PAIRINGS:
            h1 = (jet_embeddings[:, a] + jet_embeddings[:, b]) / 2.0  # (B, D)
            h2 = (jet_embeddings[:, c] + jet_embeddings[:, d]) / 2.0
            pair_repr = torch.cat([h1 + h2, (h1 - h2).abs()], dim=-1)  # (B, 2D), H1<->H2 symmetric
            s = self.scorer(pair_repr)                      # (B, 1)
            scores.append(s)
        return torch.cat(scores, dim=-1)  # (B, 3)


class ClassificationHead(nn.Module):
    """
    Signal vs background classifier from jet embeddings + assignment.
    Input: global jet representation → P(signal)
    """
    def __init__(self, embed_dim, cfg):
        super().__init__()
        D = embed_dim
        hidden = cfg['cls_hidden']

        # Global pooling: mean of 4 jet embeddings + max + assignment-weighted
        layers = []
        in_dim = D * 2 + 1  # mean_pool + max_pool + assignment_entropy
        for h in hidden:
            layers.extend([nn.Linear(in_dim, h), nn.GELU(), nn.Dropout(cfg['dropout'])])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.classifier = nn.Sequential(*layers)

    def forward(self, jet_embeddings, assign_logits):
        """
        jet_embeddings: (batch, 4, D)
        assign_logits : (batch, 3)
        Returns: (batch,) logits (before sigmoid)
        """
        mean_pool = jet_embeddings.mean(dim=1)    # (B, D)
        max_pool = jet_embeddings.max(dim=1)[0]   # (B, D)
        # Assignment entropy as a feature (how confident the assignment is)
        assign_prob = F.softmax(assign_logits, dim=-1)
        entropy = -(assign_prob * (assign_prob + 1e-8).log()).sum(dim=-1, keepdim=True)
        # (B, 1)
        features = torch.cat([mean_pool, max_pool, entropy], dim=-1)
        return self.classifier(features).squeeze(-1)


class SPANet(nn.Module):
    """
    Full SPANet model for HH→4b: jet assignment + classification.

    Version A: jets only  (4 × jet_input_dim)
    Version B: jets (4 × jet_input_dim) + constituent cloud (4 × n_const × 4)

    The active pipeline uses jet_input_dim=6, the transform_6 output
    [log_pt, η, sin_φ, cos_φ, log1p(m/M0), btag]; the SPANet checkpoint
    stores the cfg so 03_precompute_pairing reads the matching value.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.version = cfg['version']

        # default 6 matches transform_6(); 10 is the legacy raw layout kept for
        # reference but never used by the current 02_train_spanet pipeline.
        jet_input_dim = cfg.get('jet_input_dim', 6)

        if self.version == 'B':
            self.const_encoder = ConstituentEncoder(cfg)
            jet_input_dim += cfg['const_embed_dim']

        self.jet_encoder = JetEncoder(jet_input_dim, cfg)
        D = cfg['jet_embed_dim']

        self.assignment_head = AssignmentHead(D, cfg)
        self.classification_head = ClassificationHead(D, cfg)

        # Auxiliary: mass regression for assigned Higgs pairs
        self.mass_regressor = nn.Sequential(
            nn.Linear(D, 64),
            nn.GELU(),
            nn.Linear(64, 1),  # predict m_H in units of M_HIGGS
        )

    def forward(self, jets, ll_cloud=None, ll_mask=None):
        """
        jets     : (B, 4, jet_input_dim)
        ll_cloud : (B, 40, 4) [Version B only]
        ll_mask  : (B, 40) bool, True=padded [Version B only]

        Returns dict:
            assign_logits : (B, 3)
            cls_logits    : (B,)
            mass_pred     : (B, 2)  predicted Higgs masses (normalized by M_H)
        """
        jet_features = jets  # (B, 4, jet_input_dim)

        if self.version == 'B' and ll_cloud is not None:
            # Encode constituents per jet
            B = jets.shape[0]
            n_const = self.cfg['max_const_per_jet']
            # ll_cloud is (B, 40, 4) = 4 jets × 10 constituents × 4 features
            # Reshape to (B*4, 10, 4)
            ll_per_jet = ll_cloud.reshape(B, 4, n_const, 4)
            ll_flat = ll_per_jet.reshape(B * 4, n_const, 4)

            # Mask: padded slot has all-zero feature vector (pT_frac=Δη=Δφ=type=0).
            # Using pT_frac==0 as the detection signal (a real particle never has
            # pT_frac=0 because of the divisor `jet_pt + 1e-9` in extract_engine).
            if ll_mask is None:
                ll_mask_flat = (ll_flat[:, :, 0] == 0)
            else:
                ll_mask_flat = ll_mask.reshape(B * 4, n_const)

            const_emb = self.const_encoder(ll_flat, ll_mask_flat)  # (B*4, D_c)
            const_emb = const_emb.reshape(B, 4, -1)               # (B, 4, D_c)

            jet_features = torch.cat([jet_features, const_emb], dim=-1)

        # Encode jets with self-attention
        jet_emb = self.jet_encoder(jet_features)   # (B, 4, D)

        # Heads
        assign_logits = self.assignment_head(jet_emb)  # (B, 3)
        cls_logits = self.classification_head(jet_emb, assign_logits)  # (B,)

        # Auxiliary mass regression: for the best pairing, predict Higgs masses
        assign_probs = F.softmax(assign_logits.detach(), dim=-1)
        # Soft assignment: weighted sum of pair embeddings
        h1_embs = []
        h2_embs = []
        for a, b, c, d in PAIRINGS:
            h1_embs.append((jet_emb[:, a] + jet_emb[:, b]) / 2.0)
            h2_embs.append((jet_emb[:, c] + jet_emb[:, d]) / 2.0)
        h1_stack = torch.stack(h1_embs, dim=1)  # (B, 3, D)
        h2_stack = torch.stack(h2_embs, dim=1)  # (B, 3, D)

        # Weighted by assignment probability
        w = assign_probs.unsqueeze(-1)    # (B, 3, 1)
        h1_rep = (h1_stack * w).sum(dim=1)  # (B, D)
        h2_rep = (h2_stack * w).sum(dim=1)

        m1_pred = self.mass_regressor(h1_rep).squeeze(-1)  # (B,)
        m2_pred = self.mass_regressor(h2_rep).squeeze(-1)

        return {
            'assign_logits': assign_logits,
            'cls_logits': cls_logits,
            'mass_pred': torch.stack([m1_pred, m2_pred], dim=-1),  # (B, 2)
        }


# ═════════════════════════════════════════════════════════════════════════
# 3.  LOSS FUNCTION
# ═════════════════════════════════════════════════════════════════════════
class SPANetLoss(nn.Module):
    """
    Combined loss:
      L = λ_assign * L_assign + λ_cls * L_cls + λ_mass * L_mass

    L_assign: CrossEntropy over 3 pairings (only for truth_valid events)
    L_cls:    Binary CE for signal vs background
    L_mass:   Huber loss on predicted Higgs masses (only for truth_valid signal)
    """
    def __init__(self, cfg, class_weights=None):
        super().__init__()
        self.lambda_assign = cfg['lambda_assign']
        self.lambda_cls = cfg['lambda_cls']
        self.lambda_mass = cfg['lambda_mass']

        if class_weights is not None:
            self.cls_loss_fn = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([class_weights])
            )
        else:
            self.cls_loss_fn = nn.BCEWithLogitsLoss()

        self.assign_loss_fn = nn.CrossEntropyLoss(reduction='none')
        self.mass_loss_fn = nn.HuberLoss(delta=0.5)

    def forward(self, outputs, batch):
        losses = {}

        # Classification loss (all events)
        cls_logits = outputs['cls_logits']
        cls_target = batch['label_cls'].float()
        losses['cls'] = self.cls_loss_fn(cls_logits, cls_target)

        # Assignment loss (only truth-valid signal events)
        valid = batch['truth_valid']
        if valid.any():
            assign_logits = outputs['assign_logits'][valid]
            assign_target = batch['label_assign'][valid]
            # Filter out -1 labels (shouldn't happen if truth_valid, but safety)
            good = assign_target >= 0
            if good.any():
                assign_loss = self.assign_loss_fn(
                    assign_logits[good], assign_target[good]
                ).mean()
                losses['assign'] = assign_loss
            else:
                losses['assign'] = torch.tensor(0.0, device=cls_logits.device)
        else:
            losses['assign'] = torch.tensor(0.0, device=cls_logits.device)

        # Mass regression loss (truth-valid signal only)
        if valid.any():
            mass_pred = outputs['mass_pred'][valid]  # (N_valid, 2)
            # Compute true Higgs masses from jet 4-vectors
            jets_valid = batch['jets'][valid]        # (N_valid, 4, jet_input_dim)
            assign_valid = batch['label_assign'][valid]
            good = assign_valid >= 0
            if good.any():
                true_masses = self._compute_true_masses(
                    jets_valid[good], assign_valid[good]
                )
                mass_target = true_masses / M_HIGGS  # normalize
                losses['mass'] = self.mass_loss_fn(mass_pred[good], mass_target)
            else:
                losses['mass'] = torch.tensor(0.0, device=cls_logits.device)
        else:
            losses['mass'] = torch.tensor(0.0, device=cls_logits.device)

        # Total
        total = (self.lambda_cls * losses['cls'] +
                 self.lambda_assign * losses['assign'] +
                 self.lambda_mass * losses['mass'])
        losses['total'] = total
        return losses

    @staticmethod
    def _compute_true_masses(jets, assign_indices):
        """
        Compute true Higgs masses from RAW jet 4-vectors given assignment.
        jets: (N, 4, ≥4) — first 4 columns MUST be raw [pT, η, φ, mass].
        assign_indices: (N,) — pairing index (0, 1, or 2)
        Returns: (N, 2) — [m_H1, m_H2]

        Note: do NOT pass `transform_6` output (normalized log_pt, sin_φ, cos_φ
        in idx 0..3) here — px/py/pz computation would be invalid.  In the
        active pipeline this is only called when lambda_mass > 0, which is
        currently set to 0.0 in `02_train_spanet.build_cfg_from_hp`.
        """
        pt   = jets[:, :, 0]
        eta  = jets[:, :, 1]
        phi  = jets[:, :, 2]
        mass = jets[:, :, 3]

        px = pt * torch.cos(phi)
        py = pt * torch.sin(phi)
        pz = pt * torch.sinh(eta)
        E  = torch.sqrt(mass**2 + px**2 + py**2 + pz**2)

        masses = torch.zeros(jets.shape[0], 2, device=jets.device)

        for pi, (a, b, c, d) in enumerate(PAIRINGS):
            m = (assign_indices == pi)
            if m.any():
                E1  = E[m, a] + E[m, b]
                px1 = px[m, a] + px[m, b]
                py1 = py[m, a] + py[m, b]
                pz1 = pz[m, a] + pz[m, b]
                m1 = torch.sqrt(torch.clamp(E1**2 - px1**2 - py1**2 - pz1**2, min=0))

                E2  = E[m, c] + E[m, d]
                px2 = px[m, c] + px[m, d]
                py2 = py[m, c] + py[m, d]
                pz2 = pz[m, c] + pz[m, d]
                m2 = torch.sqrt(torch.clamp(E2**2 - px2**2 - py2**2 - pz2**2, min=0))

                # Leading pT convention: sort
                pt1 = torch.sqrt(px1**2 + py1**2)
                pt2 = torch.sqrt(px2**2 + py2**2)
                swap = pt2 > pt1
                masses[m, 0] = torch.where(swap, m2, m1)
                masses[m, 1] = torch.where(swap, m1, m2)

        return masses


# ═════════════════════════════════════════════════════════════════════════
# 4.  TRAINING LOOP
# ═════════════════════════════════════════════════════════════════════════
def get_lr_scheduler(optimizer, cfg, n_steps_per_epoch):
    """Create learning rate scheduler with warmup."""
    warmup_steps = cfg['warmup_epochs'] * n_steps_per_epoch
    total_steps = cfg['epochs'] * n_steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        if cfg['lr_schedule'] == 'cosine':
            return 0.5 * (1.0 + np.cos(np.pi * progress))
        else:  # step
            if progress < 0.5:
                return 1.0
            elif progress < 0.8:
                return 0.1
            else:
                return 0.01

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device):
    model.train()
    metrics = defaultdict(float)
    n_batches = 0

    for batch in loader:
        # Move to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        # Forward
        ll_cloud = batch.get('ll_cloud', None)
        outputs = model(batch['jets'], ll_cloud=ll_cloud)
        losses = criterion(outputs, batch)

        # Backward
        optimizer.zero_grad()
        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # Track metrics
        for k, v in losses.items():
            metrics[k] += v.item()
        n_batches += 1

    return {k: v / n_batches for k, v in metrics.items()}


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    metrics = defaultdict(float)
    n_batches = 0

    all_cls_logits = []
    all_cls_labels = []
    all_assign_preds = []
    all_assign_labels = []
    all_truth_valid = []

    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        ll_cloud = batch.get('ll_cloud', None)
        outputs = model(batch['jets'], ll_cloud=ll_cloud)
        losses = criterion(outputs, batch)

        for k, v in losses.items():
            metrics[k] += v.item()
        n_batches += 1

        all_cls_logits.append(outputs['cls_logits'].cpu())
        all_cls_labels.append(batch['label_cls'].cpu())
        all_assign_preds.append(outputs['assign_logits'].argmax(dim=-1).cpu())
        all_assign_labels.append(batch['label_assign'].cpu())
        all_truth_valid.append(batch['truth_valid'].cpu())

    avg_metrics = {k: v / n_batches for k, v in metrics.items()}

    # Classification AUC
    cls_logits = torch.cat(all_cls_logits).numpy()
    cls_labels = torch.cat(all_cls_labels).numpy()
    cls_probs = 1.0 / (1.0 + np.exp(-cls_logits))
    try:
        avg_metrics['auc'] = roc_auc_score(cls_labels, cls_probs)
    except ValueError:
        avg_metrics['auc'] = 0.0

    # Assignment accuracy (truth-valid events only)
    assign_preds = torch.cat(all_assign_preds).numpy()
    assign_labels = torch.cat(all_assign_labels).numpy()
    truth_valid = torch.cat(all_truth_valid).numpy()
    valid_mask = truth_valid & (assign_labels >= 0)
    if valid_mask.sum() > 0:
        avg_metrics['assign_acc'] = float(
            (assign_preds[valid_mask] == assign_labels[valid_mask]).mean()
        )
    else:
        avg_metrics['assign_acc'] = 0.0

    return avg_metrics


# ═════════════════════════════════════════════════════════════════════════
# 5.  INFERENCE: APPLY SPANet + RECOMPUTE HL FEATURES
# ═════════════════════════════════════════════════════════════════════════
def recompute_hl_from_assignment(jets_raw, assignment, met, met_phi):
    """
    Recompute assignment-dependent HL features from jet 4-vectors
    using SPANet's predicted jet assignment.

    Args:
        jets_raw  : ndarray (N, 4, 10) — UN-normalized jet features
        assignment: ndarray (N,) int — predicted pairing index (0, 1, or 2)
        met       : ndarray (N,) — MET values
        met_phi   : ndarray (N,) — MET phi (if available, else 0)

    Returns:
        dict of recomputed HL feature arrays
    """
    N = len(jets_raw)
    j_pt   = jets_raw[:, :, 0]
    j_eta  = jets_raw[:, :, 1]
    j_phi  = jets_raw[:, :, 2]
    j_mass = jets_raw[:, :, 3]
    j_btag = jets_raw[:, :, 4]

    j_px = j_pt * np.cos(j_phi)
    j_py = j_pt * np.sin(j_phi)
    j_pz = j_pt * np.sinh(j_eta)
    j_E  = np.sqrt(j_mass**2 + j_px**2 + j_py**2 + j_pz**2)

    # Initialize arrays
    H1_E  = np.zeros(N); H1_px = np.zeros(N); H1_py = np.zeros(N); H1_pz = np.zeros(N)
    H2_E  = np.zeros(N); H2_px = np.zeros(N); H2_py = np.zeros(N); H2_pz = np.zeros(N)
    H1_btag = np.zeros(N, dtype=int)
    H2_btag = np.zeros(N, dtype=int)

    for pi, (a, b, c, d) in enumerate(PAIRINGS):
        m = assignment == pi
        if m.sum() == 0:
            continue

        E1 = j_E[m, a] + j_E[m, b]; px1 = j_px[m, a] + j_px[m, b]
        py1 = j_py[m, a] + j_py[m, b]; pz1 = j_pz[m, a] + j_pz[m, b]
        E2 = j_E[m, c] + j_E[m, d]; px2 = j_px[m, c] + j_px[m, d]
        py2 = j_py[m, c] + j_py[m, d]; pz2 = j_pz[m, c] + j_pz[m, d]

        pt1 = np.sqrt(px1**2 + py1**2)
        pt2 = np.sqrt(px2**2 + py2**2)
        swap = pt2 > pt1

        H1_E[m]  = np.where(swap, E2, E1)
        H1_px[m] = np.where(swap, px2, px1)
        H1_py[m] = np.where(swap, py2, py1)
        H1_pz[m] = np.where(swap, pz2, pz1)
        H2_E[m]  = np.where(swap, E1, E2)
        H2_px[m] = np.where(swap, px1, px2)
        H2_py[m] = np.where(swap, py1, py2)
        H2_pz[m] = np.where(swap, pz1, pz2)

        H1_btag[m] = np.where(swap,
                               (j_btag[m, c] + j_btag[m, d]).astype(int),
                               (j_btag[m, a] + j_btag[m, b]).astype(int))
        H2_btag[m] = np.where(swap,
                               (j_btag[m, a] + j_btag[m, b]).astype(int),
                               (j_btag[m, c] + j_btag[m, d]).astype(int))

    def _p4_kinematics(E, px, py, pz):
        pt = np.sqrt(px**2 + py**2)
        p  = np.sqrt(pt**2 + pz**2)
        safe_p = np.where(p == np.abs(pz), p + 1e-9, p)
        eta = np.where(p > 1e-9,
                       0.5 * np.log((safe_p + pz) / (safe_p - pz)), 0.0)
        phi = np.arctan2(py, px)
        m   = np.sqrt(np.clip(E**2 - p**2, 0, None))
        return pt, eta, phi, m

    H1_pt, H1_eta, H1_phi, H1_m = _p4_kinematics(H1_E, H1_px, H1_py, H1_pz)
    H2_pt, H2_eta, H2_phi, H2_m = _p4_kinematics(H2_E, H2_px, H2_py, H2_pz)

    HH_E  = H1_E + H2_E;   HH_px = H1_px + H2_px
    HH_py = H1_py + H2_py; HH_pz = H1_pz + H2_pz
    HH_pt, HH_eta, HH_phi, HH_m = _p4_kinematics(HH_E, HH_px, HH_py, HH_pz)

    def _dphi(phi1, phi2):
        d = np.abs(phi1 - phi2)
        return np.where(d > np.pi, 2*np.pi - d, d)

    def _dr(eta1, phi1, eta2, phi2):
        return np.sqrt((eta1 - eta2)**2 + _dphi(phi1, phi2)**2)

    dR_H1H2 = _dr(H1_eta, H1_phi, H2_eta, H2_phi)
    # ATLAS arXiv:2202.07288 resolved X_HH (centres 120 / 110, width 0.1·m_H):
    _eps_m = 1e-3
    _dXh1  = (H1_m - 120.0) / np.maximum(0.1 * H1_m, _eps_m)
    _dXh2  = (H2_m - 110.0) / np.maximum(0.1 * H2_m, _eps_m)
    XHH = np.sqrt(_dXh1 * _dXh1 + _dXh2 * _dXh2)

    dphi_hh_met = _dphi(HH_phi, met_phi)
    mT_HH_met = np.sqrt(np.clip(2 * HH_pt * met * (1 - np.cos(dphi_hh_met)), 0, None))

    H1_p = np.sqrt(H1_px**2 + H1_py**2 + H1_pz**2)
    H2_p = np.sqrt(H2_px**2 + H2_py**2 + H2_pz**2)
    dot_p = H1_px * H2_px + H1_py * H2_py + H1_pz * H2_pz
    cos_HH_lab = np.where((H1_p > 0) & (H2_p > 0), dot_p / (H1_p * H2_p), 0.0)

    beta_z = HH_pz / (HH_E + 1e-9)
    gamma  = HH_E / np.sqrt(np.clip(HH_E**2 - HH_pz**2, 1e-9, None))
    H1_pz_cm = gamma * (H1_pz - beta_z * H1_E)
    H1_pt_cm = np.sqrt(H1_px**2 + H1_py**2)
    H1_p_cm  = np.sqrt(H1_pt_cm**2 + H1_pz_cm**2)
    cos_theta_hel = np.where(H1_p_cm > 0, H1_pz_cm / H1_p_cm, 0.0)

    return {
        'H1_pt': H1_pt, 'H1_m': H1_m,
        'H2_pt': H2_pt, 'H2_m': H2_m,
        'H1_nbtag': H1_btag.astype(np.int8),
        'H2_nbtag': H2_btag.astype(np.int8),
        'dR_H1H2': dR_H1H2,
        'mHH': HH_m,
        'pT_HH': HH_pt,
        'XHH': XHH,
        'dphi_HH_met': dphi_hh_met,
        'mT_HH_met': mT_HH_met,
        'cos_HH_lab': cos_HH_lab,
        'cos_theta_hel': cos_theta_hel,
    }


@torch.no_grad()
def run_inference(model, jets_raw, jet_mean, jet_std, device,
                  ll_cloud=None, batch_size=2048):
    """
    Run SPANet inference on a dataset.

    Returns:
        cls_scores : ndarray (N,) — P(signal) from classification head
        assignment : ndarray (N,) — predicted pairing index (0, 1, or 2)
        assign_probs: ndarray (N, 3) — softmax over pairings
    """
    model.eval()
    N = len(jets_raw)
    jets_norm = (jets_raw - jet_mean) / jet_std

    cls_scores_list = []
    assignment_list = []
    assign_probs_list = []

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch_jets = torch.from_numpy(jets_norm[start:end]).float().to(device)

        batch_ll = None
        if ll_cloud is not None:
            batch_ll = torch.from_numpy(ll_cloud[start:end]).float().to(device)

        outputs = model(batch_jets, ll_cloud=batch_ll)

        cls_logits = outputs['cls_logits'].cpu().numpy()
        cls_probs = 1.0 / (1.0 + np.exp(-cls_logits))
        cls_scores_list.append(cls_probs)

        assign_logits = outputs['assign_logits'].cpu().numpy()
        # numerically stable softmax: shift by per-row max to prevent np.exp overflow
        # when assignment confidence is very high (review D-4).
        _shifted = assign_logits - assign_logits.max(axis=-1, keepdims=True)
        _exp = np.exp(_shifted)
        assign_probs = _exp / _exp.sum(axis=-1, keepdims=True)
        assign_probs_list.append(assign_probs)
        assignment_list.append(assign_logits.argmax(axis=-1))

    return (np.concatenate(cls_scores_list),
            np.concatenate(assignment_list),
            np.concatenate(assign_probs_list, axis=0))


