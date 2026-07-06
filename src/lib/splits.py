"""Stratified 70/15/15 train/val/test split.

Critical design choices:
  - test fold is hold-out from the model's perspective: the trainer never sees
    test rows for fit/early-stopping.  04/05 store the resulting `idx_test`
    index array in their score npz so downstream steps (06, 07) can recover
    exactly the same fold.
  - random_state is a single integer taken from `training.seed` in the config;
    passing a different seed yields an independent partition (useful for
    resampling studies).
  - stratification is on `sb_target` (sig vs bg). Stratification on per-process
    `target_everytype` would be ideal but rare bkg processes can have <2
    events per fold; sb_target is a safer choice.
"""
from __future__ import annotations
import numpy as np
from sklearn.model_selection import train_test_split


def make_split_70_15_15(
    labels: np.ndarray,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Return train/val/test index arrays + boolean masks for a 70/15/15 split
    stratified on `labels`.

    Returns dict with keys:
      idx_train, idx_val, idx_test : int arrays
      mask_train, mask_val, mask_test : bool arrays of length N
      test_fraction : float (≈ 0.15)
    """
    labels = np.asarray(labels)
    N = len(labels)
    idx = np.arange(N)

    # 70+15 : 15 (train+val vs test)
    idx_trv, idx_test = train_test_split(
        idx, test_size=0.15, random_state=seed, stratify=labels,
    )
    # 70/85 ≈ 0.8235 of train+val pool goes to train
    idx_train, idx_val = train_test_split(
        idx_trv, test_size=15.0/85.0, random_state=seed, stratify=labels[idx_trv],
    )

    def mask_from_idx(i):
        m = np.zeros(N, dtype=bool); m[i] = True; return m

    return dict(
        idx_train=idx_train, idx_val=idx_val, idx_test=idx_test,
        mask_train=mask_from_idx(idx_train),
        mask_val  =mask_from_idx(idx_val),
        mask_test =mask_from_idx(idx_test),
        test_fraction=len(idx_test) / N,
    )


def canonical_sigbg_strata(sig_lab, truth_valid):
    """Canonical stratification array for the sigbg pool.  02 uses it for
    its own train/val stratification in every mode; 04/04a (and 07's fold
    reconstruction) switch to it when SPANET_SHARED_SPLIT=1, which makes
    SPANet and ML1 draw one IDENTICAL 70/15/15 partition so SPANet's
    training/val never overlaps ML1's test fold.

    Three effective strata (= sig_lab*2 + truth_valid):
      0  background           (sig_lab=0, tv=False)
      2  signal w/o truth     (sig_lab=1, tv=False; assign loss N/A)
      3  signal w/ truth      (sig_lab=1, tv=True ; assign loss applies)
    (sig=0/tv=True, i.e. stratum 1, does not exist by construction.)"""
    return np.asarray(sig_lab).astype(np.int64) * 2 + \
           np.asarray(truth_valid).astype(np.int64)
