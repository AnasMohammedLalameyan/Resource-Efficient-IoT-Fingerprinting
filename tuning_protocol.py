"""
tuning_protocol.py — Shared hyperparameter tuning protocol for DT and RF models.

Provides deterministic stratified subsampling, grouped cross-validation fold
generation, fold fingerprinting, and candidate selection (argmax or 1-SE rule).
"""

import hashlib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def make_tuning_split(groups, y, tune_frac, max_splits, seed=42):
    """
    Generate deterministic tuning subsample and GroupKFold splits.

    Parameters
    ----------
    groups : np.ndarray or pd.Series
        Group labels (e.g., 'day') for each training sample.
    y : np.ndarray or pd.Series
        Target labels (e.g., 0/1 or IoT/non-IoT) for each training sample.
    tune_frac : float
        Fraction of data to use for hyperparameter tuning.
    max_splits : int
        Maximum number of cross-validation splits.
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    dict:
      tune_idx   : np.ndarray[int]  sorted indices into the full training arrays
      folds      : list[(np.ndarray train_pos, np.ndarray val_pos)]  positions RELATIVE to tune_idx
      n_splits   : int            (0 if fewer than 2 unique days -> caller falls back to defaults)
      tune_frac  : float
      n_tune_rows: int
      fingerprint: str            sha1 over tune_idx bytes + fold val_pos bytes
    """
    n_total = len(y)
    groups_arr = np.asarray(groups)
    y_arr = np.asarray(y)

    if tune_frac < 1.0 and n_total > 1000:
        tune_df = pd.DataFrame({"day": groups_arr, "label": y_arr, "idx": np.arange(n_total, dtype=int)})
        sampled = tune_df.groupby(["day", "label"], group_keys=False).apply(
            lambda g: g.sample(n=max(1, int(len(g) * tune_frac)), random_state=seed)
        )
        tune_idx = np.sort(sampled["idx"].values.astype(int))
    else:
        tune_idx = np.arange(n_total, dtype=int)

    groups_tune = groups_arr[tune_idx]
    unique_days = np.unique(groups_tune)

    if len(unique_days) < 2:
        n_splits = 0
        folds = []
    else:
        n_splits = min(max_splits, len(unique_days))
        gkf = GroupKFold(n_splits=n_splits)
        folds = list(gkf.split(tune_idx, y_arr[tune_idx], groups_tune))

    hasher = hashlib.sha1()
    hasher.update(np.ascontiguousarray(tune_idx, dtype=np.int64).tobytes())
    for _, val_pos in folds:
        hasher.update(np.ascontiguousarray(val_pos, dtype=np.int64).tobytes())
    fingerprint = hasher.hexdigest()

    return {
        "tune_idx": tune_idx,
        "folds": folds,
        "n_splits": n_splits,
        "tune_frac": float(tune_frac),
        "n_tune_rows": int(len(tune_idx)),
        "fingerprint": fingerprint,
    }


def select_candidate(results, one_se, simplicity_key):
    """
    Select best candidate based on CV performance and simplicity.

    Parameters
    ----------
    results : list of dicts
        Each dict must contain 'mean_f1' and 'std_f1'.
    one_se : bool
        If True, apply the 1-SE rule; if False, return argmax.
    simplicity_key : callable
        Function mapping result dict to a comparable value where larger = simpler.

    Returns
    -------
    int : Index of the chosen candidate in results.
    """
    if not results:
        raise ValueError("Cannot select candidate from empty results list")

    best_idx = 0
    best_mean_f1 = results[0]["mean_f1"]
    for i in range(1, len(results)):
        if results[i]["mean_f1"] > best_mean_f1:
            best_mean_f1 = results[i]["mean_f1"]
            best_idx = i

    if not one_se:
        return best_idx

    threshold = results[best_idx]["mean_f1"] - results[best_idx]["std_f1"]
    eligible = [i for i, r in enumerate(results) if r["mean_f1"] >= threshold]
    if not eligible:
        return best_idx

    chosen_idx = eligible[0]
    best_simplicity = simplicity_key(results[chosen_idx])

    for i in eligible[1:]:
        val = simplicity_key(results[i])
        if val > best_simplicity:
            best_simplicity = val
            chosen_idx = i

    return chosen_idx
