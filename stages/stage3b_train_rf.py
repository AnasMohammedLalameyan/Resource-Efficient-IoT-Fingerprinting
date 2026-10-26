"""
stage3b_train_rf.py — Random Forest training with GroupKFold tuning.

Uses the same data, folds, and 1-SE rule as the Decision Tree (WP4: fair comparison).
Tunes on a stratified subsample (RF_TUNE_FRAC), then refits on all training data.

Saves to config.RF_MODEL_PATH.
Writes RF metadata to manifest 'rf' section.
"""

import time
import os
import logging

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import config
import io_utils
import manifest
import tuning_protocol

logger = logging.getLogger(__name__)


def _check_manifest_guard():
    """Verify manifest features exist and return kept_cols."""
    m = manifest.read()
    feat = m.get("features")
    if feat is None:
        logger.warning("No 'features' section in manifest — skipping guard.")
        return None
    return feat.get("kept_cols", None)


def run(input_path=None, kept_cols=None, n_jobs=None):
    """
    Train a Random Forest classifier with GroupKFold CV tuning.

    Parameters
    ----------
    input_path : Path or None
        Unused (kept for backward compat). Data loaded via io_utils.
    kept_cols : list[str] or None
        Feature columns. If None, read from manifest.
    n_jobs : int or None
        Number of parallel jobs for RF fitting.

    Returns
    -------
    dict with keys: 'model', 'train_time_s', 'n_estimators',
        'total_nodes', 'mean_depth', 'model_size_kb',
        'best_params', 'cv_f1', 'tuning_time_s'
    """
    t_start = time.perf_counter()

    # Guard: verify manifest features.kept_cols equals the table's feature columns (same order)
    table_path = io_utils.find_table(config.SELECTED_FEATURES_DIR, "selected_features")
    if table_path.suffix == ".parquet":
        import pyarrow.parquet as pq
        cols_on_disk = list(pq.read_schema(str(table_path)).names)
    else:
        import pandas as pd
        cols_on_disk = list(pd.read_csv(table_path, nrows=0).columns)
    table_cols = [c for c in cols_on_disk if c not in ("mac_src", "day", "label")]

    manifest_cols = _check_manifest_guard()
    if manifest_cols is not None and manifest_cols != table_cols:
        raise RuntimeError(
            f"Feature mismatch: manifest features.kept_cols ({len(manifest_cols)}) "
            f"does not match selected table feature columns ({len(table_cols)}). "
            f"Re-run stage 2."
        )

    if kept_cols is None:
        kept_cols = manifest_cols if manifest_cols is not None else table_cols

    logger.info("=== STAGE 3b: Random Forest Training (GroupKFold Tuning) ===")

    # --- Load full training data ---
    X_train, y_train, meta_train = io_utils.load_split("train", cols=kept_cols)
    groups = meta_train["day"].values

    logger.info("Training rows: %s, features: %d", f"{len(y_train):,}", X_train.shape[1])

    rf_n_jobs = n_jobs if n_jobs else -1

    # --- GroupKFold CV tuning on subsample using shared tuning_protocol ---
    split_info = tuning_protocol.make_tuning_split(
        groups=groups,
        y=y_train,
        tune_frac=config.TUNE_FRAC,
        max_splits=config.GROUP_KFOLD_SPLITS,
    )
    tune_idx = split_info["tune_idx"]
    fold_splits = split_info["folds"]
    n_splits = split_info["n_splits"]
    tune_frac = split_info["tune_frac"]
    fold_fingerprint = split_info["fingerprint"]
    n_rows_tuning = split_info["n_tune_rows"]

    X_tune = X_train[tune_idx]
    y_tune = y_train[tune_idx]
    logger.info("Tuning subsample: %d rows (%.0f%% of training)", len(tune_idx), tune_frac * 100)

    best_params = None
    best_cv_f1 = -1.0
    cv_results = []
    tuning_time_s = 0.0
    tuning_cpu_s = 0.0
    n_candidates = 0
    n_fits_tuning = 0

    if n_splits >= 2:
        t_tune_start = time.perf_counter()
        t_tune_cpu_start = time.process_time()
        grid = config.RF_GRID
        tune_n_estimators = 30  # fewer trees during tuning

        for msl in grid["min_samples_leaf"]:
            for mf in grid["max_features"]:
                n_candidates += 1
                fold_f1s = []
                for train_idx, val_idx in fold_splits:
                    n_fits_tuning += 1
                    rf = RandomForestClassifier(
                        n_estimators=tune_n_estimators,
                        criterion="gini",
                        min_samples_leaf=msl,
                        max_features=mf,
                        n_jobs=rf_n_jobs,
                        random_state=42,
                    )
                    rf.fit(X_tune[train_idx], y_tune[train_idx])
                    y_pred = rf.predict(X_tune[val_idx])
                    fold_f1s.append(f1_score(y_tune[val_idx], y_pred, zero_division=0))

                mean_f1 = float(np.mean(fold_f1s))
                std_f1 = float(np.std(fold_f1s))
                cv_results.append({
                    "min_samples_leaf": msl,
                    "max_features": mf,
                    "mean_f1": mean_f1,
                    "std_f1": std_f1,
                })
                logger.info("  msl=%s, mf=%s -> CV F1=%.6f +/- %.6f", msl, mf, mean_f1, std_f1)

        tuning_time_s = time.perf_counter() - t_tune_start
        tuning_cpu_s = time.process_time() - t_tune_cpu_start

        # Apply selection rule via shared tuning_protocol
        chosen_idx = tuning_protocol.select_candidate(
            cv_results, config.CV_ONE_SE, lambda r: r["min_samples_leaf"]
        )
        best_entry = cv_results[chosen_idx]
        best_params = {
            "min_samples_leaf": best_entry["min_samples_leaf"],
            "max_features": best_entry["max_features"],
        }
        best_cv_f1 = best_entry["mean_f1"]

        logger.info("Best params: %s (CV F1=%.6f, tuning time=%.1f s)",
                    best_params, best_cv_f1, tuning_time_s)
    else:
        logger.warning("Only %d unique day(s) — skipping CV tuning, using defaults.", len(np.unique(groups)))
        best_params = {"min_samples_leaf": config.RF_MIN_SAMPLES_LEAF, "max_features": "sqrt"}

    # --- Final fit on all training data ---
    logger.info("Fitting final RF on all %d training rows with n_estimators=%d...",
                len(y_train), config.RF_N_ESTIMATORS)
    rf_final = RandomForestClassifier(
        n_estimators=config.RF_N_ESTIMATORS,
        criterion="gini",
        min_samples_leaf=best_params["min_samples_leaf"],
        max_features=best_params["max_features"],
        n_jobs=rf_n_jobs,
        random_state=42,
    )
    t_fit = time.perf_counter()
    t_fit_cpu = time.process_time()
    rf_final.fit(X_train, y_train)
    fit_time_s = time.perf_counter() - t_fit
    fit_cpu_s = time.process_time() - t_fit_cpu
    logger.info("Final fit time: %.3f s", fit_time_s)

    # Stats
    total_nodes = sum(t.tree_.node_count for t in rf_final.estimators_)
    depths = [t.get_depth() for t in rf_final.estimators_]
    mean_depth = float(np.mean(depths))
    logger.info("Number of trees: %d", rf_final.n_estimators)
    logger.info("Total node count across trees: %d", total_nodes)
    logger.info("Mean tree depth: %.1f", mean_depth)

    # Save
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf_final, config.RF_MODEL_PATH)
    model_size_kb = os.path.getsize(config.RF_MODEL_PATH) / 1024
    logger.info("Model saved to: %s (%.1f KB)", config.RF_MODEL_PATH, model_size_kb)

    # --- Write to manifest ---
    manifest.update("rf", {
        "best_params": best_params,
        "cv_f1": best_cv_f1,
        "tuning_time_s": tuning_time_s,
        "tuning_cpu_s": tuning_cpu_s,
        "fit_time_s": fit_time_s,
        "fit_cpu_s": fit_cpu_s,
        "total_training_cost_s": tuning_time_s + fit_time_s,
        "n_rows_tuning": n_rows_tuning,
        "n_rows_fit": len(y_train),
        "n_candidates": n_candidates,
        "n_fits_tuning": n_fits_tuning,
        "n_jobs_tuning": rf_n_jobs,
        "n_jobs_fit": rf_n_jobs,
        "tune_frac": tune_frac,
        "fold_fingerprint": fold_fingerprint,
        "dedup_used": False,
        "n_estimators": config.RF_N_ESTIMATORS,
        "total_nodes": total_nodes,
        "mean_depth": mean_depth,
        "model_size_kb": model_size_kb,
        "cv_results": cv_results,
        "one_se_rule": config.CV_ONE_SE,
    })
    logger.info("Manifest 'rf' section updated")

    elapsed = time.perf_counter() - t_start
    logger.info("Stage 3b completed in %.1f s", elapsed)

    return {
        "model": rf_final,
        "train_time_s": fit_time_s,
        "n_estimators": rf_final.n_estimators,
        "total_nodes": total_nodes,
        "mean_depth": mean_depth,
        "model_size_kb": model_size_kb,
        "best_params": best_params,
        "cv_f1": best_cv_f1,
        "tuning_time_s": tuning_time_s,
    }
