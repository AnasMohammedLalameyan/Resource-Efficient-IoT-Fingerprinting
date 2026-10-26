"""
stage3_train.py — Decision Tree training with cost-complexity pruning.

Reads the selected-features table (Parquet preferred), performs:
  1. Fit a full unpruned tree to get the cost-complexity pruning path
  2. Select best ccp_alpha via GroupKFold (grouped by day) cross-validation
     with log-spaced alpha candidates and optional 1-SE rule
  3. Refit final pruned tree on entire training set

Saves the trained model to config.DT_MODEL_PATH.
Writes DT metadata to manifest 'dt' section.
"""

import time
import logging

import numpy as np
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import config
import io_utils
import manifest
import tuning_protocol

logger = logging.getLogger(__name__)


def _check_manifest_guard():
    """Verify manifest features match the selected table's columns."""
    m = manifest.read()
    feat = m.get("features")
    if feat is None:
        logger.warning("No 'features' section in manifest — skipping guard.")
        return None

    kept_cols = feat.get("kept_cols", [])
    return kept_cols


def run(input_path=None, kept_cols=None, n_jobs=None):
    """
    Run stage 3: Decision Tree training with cost-complexity pruning.

    Parameters
    ----------
    input_path : Path or None
        Path to selected-features table. Auto-detected if None.
    kept_cols : list[str] or None
        Feature column names. If None, read from manifest then infer from table.
    n_jobs : int or None
        Number of parallel jobs for CV folds (default: number of physical cores).

    Returns
    -------
    dict with keys: 'model', 'kept_cols', 'ccp_alpha', 'cv_f1',
        'depth', 'leaves', 'unpruned_depth', 'unpruned_leaves',
        'train_time_s', 'n_alphas_considered'
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

    # --- Load training data ---
    logger.info("=== STAGE 3: Decision Tree Training (Cost-Complexity Pruning) ===")

    if n_jobs is not None:
        logger.info("Note: n_jobs=%s passed to stage3_train.run() is ignored; folds run sequentially for comparable CPU-time accounting.", n_jobs)

    X_train, y_train, meta_train = io_utils.load_split("train", cols=kept_cols)
    groups = meta_train["day"].values

    logger.info("Training rows: %s, features: %d", f"{len(y_train):,}", X_train.shape[1])
    logger.info("X dtype: %s, C-contiguous: %s", X_train.dtype, X_train.flags['C_CONTIGUOUS'])

    # --- Tuning split via shared tuning_protocol ---
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

    # --- Tuning Timer: full unpruned fit on subsample + pruning path + sequential CV ---
    t_tune_start = time.perf_counter()
    t_tune_cpu_start = time.process_time()

    logger.info("Fitting unpruned tree on tuning subsample to get pruning path...")
    full_tree = DecisionTreeClassifier(criterion="gini", random_state=42)
    full_tree.fit(X_tune, y_tune)
    unpruned_depth = full_tree.get_depth()
    unpruned_leaves = full_tree.get_n_leaves()
    n_fits_tuning = 1

    path = full_tree.cost_complexity_pruning_path(X_tune, y_tune)
    all_alphas = path.ccp_alphas

    # Log-spaced alpha selection
    pos_alphas = all_alphas[all_alphas > 0]
    if len(pos_alphas) > 0:
        log_min = np.log10(pos_alphas.min())
        log_max = np.log10(pos_alphas.max())
        log_spaced = np.logspace(log_min, log_max, config.CV_N_ALPHAS)
        sampled_alphas = []
        for la in log_spaced:
            idx = np.argmin(np.abs(pos_alphas - la))
            sampled_alphas.append(float(pos_alphas[idx]))
        sampled_alphas = sorted(set(sampled_alphas))
        sampled_alphas = [0.0] + sampled_alphas
    else:
        sampled_alphas = [0.0]

    n_alphas_total = len(all_alphas)
    n_candidates = len(sampled_alphas)
    logger.info("Total alphas from pruning path: %d, sampled for CV: %d", n_alphas_total, n_candidates)

    cv_curve = []
    if n_splits < 2:
        logger.warning("Only %d unique training day(s). Cannot do GroupKFold. Using alpha=0.", len(np.unique(groups)))
        best_alpha = 0.0
        best_cv_f1 = float("nan")
    else:
        logger.info("Running sequential GroupKFold CV with %d splits over %d candidate alphas...",
                    n_splits, n_candidates)
        for alpha in sampled_alphas:
            fold_f1s = []
            for tr_idx, v_idx in fold_splits:
                n_fits_tuning += 1
                dt = DecisionTreeClassifier(criterion="gini", random_state=42, ccp_alpha=alpha)
                dt.fit(X_tune[tr_idx], y_tune[tr_idx])
                y_pred = dt.predict(X_tune[v_idx])
                fold_f1s.append(f1_score(y_tune[v_idx], y_pred, zero_division=0))
            mean_f1 = float(np.mean(fold_f1s))
            std_f1 = float(np.std(fold_f1s))
            cv_curve.append({
                "alpha": alpha,
                "mean_f1": mean_f1,
                "std_f1": std_f1,
            })

        chosen_idx = tuning_protocol.select_candidate(
            cv_curve, config.CV_ONE_SE, lambda r: r["alpha"]
        )
        best_alpha = cv_curve[chosen_idx]["alpha"]
        best_cv_f1 = cv_curve[chosen_idx]["mean_f1"]

    tuning_time_s = time.perf_counter() - t_tune_start
    tuning_cpu_s = time.process_time() - t_tune_cpu_start
    logger.info("Selected alpha: %.8f (CV F1=%.6f, tuning time=%.1f s)",
                best_alpha, best_cv_f1, tuning_time_s)

    # --- Final fit on all training data ---
    logger.info("Fitting final Decision Tree on all %d training rows with ccp_alpha=%.8f...",
                len(y_train), best_alpha)
    t_fit_start = time.perf_counter()
    t_fit_cpu_start = time.process_time()
    clf = DecisionTreeClassifier(criterion="gini", random_state=42, ccp_alpha=best_alpha)
    clf.fit(X_train, y_train)
    fit_time_s = time.perf_counter() - t_fit_start
    fit_cpu_s = time.process_time() - t_fit_cpu_start

    pruned_depth = clf.get_depth()
    pruned_leaves = clf.get_n_leaves()
    logger.info("Pruned tree: depth=%d, leaves=%d", pruned_depth, pruned_leaves)
    logger.info("Final tree fitted in %.3f s", fit_time_s)

    # --- Save model ---
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, config.DT_MODEL_PATH)
    logger.info("Model saved to: %s", config.DT_MODEL_PATH)

    # --- Write to manifest ---
    manifest.update("dt", {
        "ccp_alpha": best_alpha,
        "cv_f1": best_cv_f1,
        "depth": pruned_depth,
        "leaves": pruned_leaves,
        "unpruned_depth": unpruned_depth,
        "unpruned_leaves": unpruned_leaves,
        "unpruned_fit_data": "tuning_subsample",
        "tuning_time_s": tuning_time_s,
        "tuning_cpu_s": tuning_cpu_s,
        "fit_time_s": fit_time_s,
        "fit_cpu_s": fit_cpu_s,
        "total_training_cost_s": tuning_time_s + fit_time_s,
        "n_rows_tuning": n_rows_tuning,
        "n_rows_fit": len(y_train),
        "n_candidates": n_candidates,
        "n_fits_tuning": n_fits_tuning,
        "n_jobs_tuning": 1,
        "n_jobs_fit": 1,
        "tune_frac": tune_frac,
        "fold_fingerprint": fold_fingerprint,
        "dedup_used": False,
        "n_alphas_total": n_alphas_total,
        "n_alphas_cv": len(sampled_alphas),
        "cv_curve": cv_curve,
        "one_se_rule": config.CV_ONE_SE,
    })
    logger.info("Manifest 'dt' section updated")

    elapsed = time.perf_counter() - t_start
    logger.info("Stage 3 completed in %.1f s", elapsed)

    return {
        "model": clf,
        "kept_cols": kept_cols,
        "ccp_alpha": best_alpha,
        "cv_f1": best_cv_f1,
        "depth": pruned_depth,
        "leaves": pruned_leaves,
        "unpruned_depth": unpruned_depth,
        "unpruned_leaves": unpruned_leaves,
        "train_time_s": fit_time_s,
        "n_alphas_considered": n_alphas_total,
    }
