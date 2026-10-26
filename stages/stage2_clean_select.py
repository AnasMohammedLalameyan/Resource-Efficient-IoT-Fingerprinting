"""
stage2_clean_select.py — Cleaning + two-stage feature selection.

Reads the raw-features table from stage 1, performs:
  1. NaN fill (feature columns filled with config.NAN_FILL, default -1)
  2. Drop rows with missing label
  3. Optional stratified subsampling if sample_frac < 1.0
  4. Zero-variance pruning on training set
  5. RF importance ranking (20-tree RF, threshold 0.001) on stratified subsample of training set

Writes feature-selected CSV and Parquet to SELECTED_FEATURES_DIR.
Writes 'features' section to manifest.json.
Returns metadata about kept/dropped features for downstream use.
"""

import time
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
import io_utils
import manifest

logger = logging.getLogger(__name__)


def run(input_path=None, output_dir=None, sample_frac=None):
    """
    Run stage 2: cleaning and feature selection.

    Parameters
    ----------
    input_path : Path or str, optional
        Path to the raw-features table. Defaults to auto-detect via io_utils.find_table.
    output_dir : Path or str, optional
        Where to write the selected-features files. Defaults to config.SELECTED_FEATURES_DIR.
    sample_frac : float, optional
        Fraction of rows to sample. Defaults to config.SAMPLE_FRAC.

    Returns
    -------
    dict with keys:
        'output_path', 'kept_cols', 'pruned_zero_var', 'pruned_low_imp',
        'n_train_rows', 'n_test_rows', 'train_label_counts', 'test_label_counts'
    """
    t_start = time.perf_counter()
    output_dir = Path(output_dir) if output_dir else config.SELECTED_FEATURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Locate input table ---
    if input_path is None:
        input_path = io_utils.find_table(config.RAW_FEATURES_DIR, "raw_features")
    else:
        input_path = Path(input_path)

    logger.info("=== STAGE 2: Cleaning & Feature Selection ===")
    logger.info("Reading input: %s", input_path)

    if input_path.suffix == ".parquet":
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)

    logger.info("Input shape: %s rows x %s columns", f"{len(df):,}", len(df.columns))

    # --- Stratified Subsampling (for smoke runs or sample-frac) ---
    if sample_frac is None:
        sample_frac = config.SAMPLE_FRAC

    if 0.0 < sample_frac < 1.0:
        logger.info("Applying sample_frac=%.4f (stratified by day and label)", sample_frac)
        sampled_groups = []
        for (d, lbl), grp in df.groupby(["day", "label"], sort=False):
            n_grp = len(grp)
            n_sample = max(1, int(round(n_grp * sample_frac)))
            sampled_groups.append(grp.sample(n=min(n_sample, n_grp), random_state=42))
        df = pd.concat(sampled_groups, axis=0).sort_index()
        logger.info("Subsampled shape: %s rows x %s columns", f"{len(df):,}", len(df.columns))

    # --- Step 2: Cleaning ---
    pre = len(df)
    # Check per-column NaN rate and verify no feature legitimately takes sentinel -1
    available_feat_cols = [c for c in config.FEATURE_COLS if c in df.columns]
    nan_rates = df[available_feat_cols].isna().mean()
    logger.info("Per-column NaN rates (only non-zero shown):")
    for col, rate in nan_rates.items():
        if rate > 0:
            logger.info("  %-35s: %.4f (%s missing)", col, rate, f"{int(rate * len(df)):,}")

    cols_with_minus_one = [c for c in available_feat_cols if (df[c] == config.NAN_FILL).any()]
    if cols_with_minus_one:
        logger.warning("Features legitimately containing sentinel %s before fillna: %s",
                       config.NAN_FILL, cols_with_minus_one)
    else:
        logger.info("Verified: no feature column legitimately contains sentinel %s prior to fillna.",
                    config.NAN_FILL)

    # Fill candidate features with sentinel NAN_FILL (-1)
    df[available_feat_cols] = df[available_feat_cols].fillna(config.NAN_FILL)
    df = df.dropna(subset=["label"])
    post = len(df)
    logger.info("Cleaning: %s rows before, %s after, %s dropped (missing label)",
                f"{pre:,}", f"{post:,}", f"{pre - post:,}")
    logger.info("NaN fill value applied: %s (Note: changes feature distribution from legacy 0 fill; downstream models must be retrained)", config.NAN_FILL)

    # --- Label distribution ---
    logger.info("Label distribution:\n%s", df["label"].value_counts().to_string())

    # --- Step 4: Train/test split (day-based) ---
    train_df = df[df["day"].isin(config.TRAIN_DAYS)].copy()
    test_df  = df[df["day"].isin(config.TEST_DAYS)].copy()
    logger.info("Train: days=%s -> %s rows", sorted(config.TRAIN_DAYS), f"{len(train_df):,}")
    logger.info("Test:  days=%s -> %s rows", sorted(config.TEST_DAYS), f"{len(test_df):,}")
    logger.info("Train label dist:\n%s", train_df["label"].value_counts().to_string())
    logger.info("Test  label dist:\n%s", test_df["label"].value_counts().to_string())

    train_label_counts = {str(k): int(v) for k, v in train_df["label"].value_counts().to_dict().items()}
    test_label_counts  = {str(k): int(v) for k, v in test_df["label"].value_counts().to_dict().items()}

    # --- Step 5: Feature selection ---
    logger.info("--- Feature Selection ---")
    X_tr = train_df[available_feat_cols].values
    y_tr = (train_df["label"] == "IoT").astype(int).values

    # 5a. Zero-variance pruning
    variances = X_tr.var(axis=0)
    pruned_zero_var = [available_feat_cols[i] for i, v in enumerate(variances) if v == 0]
    active_cols = [c for c in available_feat_cols if c not in pruned_zero_var]
    logger.info("Zero-variance features dropped (%d): %s",
                len(pruned_zero_var), pruned_zero_var if pruned_zero_var else "none")

    # 5b. RF importance ranking (on stratified subsample if large)
    max_rf_rows = getattr(config, "SELECTION_RF_MAX_ROWS", 500_000)

    if len(train_df) > max_rf_rows:
        logger.info(
            "Subsampling training rows for RF importance: %s -> %s rows",
            f"{len(train_df):,}",
            f"{max_rf_rows:,}"
        )

        # Sample each class separately while preserving all columns.
        # This avoids pandas groupby.apply() index/column behavior.
        samples_per_class = max(1, max_rf_rows // train_df["label"].nunique())

        rf_sample = pd.concat(
            [
                group.sample(
                    n=min(len(group), samples_per_class),
                    random_state=42
                )
                for _, group in train_df.groupby("label", sort=False)
            ],
            axis=0
        )

        # Verify the target is still available
        if "label" not in rf_sample.columns:
            raise ValueError(
                f"'label' column missing after sampling. "
                f"Columns: {rf_sample.columns.tolist()}"
            )

        X_active = rf_sample[active_cols].to_numpy()
        y_active = rf_sample["label"].eq("IoT").astype(int).to_numpy()

    else:
        X_active = train_df[active_cols].to_numpy()
        y_active = y_tr

    # Train Random Forest and calculate feature importance
    rf = RandomForestClassifier(
        n_estimators=20,
        n_jobs=-1,
        random_state=42
    )

    rf.fit(X_active, y_active)
    importances = rf.feature_importances_

    # Remove low-importance features
    pruned_low_imp = [
        active_cols[i]
        for i, imp in enumerate(importances)
        if imp < config.RF_IMPORTANCE_THRESHOLD
    ]

    kept_cols = [
        active_cols[i]
        for i, imp in enumerate(importances)
        if imp >= config.RF_IMPORTANCE_THRESHOLD
    ]

    logger.info(
        "Low-importance features dropped (< %s) (%d): %s",
        config.RF_IMPORTANCE_THRESHOLD,
        len(pruned_low_imp),
        pruned_low_imp if pruned_low_imp else "none"
    )

    logger.info("Final feature set (%d): %s", len(kept_cols), kept_cols)
    # Log importance table
    imp_table = sorted(zip(active_cols, importances), key=lambda x: -x[1])
    for name, imp in imp_table:
        marker = " [DROPPED]" if name in pruned_low_imp else ""
        logger.info("  %-35s %.6f%s", name, imp, marker)

    # --- Write selected-features CSV and Parquet ---
    output_cols = ["mac_src", "day", "label"] + kept_cols
    selected_df = df[output_cols].copy()

    ext = ".csv.gz" if config.CSV_COMPRESS else ".csv"
    output_path = output_dir / f"selected_features{ext}"
    compression = "gzip" if config.CSV_COMPRESS else None
    selected_df.to_csv(output_path, index=False, compression=compression)

    parquet_path = output_dir / "selected_features.parquet"
    selected_df.to_parquet(parquet_path, index=False, engine="pyarrow")
    logger.info("Output written to: %s and %s", output_path, parquet_path)
    logger.info("Output shape: %s rows x %s columns", f"{len(selected_df):,}", len(selected_df.columns))

    # --- Write features metadata to Manifest ---
    manifest.update("features", {
        "kept_cols": kept_cols,
        "pruned_zero_var": pruned_zero_var,
        "pruned_low_imp": pruned_low_imp,
        "nan_fill": config.NAN_FILL,
        "train_days": sorted(list(config.TRAIN_DAYS)),
        "test_days": sorted(list(config.TEST_DAYS)),
        "n_train_rows": len(train_df),
        "n_test_rows": len(test_df),
        "train_label_counts": train_label_counts,
        "test_label_counts": test_label_counts,
        "sample_frac": sample_frac,
    })
    logger.info("Manifest 'features' section updated")

    elapsed = time.perf_counter() - t_start
    logger.info("Stage 2 completed in %.1f s", elapsed)

    return {
        "output_path": output_path,
        "kept_cols": kept_cols,
        "pruned_zero_var": pruned_zero_var,
        "pruned_low_imp": pruned_low_imp,
        "n_train_rows": len(train_df),
        "n_test_rows": len(test_df),
        "train_label_counts": train_label_counts,
        "test_label_counts": test_label_counts,
    }
