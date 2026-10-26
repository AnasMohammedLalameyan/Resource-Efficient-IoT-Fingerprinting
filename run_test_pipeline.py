"""
run_test_pipeline.py — Test / evaluation orchestrator (thin wrapper).

Runs: stage4 -> stage5 -> stage5b -> stage6
Communicates purely via files on disk and manifest.json.
"""

import sys
import os
import time
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import joblib

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
import io_utils
import manifest
from stages import stage4_evaluate, stage5_benchmark, stage5c_resource, stage5b_compare, stage6_report


def setup_logging(run_id):
    """Configure logging to console and per-run log file. Idempotent."""
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # File
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = config.LOGS_DIR / f"run_{run_id}.log"
    fh = logging.FileHandler(str(log_path), encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    return log_path


def run_pipeline(run_id=None):
    """
    Run the full test / evaluation pipeline.

    Parameters
    ----------
    run_id : str, optional
        Unique run ID. Generated from timestamp if not provided.

    Returns
    -------
    run_id : str
    """
    if run_id is None:
        run_id = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    log_path = setup_logging(run_id)
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("IoT Device Fingerprinting Pipeline - TESTING")
    logger.info("Run ID: %s", run_id)
    logger.info("=" * 60)

    t_start = time.perf_counter()

    # -----------------------------------------------------------------------
    # 1. Stage 4: Evaluate models on full test set
    # -----------------------------------------------------------------------
    X_test, y_test, meta_test = io_utils.load_split("test")
    logger.info("Loaded test set: %s rows", f"{len(y_test):,}")

    m_data = manifest.read()
    feat_section = m_data.get("features", {})
    kept_cols = feat_section.get("kept_cols", None)

    for model_name in config.MODEL_NAMES:
        model_path = config.MODELS_DIR / f"{model_name}_model.joblib"
        if not model_path.exists():
            logger.warning("Model %s not found at %s — skipping", model_name, model_path)
            continue

        model = joblib.load(model_path)
        logger.info("Loaded %s model from %s", model_name, model_path)

        # Guard: check n_features_in_
        if kept_cols and hasattr(model, "n_features_in_"):
            if model.n_features_in_ != len(kept_cols):
                raise RuntimeError(
                    f"Feature mismatch for {model_name}: model expects "
                    f"{model.n_features_in_} features but manifest has "
                    f"{len(kept_cols)}. Re-run stage 2 then stage 3/3b."
                )

        metrics = stage4_evaluate.run(
            model, model_name, X_test, y_test,
            meta_test=meta_test, run_id=run_id,
        )

        # Write metrics JSON
        config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        metrics_json_path = config.RESULTS_DIR / f"run_{run_id}_metrics_{model_name}.json"
        with open(metrics_json_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, default=str)
        logger.info("Metrics JSON written: %s", metrics_json_path)

    # -----------------------------------------------------------------------
    # 2. Stage 5: Time-focused benchmark
    # -----------------------------------------------------------------------
    stage5_benchmark.run(run_id=run_id)

    stage5c_resource.run(run_id=run_id)

    # -----------------------------------------------------------------------
    # 3. Stage 5b: Two-model comparison & graphs
    # -----------------------------------------------------------------------
    stage5b_compare.run(run_id=run_id)

    # -----------------------------------------------------------------------
    # 4. Stage 6: PDF report generation
    # -----------------------------------------------------------------------
    stage6_report.run(run_id=run_id)

    elapsed = time.perf_counter() - t_start
    logger.info("=" * 60)
    logger.info("TESTING COMPLETE in %.1f s", elapsed)
    logger.info("Run ID: %s", run_id)
    logger.info("Log:    %s", log_path)
    logger.info("Results dir: %s", config.RESULTS_DIR)
    logger.info("=" * 60)

    return run_id


def main():
    run_pipeline()


if __name__ == "__main__":
    main()
