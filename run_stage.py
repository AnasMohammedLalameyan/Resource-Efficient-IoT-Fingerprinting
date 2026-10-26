"""
run_stage.py — Unified CLI for running pipeline stages.

Usage examples:
    python run_stage.py convert  [--input PATH]
    python run_stage.py 2   --input <RAW_FEATURES_PATH>
    python run_stage.py 3   [--n-jobs N]
    python run_stage.py 3b  [--n-jobs N]
    python run_stage.py 4   [--model dt rf] [--run-id ID]
    python run_stage.py 5   [--run-id ID]
    python run_stage.py 5b  [--run-id ID]
    python run_stage.py 5c  [--run-id ID]
    python run_stage.py 6   [--run-id ID]
    python run_stage.py range 2 3b            # inclusive range
    global flags: --tag NAME --sample-frac F
"""

import sys
import os
import argparse
import logging
import time
from pathlib import Path
from datetime import datetime

# Ensure project root on sys.path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Canonical stage order
# ---------------------------------------------------------------------------
CANONICAL_ORDER = ["convert", "2", "3", "3b", "4", "5", "5c", "5b", "6"]


def setup_logging(stage_name):
    """Configure logging to console and per-stage log file. Idempotent."""
    import config

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
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = config.LOGS_DIR / f"{stage_name}_{ts}.log"
    fh = logging.FileHandler(str(log_path), encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    return log_path


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------

def _run_convert(args):
    import config
    import io_utils

    logger = logging.getLogger("run_stage.convert")
    if args.input:
        input_path = Path(args.input)
    else:
        input_path = io_utils.find_table(config.RAW_FEATURES_DIR, "raw_features")

    logger.info("Converting %s to Parquet...", input_path)
    out = io_utils.convert(input_path)
    logger.info("Parquet written to: %s", out)
    return {"output_path": str(out)}


def _run_stage2(args):
    from stages import stage2_clean_select

    input_path = Path(args.input) if args.input else None
    return stage2_clean_select.run(input_path=input_path)


def _run_stage3(args):
    from stages import stage3_train

    n_jobs = getattr(args, "n_jobs", None)
    return stage3_train.run(n_jobs=n_jobs)


def _run_stage3b(args):
    from stages import stage3b_train_rf

    n_jobs = getattr(args, "n_jobs", None)
    return stage3b_train_rf.run(n_jobs=n_jobs)


def _run_stage4(args):
    import config
    import io_utils
    import manifest
    import json
    import joblib
    import numpy as np
    from stages import stage4_evaluate
    from datetime import datetime

    logger = logging.getLogger("run_stage.4")

    run_id = args.run_id or datetime.now().strftime("%Y-%m-%d_%H%M%S")
    models_to_eval = args.model if args.model else list(config.MODEL_NAMES)

    # Load test data
    X_test, y_test, meta_test = io_utils.load_split("test")
    logger.info("Test set loaded: %d rows", len(y_test))

    # Get kept_cols from manifest
    m = manifest.read()
    feat_section = m.get("features", {})
    kept_cols = feat_section.get("kept_cols", None)

    for model_name in models_to_eval:
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
        json_path = config.RESULTS_DIR / f"run_{run_id}_metrics_{model_name}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, default=str)
        logger.info("Metrics JSON written: %s", json_path)

    return {"run_id": run_id}


def _run_stage5(args):
    from stages import stage5_benchmark

    run_id = args.run_id or _find_latest_run_id()
    return stage5_benchmark.run(run_id=run_id)

def _run_stage5c(args):
    from stages import stage5c_resource

    run_id = args.run_id or _find_latest_run_id()
    return stage5c_resource.run(run_id=run_id)

def _run_stage5b(args):
    from stages import stage5b_compare

    run_id = args.run_id or _find_latest_run_id()
    return stage5b_compare.run(run_id=run_id)


def _run_stage6(args):
    from stages import stage6_report

    run_id = args.run_id or _find_latest_run_id()
    return stage6_report.run(run_id=run_id)


def _find_latest_run_id():
    """Find latest run_id by looking at metrics JSON files in RESULTS_DIR."""
    import config

    results = sorted(config.RESULTS_DIR.glob("run_*_metrics_*.json"), reverse=True)
    if not results:
        raise FileNotFoundError(
            f"No metrics JSON found in {config.RESULTS_DIR}. Run stage 4 first."
        )
    # Extract run_id from filename like run_2026-10-07_130000_metrics_dt.json
    name = results[0].stem  # run_2026-10-07_130000_metrics_dt
    # Remove _metrics_<model> suffix
    parts = name.split("_metrics_")
    run_id = parts[0].replace("run_", "", 1)
    return run_id


# ---------------------------------------------------------------------------
# Stage dispatcher
# ---------------------------------------------------------------------------
STAGE_MAP = {
    "convert": _run_convert,
    "2": _run_stage2,
    "3": _run_stage3,
    "3b": _run_stage3b,
    "4": _run_stage4,
    "5": _run_stage5,
    "5c": _run_stage5c,
    "5b": _run_stage5b,
    "6": _run_stage6,
}


def main():
    parser = argparse.ArgumentParser(
        description="Run pipeline stages individually or as a range.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("stage", nargs="?", default=None,
                        help="Stage to run: convert, 2, 3, 3b, 4, 5, 5b, 6, or 'range'")
    parser.add_argument("range_args", nargs="*", default=[],
                        help="Start and end stages when using 'range' mode (e.g. 2 3b)")
    parser.add_argument("--tag", type=str, default=None,
                        help="Experiment tag (creates namespaced output directories)")
    parser.add_argument("--sample-frac", type=float, default=None,
                        help="Fraction of data to sample (0.0-1.0)")
    parser.add_argument("--input", type=str, default=None,
                        help="Input path for stage 2 or convert")
    parser.add_argument("--n-jobs", type=int, default=None,
                        help="Number of parallel jobs for stages 3/3b")
    parser.add_argument("--model", nargs="+", default=None,
                        help="Models to evaluate in stage 4 (e.g., dt rf)")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Run ID for stages 4-6")

    args = parser.parse_args()

    if args.stage is None:
        parser.print_help()
        sys.exit(1)

    # Apply global flags to environment before importing config
    if args.tag is not None:
        os.environ["IOT_TAG"] = args.tag
    if args.sample_frac is not None:
        os.environ["IOT_SAMPLE_FRAC"] = str(args.sample_frac)

    # Now import and refresh config
    import config
    config.refresh_paths(tag=args.tag, sample_frac=args.sample_frac)

    logger = logging.getLogger("run_stage")

    if args.stage == "range":
        if len(args.range_args) != 2:
            print("Usage: python run_stage.py range <start_stage> <end_stage> [--tag ...]")
            sys.exit(1)

        start_s, end_s = args.range_args[0], args.range_args[1]

        if start_s not in CANONICAL_ORDER or end_s not in CANONICAL_ORDER:
            print(f"Unknown stage(s): {start_s}, {end_s}. Valid: {CANONICAL_ORDER}")
            sys.exit(1)

        si = CANONICAL_ORDER.index(start_s)
        ei = CANONICAL_ORDER.index(end_s)
        if si > ei:
            print(f"Start stage '{start_s}' comes after end stage '{end_s}'")
            sys.exit(1)

        stages_to_run = CANONICAL_ORDER[si:ei+1]
        logger.info("Running stages: %s", " -> ".join(stages_to_run))

        for s in stages_to_run:
            log_path = setup_logging(f"stage{s}")
            logger.info("=" * 60)
            logger.info("Running stage: %s", s)
            logger.info("=" * 60)
            t0 = time.perf_counter()
            STAGE_MAP[s](args)
            elapsed = time.perf_counter() - t0
            logger.info("Stage %s finished in %.1f s", s, elapsed)

    else:
        # Single stage mode
        if args.range_args:
            print(f"Unexpected extra arguments: {args.range_args}")
            sys.exit(1)
        stage = args.stage
        if stage not in STAGE_MAP:
            print(f"Unknown stage: {stage}. Valid: {list(STAGE_MAP.keys())} or 'range'")
            sys.exit(1)

        log_path = setup_logging(f"stage{stage}")
        logger.info("=" * 60)
        logger.info("Running stage: %s (tag=%s, sample_frac=%s)",
                     stage, config.TAG, config.SAMPLE_FRAC)
        logger.info("=" * 60)
        t0 = time.perf_counter()
        result = STAGE_MAP[stage](args)
        elapsed = time.perf_counter() - t0
        logger.info("Stage %s finished in %.1f s", stage, elapsed)


if __name__ == "__main__":
    main()
