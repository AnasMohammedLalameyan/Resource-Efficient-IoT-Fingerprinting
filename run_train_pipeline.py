"""
run_train_pipeline.py — Training orchestrator (thin wrapper).

Runs: stage2 -> stage3 -> stage3b, starting from an EXISTING raw table.
Does NOT call stage 1. Does NOT write legacy artifacts.
"""

import sys
import time
import logging
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from stages import stage2_clean_select, stage3_train
from stages import stage3b_train_rf


def setup_logging():
    """Configure logging to console. Idempotent."""
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    ch.setFormatter(fmt)
    root.addHandler(ch)


def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("IoT Device Fingerprinting Pipeline - TRAINING")
    logger.info("(stages 2 -> 3 -> 3b, from existing raw table)")
    logger.info("=" * 60)
    t_start = time.perf_counter()

    # Stage 2: Cleaning + Feature Selection
    stage2_result = stage2_clean_select.run()

    # Stage 3: Decision Tree Training (with pruning)
    stage3_result = stage3_train.run()

    # Stage 3b: Random Forest Training
    stage3b_result = stage3b_train_rf.run()

    elapsed = time.perf_counter() - t_start
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE in %.1f s", elapsed)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
