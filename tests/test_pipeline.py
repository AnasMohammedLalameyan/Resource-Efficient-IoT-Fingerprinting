"""
tests/test_pipeline.py — Comprehensive unit and integration test suite (WP7).

Covers all 8 required test cases from guidelines.md:
  1. _prune_tree equivalence (WP3 Tier 2)
  2. Dedup-weighted tree equivalence (WP3 Tier 3a)
  3. confusion_matrix with a single class (WP2 item 4)
  4. Manifest guard fires on column mismatch (WP1 item 5)
  5. Stage 3 idempotence (two runs produce identical alpha and predictions)
  6. make_deploy_artifacts works without training_artifacts.joblib (WP2 item 3)
  7. Full smoke pipeline: stages 2 to 6 on synthetic fixture ending with a PDF
  8. Timing sanity: repeating batch timing twice is within loose tolerance
"""

import os
import sys
import copy
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import joblib
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
import io_utils
import manifest
from tests.make_fixture import create_synthetic_raw_features


@pytest.fixture(scope="session")
def synthetic_data_path(tmp_path_factory):
    """Generate a fast synthetic fixture for testing."""
    tmp_dir = tmp_path_factory.mktemp("raw_data")
    fixture_path = tmp_dir / "raw_features.csv.gz"
    create_synthetic_raw_features(fixture_path, n_rows_per_day=300, random_state=42)
    return fixture_path


# ---------------------------------------------------------------------------
# Test 1: _prune_tree equivalence (WP3 Tier 2)
# ---------------------------------------------------------------------------
def test_prune_tree_equivalence():
    """Verify in-place _prune_tree() matches fresh fit with ccp_alpha."""
    rng = np.random.RandomState(42)
    X = rng.randn(1000, 10).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.int8)

    full_tree = DecisionTreeClassifier(random_state=42)
    full_tree.fit(X, y)
    assert hasattr(full_tree, "_prune_tree"), "sklearn DecisionTreeClassifier lacks _prune_tree"

    path = full_tree.cost_complexity_pruning_path(X, y)
    alphas = [0.0, float(path.ccp_alphas[len(path.ccp_alphas)//3]), float(path.ccp_alphas[2 * len(path.ccp_alphas)//3])]

    for a in alphas:
        # Refit method
        dt_fit = DecisionTreeClassifier(random_state=42, ccp_alpha=a)
        dt_fit.fit(X, y)
        pred_fit = dt_fit.predict(X)

        # In-place prune method
        dt_pruned = copy.deepcopy(full_tree)
        dt_pruned.ccp_alpha = a
        dt_pruned._prune_tree()
        pred_pruned = dt_pruned.predict(X)

        mismatches = int(np.sum(pred_fit != pred_pruned))
        assert mismatches == 0, f"Mismatch at alpha {a}: {mismatches} predictions differ"


# ---------------------------------------------------------------------------
# Test 2: Dedup-weighted tree equivalence (WP3 Tier 3a)
# ---------------------------------------------------------------------------
def test_dedup_weighted_equivalence():
    """Verify training with sample_weight on deduplicated rows matches unweighted."""
    X_base = np.array([
        [10.0, 20.0],
        [10.0, 20.0],
        [10.0, 20.0],
        [5.0, 15.0],
        [5.0, 15.0],
        [30.0, 40.0],
        [30.0, 40.0],
        [30.0, 40.0],
        [30.0, 40.0],
    ], dtype=np.float32)
    y_base = np.array([1, 1, 1, 0, 0, 1, 1, 1, 1], dtype=np.int8)

    df = pd.DataFrame(X_base, columns=["f1", "f2"])
    df["label"] = y_base

    dedup = df.groupby(["f1", "f2", "label"]).size().reset_index(name="weight")
    X_dedup = dedup[["f1", "f2"]].values.astype(np.float32)
    y_dedup = dedup["label"].values.astype(np.int8)
    w_dedup = dedup["weight"].values.astype(np.float64)

    dt_full = DecisionTreeClassifier(random_state=42)
    dt_full.fit(X_base, y_base)

    dt_weighted = DecisionTreeClassifier(random_state=42)
    dt_weighted.fit(X_dedup, y_dedup, sample_weight=w_dedup)

    pred_full = dt_full.predict(X_base)
    pred_weighted = dt_weighted.predict(X_base)

    diffs = int(np.sum(pred_full != pred_weighted))
    assert diffs == 0, f"Deduplicated tree gave {diffs} different predictions"


# ---------------------------------------------------------------------------
# Test 3: confusion_matrix with a single class (WP2 item 4)
# ---------------------------------------------------------------------------
def test_confusion_matrix_single_class():
    """Verify confusion_matrix always produces a 2x2 matrix with single-class y_test."""
    from stages import stage4_evaluate

    rng = np.random.RandomState(42)
    X_test = rng.randn(100, 5).astype(np.float32)
    y_test_single = np.ones(100, dtype=np.int8)  # Single class: all IoT

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_test, y_test_single)

    metrics = stage4_evaluate.run(model, "dt", X_test, y_test_single)
    cm = metrics["confusion_matrix"]

    assert len(cm) == 2, f"CM rows expected 2, got {len(cm)}"
    assert len(cm[0]) == 2, f"CM cols expected 2, got {len(cm[0])}"
    assert len(cm[1]) == 2, f"CM cols expected 2, got {len(cm[1])}"


# ---------------------------------------------------------------------------
# Test 4: Manifest guard fires on column mismatch (WP1 item 5)
# ---------------------------------------------------------------------------
def test_manifest_guard_mismatch(synthetic_data_path):
    """Verify stage 3 raises RuntimeError when manifest kept_cols mismatches table."""
    from stages import stage2_clean_select, stage3_train

    # Set up tag
    test_tag = "test_guard"
    config.refresh_paths(tag=test_tag)

    # Run stage 2 to create selected table
    stage2_clean_select.run(input_path=synthetic_data_path)

    # Corrupt manifest
    manifest.update("features", {
        "kept_cols": ["corrupted_feature_1", "corrupted_feature_2"],
    })

    with pytest.raises(RuntimeError) as exc_info:
        stage3_train.run()

    assert "Feature mismatch" in str(exc_info.value)
    assert "Re-run stage 2" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test 5: Stage idempotence (run stage 3 twice)
# ---------------------------------------------------------------------------
def test_stage_idempotence(synthetic_data_path):
    """Verify running stage 3 twice produces identical alpha and predictions."""
    test_tag = "test_idempotence"
    config.refresh_paths(tag=test_tag)

    from stages import stage2_clean_select, stage3_train

    stage2_clean_select.run(input_path=synthetic_data_path)

    res1 = stage3_train.run()
    alpha1 = res1["ccp_alpha"]
    model1 = res1["model"]

    res2 = stage3_train.run()
    alpha2 = res2["ccp_alpha"]
    model2 = res2["model"]

    assert alpha1 == alpha2, f"Alphas differ between runs: {alpha1} vs {alpha2}"

    X_test, _, _ = io_utils.load_split("test")
    pred1 = model1.predict(X_test)
    pred2 = model2.predict(X_test)
    assert np.array_equal(pred1, pred2), "Predictions differ between idempotent stage 3 runs"


# ---------------------------------------------------------------------------
# Test 6: make_deploy_artifacts works without training_artifacts.joblib (WP2 item 3)
# ---------------------------------------------------------------------------
def test_make_deploy_artifacts_without_training_artifacts(synthetic_data_path):
    """Verify make_deploy_artifacts runs purely from manifest + table."""
    test_tag = "test_deploy"
    config.refresh_paths(tag=test_tag)

    from stages import stage2_clean_select, stage3_train
    stage2_clean_select.run(input_path=synthetic_data_path)
    stage3_train.run()

    # Ensure no training_artifacts.joblib in project root
    art_path = config.PROJECT_ROOT / "training_artifacts.joblib"
    if art_path.exists():
        art_path.unlink()

    import make_deploy_artifacts
    out = make_deploy_artifacts.build()

    assert out.exists()
    payload = joblib.load(out)
    assert "kept_cols" in payload
    assert "X_test" in payload
    assert "y_test" in payload
    assert "test_label_counts" in payload


# ---------------------------------------------------------------------------
# Test 7: Full smoke pipeline: stages 2 to 6 ending with PDF
# ---------------------------------------------------------------------------
def test_full_smoke_pipeline(synthetic_data_path):
    """Verify end-to-end execution of stages 2 through 6 producing a PDF."""
    test_tag = "test_full_pipe"
    config.refresh_paths(tag=test_tag)

    from stages import (
        stage2_clean_select, stage3_train, stage3b_train_rf,
        stage4_evaluate, stage5_benchmark, stage5b_compare, stage6_report
    )

    run_id = "test_smoke_run"

    # Stage 2
    s2 = stage2_clean_select.run(input_path=synthetic_data_path)
    assert Path(s2["output_path"]).exists()

    # Stage 3 (DT)
    s3 = stage3_train.run()
    assert config.DT_MODEL_PATH.exists()

    # Stage 3b (RF)
    s3b = stage3b_train_rf.run()
    assert config.RF_MODEL_PATH.exists()

    # Stage 4 (Eval DT and RF)
    X_test, y_test, meta_test = io_utils.load_split("test")
    dt_model = joblib.load(config.DT_MODEL_PATH)
    rf_model = joblib.load(config.RF_MODEL_PATH)

    dt_m = stage4_evaluate.run(dt_model, "dt", X_test, y_test, meta_test=meta_test, run_id=run_id)
    rf_m = stage4_evaluate.run(rf_model, "rf", X_test, y_test, meta_test=meta_test, run_id=run_id)

    import json
    with open(config.RESULTS_DIR / f"run_{run_id}_metrics_dt.json", "w") as f:
        json.dump(dt_m, f)
    with open(config.RESULTS_DIR / f"run_{run_id}_metrics_rf.json", "w") as f:
        json.dump(rf_m, f)

    # Stage 5 (Benchmark)
    bench = stage5_benchmark.run(run_id=run_id)
    assert (config.RESULTS_DIR / f"run_{run_id}_timing.json").exists()

    # Stage 5b (Compare)
    comp = stage5b_compare.run(run_id=run_id)
    assert (config.RESULTS_DIR / f"run_{run_id}_comparison.json").exists()
    assert (config.RESULTS_DIR / f"run_{run_id}_pareto.json").exists()

    # Stage 6 (Report)
    pdf_path = config.RESULTS_DIR / f"run_{run_id}.pdf"
    stage6_report.run(run_id=run_id, output_path=pdf_path)
    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 1000, "PDF file is suspiciously small"


# ---------------------------------------------------------------------------
# Test 8: Timing sanity (loose tolerance)
# ---------------------------------------------------------------------------
def test_timing_sanity():
    """Verify repeating inference timing twice gives stable measurements within 3x tolerance."""
    from stages.stage5_benchmark import _time_inference_batch

    rng = np.random.RandomState(42)
    X = rng.randn(500, 10).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int8)

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X, y)

    res1 = _time_inference_batch(model, X, [100], n_repeats=5)
    res2 = _time_inference_batch(model, X, [100], n_repeats=5)

    t1 = res1["100"]["median_us_per_pkt"]
    t2 = res2["100"]["median_us_per_pkt"]

    assert t1 > 0 and t2 > 0
    ratio = max(t1, t2) / min(t1, t2)
    assert ratio < 3.0, f"Timing unstable: {t1:.2f} vs {t2:.2f} (ratio {ratio:.2f} >= 3.0)"
