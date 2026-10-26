"""
stage6_report.py — PDF report generation using fpdf2.

Reads from JSON files (metrics, timing, comparison, pareto, manifest).
Fixed ln=True deprecation (new_x="LMARGIN", new_y="NEXT").
"""

import json
import logging
import os
from pathlib import Path

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import config
import manifest
from stages import stage5c_resource

logger = logging.getLogger(__name__)


def run(run_data=None, output_path=None, run_id=None):
    """
    Generate a PDF report.

    Can be called with:
    - run_data dict (legacy), or
    - run_id to read from JSON files on disk (preferred, WP6).
    """
    from fpdf import FPDF

    logger.info("=== STAGE 6: PDF Report Generation ===")

    # --- Load data from files if run_id provided ---
    if run_data is None and run_id is not None:
        run_data = _load_run_data(run_id)

    if run_data is None:
        raise ValueError("Either run_data or run_id must be provided")

    if output_path is None:
        run_id_str = run_data.get("run_id", run_id or "unknown")
        output_path = config.RESULTS_DIR / f"run_{run_id_str}.pdf"

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "IoT Device Fingerprinting - Results Summary",
             new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(5)

    # Helper: section title
    def section_title(title):
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

    # Helper: body text
    def body_text(text):
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 5, text)
        pdf.ln(2)

    # Helper: simple table
    def simple_table(headers, rows, col_widths=None):
        pdf.set_font("Helvetica", "B", 9)
        if col_widths is None:
            w = (pdf.w - pdf.l_margin - pdf.r_margin) / len(headers)
            col_widths = [w] * len(headers)
        for h, cw in zip(headers, col_widths):
            pdf.cell(cw, 7, str(h), border=1, align="C")
        pdf.ln()
        pdf.set_font("Helvetica", "", 9)
        for row in rows:
            for val, cw in zip(row, col_widths):
                pdf.cell(cw, 6, str(val), border=1, align="C")
            pdf.ln()
        pdf.ln(3)

    # Read manifest
    m = manifest.read()
    feat_section = m.get("features", {})
    dt_section = m.get("dt", {})
    rf_section = m.get("rf", {})

    # -------------------------------------------------------------------
    # Section 1: Train / Test Assignment
    # -------------------------------------------------------------------
    section_title("1. Train / Test Assignment")
    train_days = feat_section.get("train_days", sorted(list(config.TRAIN_DAYS)))
    test_days = feat_section.get("test_days", sorted(list(config.TEST_DAYS)))
    n_train = feat_section.get("n_train_rows", 0)
    n_test = feat_section.get("n_test_rows", 0)
    simple_table(
        ["Split", "Days", "Packets"],
        [
            ["Train", ", ".join(sorted(train_days) if isinstance(train_days, list) else [str(train_days)]), f"{n_train:,}"],
            ["Test", ", ".join(sorted(test_days) if isinstance(test_days, list) else [str(test_days)]), f"{n_test:,}"],
        ],
        col_widths=[25, 110, 45],
    )

    # -------------------------------------------------------------------
    # Section 2: Label Distribution
    # -------------------------------------------------------------------
    section_title("2. Label Distribution")
    train_lc = feat_section.get("train_label_counts", {})
    test_lc = feat_section.get("test_label_counts", {})
    body_text("Train:")
    for lbl, cnt in train_lc.items():
        body_text(f"  - {lbl}: {cnt:,}")
    body_text("Test:")
    for lbl, cnt in test_lc.items():
        body_text(f"  - {lbl}: {cnt:,}")

    # -------------------------------------------------------------------
    # Section 3: Missing Value Handling
    # -------------------------------------------------------------------
    section_title("3. Missing Value Handling")
    nan_fill = feat_section.get("nan_fill", config.NAN_FILL)
    body_text(
        f"NaN values in protocol-specific fields were filled with {nan_fill}. "
        f"This sentinel value indicates 'field not applicable' (e.g., TCP fields "
        f"absent in UDP packets)."
    )

    # -------------------------------------------------------------------
    # Section 4: Feature Set
    # -------------------------------------------------------------------
    section_title("4. Feature Set")
    body_text(f"Total candidate features: {len(config.FEATURE_COLS)}")
    zv = feat_section.get("pruned_zero_var", [])
    body_text(f"Dropped - zero variance ({len(zv)}): {', '.join(zv) if zv else 'none'}")
    li = feat_section.get("pruned_low_imp", [])
    body_text(f"Dropped - RF importance < {config.RF_IMPORTANCE_THRESHOLD} ({len(li)}): {', '.join(li) if li else 'none'}")
    kept = feat_section.get("kept_cols", [])
    body_text(f"Final feature set ({len(kept)}):")
    for c in kept:
        body_text(f"  - {c}")

    # -------------------------------------------------------------------
    # Section 5: Decision Tree Properties
    # -------------------------------------------------------------------
    section_title("5. Decision Tree Properties")
    body_text("Criterion: Gini impurity")
    ccp = dt_section.get("ccp_alpha", 0.0)
    cv_f1 = dt_section.get("cv_f1", 0.0)
    if ccp > 0:
        body_text(f"Cost-complexity pruning alpha: {ccp:.8f}")
    body_text(f"CV F1 at selected alpha: {cv_f1:.6f}")
    body_text(f"Tree depth: {dt_section.get('depth', 'N/A')}")
    body_text(f"Number of leaves: {dt_section.get('leaves', 'N/A')}")
    body_text(f"Unpruned depth: {dt_section.get('unpruned_depth', 'N/A')}")
    body_text(f"Unpruned leaves: {dt_section.get('unpruned_leaves', 'N/A')}")
    body_text(f"1-SE rule applied: {dt_section.get('one_se_rule', 'N/A')}")

    # -------------------------------------------------------------------
    # Section 6: Evaluation Metrics (both models)
    # -------------------------------------------------------------------
    section_title("6. Evaluation Metrics")

    for model_name in config.MODEL_NAMES:
        metrics = run_data.get("metrics", {}).get(model_name, {})
        if not metrics:
            continue

        display = config.MODEL_DISPLAY_NAMES.get(model_name, model_name)
        body_text(f"--- {display} ---")
        simple_table(
            ["Metric", "Value"],
            [
                ["Accuracy", f"{metrics.get('accuracy', 0):.4f}"],
                ["Precision (IoT)", f"{metrics.get('precision', 0):.4f}"],
                ["Recall (IoT)", f"{metrics.get('recall', 0):.4f}"],
                ["F1 (IoT)", f"{metrics.get('f1', 0):.4f}"],
                ["Specificity", f"{metrics.get('specificity', 0):.4f}"],
                ["Balanced Accuracy", f"{metrics.get('balanced_accuracy', 0):.4f}"],
                ["MCC", f"{metrics.get('mcc', 0):.4f}"],
                ["ROC-AUC", f"{metrics.get('roc_auc', 'N/A')}"],
                ["PR-AUC", f"{metrics.get('pr_auc', 'N/A')}"],
            ],
            col_widths=[70, 50],
        )

        # Confusion matrix
        cm = metrics.get("confusion_matrix", [[0, 0], [0, 0]])
        body_text(f"{display} Confusion Matrix:")
        simple_table(
            ["", "Pred non-IoT", "Pred IoT"],
            [
                ["Actual non-IoT", f"{cm[0][0]:,}", f"{cm[0][1]:,}"],
                ["Actual IoT", f"{cm[1][0]:,}", f"{cm[1][1]:,}"],
            ],
            col_widths=[50, 45, 45],
        )

    # -------------------------------------------------------------------
    # Section 7: Timing Evaluation
    # -------------------------------------------------------------------
    section_title("7. Timing Evaluation")
    timing = run_data.get("timing", {})
    if timing:
        for model_name in config.MODEL_NAMES:
            model_timing = timing.get("models", {}).get(model_name, {})
            if not model_timing:
                continue

            display = config.MODEL_DISPLAY_NAMES.get(model_name, model_name)
            body_text(f"--- {display} ---")

            # Static info
            static = model_timing.get("static", {})
            if "depth" in static:
                body_text(f"  Depth: {static['depth']}, Leaves: {static['leaves']}")
            if "n_estimators" in static:
                body_text(f"  Trees: {static['n_estimators']}, Total nodes: {static['total_nodes']}")
            body_text(f"  Model size: {static.get('model_size_kb', 'N/A')} KB")

            # Training cost
            cost = model_timing.get("training_cost", {})
            body_text(f"  Tuning time: {cost.get('tuning_time_s', 0):.2f} s")
            body_text(f"  Final fit time: {cost.get('fit_time_s', 0):.2f} s")
            body_text(f"  Total training cost: {cost.get('total_training_cost_s', 0):.2f} s")

            # Single-packet
            single = model_timing.get("inference_single", {})
            if single:
                body_text(f"  Single-packet: p50={single.get('p50_us', 0):.1f} us, "
                          f"p95={single.get('p95_us', 0):.1f} us, "
                          f"p99={single.get('p99_us', 0):.1f} us")
    else:
        body_text("No timing data available.")

    body_text("Note: Feature extraction cost is NOT included in these timings.")

    # -----------------------------------------------------------------------
    # Section 7b: CPU and Memory Profiling for Testing / Inference
    # -----------------------------------------------------------------------
    # Section 7b: CPU and memory benchmark (stage 5c, fresh-process, model-only)
    stage5c_resource.add_report_section(pdf, run_data, body_text, simple_table)

    # -------------------------------------------------------------------
    # Section 8: Per-Day Metrics
    # -------------------------------------------------------------------
    section_title("8. Per-Day Metrics")
    for model_name in config.MODEL_NAMES:
        metrics = run_data.get("metrics", {}).get(model_name, {})
        per_day = metrics.get("per_day", {})
        if not per_day:
            continue

        display = config.MODEL_DISPLAY_NAMES.get(model_name, model_name)
        body_text(f"--- {display} ---")
        rows = []
        for day in sorted(per_day.keys()):
            d = per_day[day]
            rows.append([day, str(d.get("n_rows", 0)), f"{d.get('accuracy', 0):.4f}",
                        f"{d.get('f1', 0):.4f}"])
        if rows:
            simple_table(["Day", "Rows", "Accuracy", "F1"], rows,
                        col_widths=[40, 30, 35, 35])

    # -------------------------------------------------------------------
    # Section 9: Model Comparison (if available)
    # -------------------------------------------------------------------
    comp = run_data.get("comparison")
    if comp:
        pdf.add_page()
        section_title("9. Model Comparison")

        # RF seed variance
        seed_var = comp.get("rf_seed_variance")
        if seed_var:
            body_text(f"RF Seed Variance (seeds={seed_var['seeds']}): "
                      f"F1 = {seed_var['mean_f1']:.6f} +/- {seed_var['std_f1']:.6f}")

        # Embed graphs
        for graph_name in ["metrics_comparison", "confusion_matrices",
                           "inference_latency", "training_time", "pareto", "model_size",
                           "resource_cpu", "resource_memory"]:
            filename = comp.get("graphs", {}).get(graph_name)
            if filename:
                img_path = config.RESULTS_DIR / filename
                if os.path.exists(img_path):
                    pdf.add_page()
                    page_width = pdf.w - pdf.l_margin - pdf.r_margin
                    pdf.image(str(img_path), x=pdf.l_margin, w=page_width)
                    pdf.ln(3)

    # -------------------------------------------------------------------
    # Notes
    # -------------------------------------------------------------------
    pdf.add_page()
    section_title("Notes")

    notes = [
        "DT was tuned via cost-complexity pruning with GroupKFold CV and 1-SE rule.",
        f"RF was tuned via grid search on {config.RF_TUNE_FRAC*100:.0f}% subsample with GroupKFold CV.",
        "RF and DT share the same fold definition (GroupKFold by day).",
    ]

    if comp and comp.get("rf_seed_variance"):
        sv = comp["rf_seed_variance"]
        notes.append(f"RF seed variance across seeds {sv['seeds']}: "
                     f"F1 std = {sv['std_f1']:.6f}.")

    notes.append("Feature extraction time is excluded from all timing figures.")
    notes.append(f"Hardware: {os.cpu_count()} cores, {sys.platform}")

    # Check if dedup was applied (from manifest)
    dt_sec = manifest.read().get("dt", {})
    if dt_sec.get("dedup_applied"):
        notes.append("DT training used deduplicated rows with sample weights.")

    for note in notes:
        body_text(f"- {note}")

    # -------------------------------------------------------------------
    # Write PDF
    # -------------------------------------------------------------------
    pdf.output(str(output_path))
    logger.info("PDF report written to: %s", output_path)
    return {"pdf_path": str(output_path)}


def _load_run_data(run_id):
    """Load run data from JSON files on disk."""
    run_data = {"run_id": run_id, "metrics": {}, "timing": {}, "comparison": None}

    # Load metrics for each model
    for model_name in config.MODEL_NAMES:
        json_path = config.RESULTS_DIR / f"run_{run_id}_metrics_{model_name}.json"
        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                run_data["metrics"][model_name] = json.load(f)

    # Load timing
    timing_path = config.RESULTS_DIR / f"run_{run_id}_timing.json"
    if timing_path.exists():
        with open(timing_path, "r", encoding="utf-8") as f:
            run_data["timing"] = json.load(f)

    # Load comparison
    comp_path = config.RESULTS_DIR / f"run_{run_id}_comparison.json"
    if comp_path.exists():
        with open(comp_path, "r", encoding="utf-8") as f:
            run_data["comparison"] = json.load(f)

    # Load pareto
    pareto_path = config.RESULTS_DIR / f"run_{run_id}_pareto.json"
    if pareto_path.exists():
        with open(pareto_path, "r", encoding="utf-8") as f:
            run_data["pareto"] = json.load(f)

    resource_path = config.RESULTS_DIR / f"run_{run_id}_resource.json"
    if resource_path.exists():
        with open(resource_path, "r", encoding="utf-8") as f:
            run_data["resource"] = json.load(f)

    return run_data
