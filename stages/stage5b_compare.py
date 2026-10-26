"""
stage5b_compare.py — Two-model comparison (DT vs RF).

Reads metrics, timing, and manifest JSONs to build comparison data.
Produces graphs: metrics_comparison, confusion_matrices, inference_latency,
training_time, pareto, model_size.
"""

import os
import gc
import time
import json
import tempfile
import logging

import numpy as np
import joblib

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import config
import io_utils
import manifest
from stages import stage5c_resource

logger = logging.getLogger(__name__)

# Model colours — consistent across all graphs
MODEL_COLORS = {"dt": "#2196F3", "rf": "#4CAF50"}


def _generate_pareto_data(run_id):
    """Generate Pareto front data: F1 vs inference µs/packet for DT and RF."""
    pareto = {}

    X_test, y_test, _ = io_utils.load_split("test")
    X_test = np.ascontiguousarray(X_test, dtype=np.float32)

    # --- DT Pareto: multiple alphas ---
    dt_path = config.DT_MODEL_PATH
    if dt_path.exists():
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.metrics import f1_score as f1

        m = manifest.read()
        dt_section = m.get("dt", {})
        cv_curve = dt_section.get("cv_curve", [])

        # Pick 5-8 alphas from the CV curve
        if cv_curve:
            n_pick = min(8, len(cv_curve))
            indices = np.linspace(0, len(cv_curve) - 1, n_pick, dtype=int)
            selected_alphas = [cv_curve[i]["alpha"] for i in indices]
        else:
            selected_alphas = [0.0]

        X_train, y_train, _ = io_utils.load_split("train")
        X_train = np.ascontiguousarray(X_train, dtype=np.float32)

        dt_points = []
        for alpha in selected_alphas:
            dt = DecisionTreeClassifier(criterion="gini", random_state=42, ccp_alpha=alpha)
            dt.fit(X_train, y_train)
            y_pred = dt.predict(X_test)
            test_f1 = float(f1(y_test, y_pred, zero_division=0))

            # Inference time
            gc.collect()
            gc.disable()
            try:
                times = []
                for _ in range(5):
                    t0 = time.perf_counter()
                    dt.predict(X_test)
                    times.append((time.perf_counter() - t0) / len(X_test) * 1e6)
            finally:
                gc.enable()

            dt_points.append({
                "alpha": alpha,
                "leaves": dt.get_n_leaves(),
                "depth": dt.get_depth(),
                "test_f1": test_f1,
                "inference_us_per_pkt": float(np.median(times)),
            })
        pareto["dt"] = dt_points
        del X_train, y_train

    # --- RF Pareto: prefix subsets of estimators ---
    rf_path = config.RF_MODEL_PATH
    if rf_path.exists():
        from sklearn.metrics import f1_score as f1

        rf = joblib.load(rf_path)
        n_est = len(rf.estimators_)
        ks = sorted(set([1, 5, 10, 25, 50] + [n_est]))
        ks = [k for k in ks if k <= n_est]

        rf_points = []
        for k in ks:
            # Average predict_proba of first k trees
            probas = np.zeros(len(X_test))
            for tree in rf.estimators_[:k]:
                probas += tree.predict_proba(X_test)[:, 1]
            probas /= k
            y_pred = (probas >= 0.5).astype(int)
            test_f1 = float(f1(y_test, y_pred, zero_division=0))

            # Inference time
            gc.collect()
            gc.disable()
            try:
                times = []
                for _ in range(5):
                    t0 = time.perf_counter()
                    p = np.zeros(len(X_test))
                    for tree in rf.estimators_[:k]:
                        p += tree.predict_proba(X_test)[:, 1]
                    p /= k
                    times.append((time.perf_counter() - t0) / len(X_test) * 1e6)
            finally:
                gc.enable()

            rf_points.append({
                "n_trees": k,
                "test_f1": test_f1,
                "inference_us_per_pkt": float(np.median(times)),
            })
        pareto["rf"] = rf_points

    return pareto


# ---------------------------------------------------------------------------
# Graph functions — each failure is logged, not propagated
# ---------------------------------------------------------------------------

def _graph_metrics_comparison(comp, run_id):
    """Grouped bar chart: accuracy, precision, recall, F1."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        metrics_names = ["accuracy", "precision", "recall", "f1"]
        models = [m for m in config.MODEL_NAMES if m in comp.get("models", {})]
        x = np.arange(len(metrics_names))
        width = 0.3

        fig, ax = plt.subplots(figsize=(10, 6))
        for i, m in enumerate(models):
            vals = [comp["models"][m]["metrics"].get(mn, 0) for mn in metrics_names]
            bars = ax.bar(x + i * width, vals, width,
                          label=config.MODEL_DISPLAY_NAMES.get(m, m), color=MODEL_COLORS.get(m, "#999"))
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                        f"{v:.4f}", ha="center", va="bottom", fontsize=8)

        ax.set_ylabel("Score")
        ax.set_title("Model Comparison — Classification Metrics")
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels([n.title() for n in metrics_names])
        all_vals = [comp["models"][m]["metrics"].get(mn, 0)
                    for m in models for mn in metrics_names]
        y_min = max(0, min(all_vals) - 0.05)
        ax.set_ylim(y_min, 1.01)
        ax.legend()
        plt.tight_layout()

        path = config.RESULTS_DIR / f"run_{run_id}_metrics_comparison.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info("Graph saved: %s", path)
        return path.name
    except Exception as e:
        logger.error("Failed to generate metrics_comparison graph: %s", e)
        return None


def _graph_confusion_matrices(comp, run_id):
    """Side-by-side confusion matrix heatmaps."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        models = [m for m in config.MODEL_NAMES if m in comp.get("models", {})]
        fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5))
        if len(models) == 1:
            axes = [axes]
        labels = ["non-IoT", "IoT"]

        for ax, m in zip(axes, models):
            cm = np.array(comp["models"][m]["metrics"]["confusion_matrix"])
            row_sums = cm.sum(axis=1, keepdims=True)
            row_pct = cm / np.where(row_sums == 0, 1, row_sums) * 100

            ax.imshow(cm, cmap="Blues")
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(labels)
            ax.set_yticklabels(labels)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title(config.MODEL_DISPLAY_NAMES.get(m, m))

            for i in range(2):
                for j in range(2):
                    ax.text(j, i, f"{cm[i, j]:,}\n({row_pct[i, j]:.1f}%)",
                            ha="center", va="center",
                            color="white" if cm[i, j] > cm.max() / 2 else "black",
                            fontsize=9)

        fig.suptitle("Confusion Matrices", fontsize=14)
        plt.tight_layout()

        path = config.RESULTS_DIR / f"run_{run_id}_confusion_matrices.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info("Graph saved: %s", path)
        return path.name
    except Exception as e:
        logger.error("Failed to generate confusion_matrices graph: %s", e)
        return None


def _graph_inference_latency(comp, run_id):
    """Inference latency from timing data."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        timing = comp.get("timing", {})
        models = [m for m in config.MODEL_NAMES if m in timing.get("models", {})]

        if not models:
            logger.warning("No timing data for inference_latency graph")
            return None

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Single-packet p50
        vals_single = []
        for m in models:
            single = timing["models"][m].get("inference_single", {})
            vals_single.append(single.get("p50_us", 0))

        bars1 = ax1.bar([config.MODEL_DISPLAY_NAMES.get(m, m) for m in models], vals_single,
                        color=[MODEL_COLORS.get(m, "#999") for m in models])
        ax1.set_ylabel("Latency (µs)")
        ax1.set_title("Single-Packet Inference (p50)")
        for bar, v in zip(bars1, vals_single):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{v:.1f}", ha="center", va="bottom", fontsize=9)

        # Batch full-set
        vals_batch = []
        for m in models:
            batch = timing["models"][m].get("inference_batch", {})
            # Use the largest batch size
            batch_int_keys = [k for k in batch.keys() if str(k).isdigit()]
            if batch_int_keys:
                largest_key = max(batch_int_keys, key=int)
                vals_batch.append(batch[largest_key].get("median_us_per_pkt", 0))
            else:
                vals_batch.append(0)

        bars2 = ax2.bar([config.MODEL_DISPLAY_NAMES.get(m, m) for m in models], vals_batch,
                        color=[MODEL_COLORS.get(m, "#999") for m in models])
        ax2.set_ylabel("Latency (µs/pkt)")
        ax2.set_title("Batch Inference (largest batch)")
        for bar, v in zip(bars2, vals_batch):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{v:.4f}", ha="center", va="bottom", fontsize=9)

        fig.suptitle("Inference Latency Comparison", fontsize=14)
        plt.tight_layout()

        path = config.RESULTS_DIR / f"run_{run_id}_inference_latency.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info("Graph saved: %s", path)
        return path.name
    except Exception as e:
        logger.error("Failed to generate inference_latency graph: %s", e)
        return None


def _graph_training_time(comp, run_id):
    """Training time comparison: tuning + fit, by model."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        m_data = manifest.read()
        models = [m for m in config.MODEL_NAMES if m in m_data]

        fig, ax = plt.subplots(figsize=(8, 5))
        tuning_times = []
        fit_times = []
        labels = []

        for m in models:
            section = m_data.get(m, {})
            tune = section.get("tuning_time_s", 0)
            fit = section.get("fit_time_s", 0)
            tuning_times.append(tune)
            fit_times.append(fit)
            labels.append(config.MODEL_DISPLAY_NAMES.get(m, m))

        x = np.arange(len(models))
        width = 0.35
        ax.bar(x - width / 2, tuning_times, width, label="Tuning", color="#FF9800")
        ax.bar(x + width / 2, fit_times, width, label="Final Fit", color="#2196F3")
        ax.set_ylabel("Time (s)")
        ax.set_title("Training Time Breakdown")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

        plt.tight_layout()
        path = config.RESULTS_DIR / f"run_{run_id}_training_time.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info("Graph saved: %s", path)
        return path.name
    except Exception as e:
        logger.error("Failed to generate training_time graph: %s", e)
        return None


def _graph_pareto(comp, run_id):
    """Pareto plot: F1 vs inference µs/packet for DT and RF."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        pareto = comp.get("pareto", {})
        if not pareto:
            return None

        fig, ax = plt.subplots(figsize=(8, 6))

        if "dt" in pareto:
            pts = pareto["dt"]
            xs = [p["inference_us_per_pkt"] for p in pts]
            ys = [p["test_f1"] for p in pts]
            ax.plot(xs, ys, "o-", color=MODEL_COLORS["dt"], label="Decision Tree", markersize=6)

        if "rf" in pareto:
            pts = pareto["rf"]
            xs = [p["inference_us_per_pkt"] for p in pts]
            ys = [p["test_f1"] for p in pts]
            ax.plot(xs, ys, "s-", color=MODEL_COLORS["rf"], label="Random Forest", markersize=6)

        ax.set_xlabel("Inference µs/packet")
        ax.set_ylabel("Test F1")
        ax.set_title("Pareto: F1 vs Inference Cost")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        path = config.RESULTS_DIR / f"run_{run_id}_pareto.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info("Graph saved: %s", path)
        return path.name
    except Exception as e:
        logger.error("Failed to generate pareto graph: %s", e)
        return None


def _graph_model_size(comp, run_id):
    """Model size on disk comparison."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        timing = comp.get("timing", {})
        models = [m for m in config.MODEL_NAMES if m in timing.get("models", {})]

        if not models:
            return None

        vals = []
        labels = []
        for m in models:
            static = timing["models"][m].get("static", {})
            vals.append(static.get("model_size_kb", 0))
            labels.append(config.MODEL_DISPLAY_NAMES.get(m, m))

        fig, ax = plt.subplots(figsize=(6, 5))
        bars = ax.bar(labels, vals, color=[MODEL_COLORS.get(m, "#999") for m in models])
        ax.set_ylabel("Size (KB)")
        ax.set_title("Model Size on Disk")
        ax.set_yscale("log")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{v:.1f}", ha="center", va="bottom", fontsize=9)
        plt.tight_layout()

        path = config.RESULTS_DIR / f"run_{run_id}_model_size.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info("Graph saved: %s", path)
        return path.name
    except Exception as e:
        logger.error("Failed to generate model_size graph: %s", e)
        return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(run_id=None, models_dict=None, metrics_dict=None,
        X_test_full=None, X_train_dict=None, y_train_dict=None):
    """
    Two-model comparison by reading from JSON files.

    Parameters
    ----------
    run_id : str
        Run identifier. Used to locate metrics and timing JSONs.

    Returns
    -------
    comparison dict (also written to JSON).
    """
    t_start = time.perf_counter()
    logger.info("=== STAGE 5b: Two-Model Comparison ===")

    # --- Read metrics JSONs ---
    comparison = {
        "models": {},
        "graphs": {},
    }

    for model_name in config.MODEL_NAMES:
        json_path = config.RESULTS_DIR / f"run_{run_id}_metrics_{model_name}.json"
        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            comparison["models"][model_name] = {
                "display_name": config.MODEL_DISPLAY_NAMES.get(model_name, model_name),
                "metrics": metrics,
            }
            logger.info("Loaded metrics for %s", model_name)
        else:
            logger.warning("Metrics JSON not found for %s at %s", model_name, json_path)

    # --- Read timing JSON ---
    timing_path = config.RESULTS_DIR / f"run_{run_id}_timing.json"
    timing = {}
    if timing_path.exists():
        with open(timing_path, "r", encoding="utf-8") as f:
            timing = json.load(f)
        comparison["timing"] = timing
        logger.info("Loaded timing data")
    else:
        logger.warning("Timing JSON not found at %s", timing_path)

    resource_path = config.RESULTS_DIR / f"run_{run_id}_resource.json"
    if resource_path.exists():
        with open(resource_path, "r", encoding="utf-8") as f:
            comparison["resource"] = json.load(f)
        logger.info("Loaded resource benchmark data")
    else:
        logger.warning("Resource JSON not found at %s (run stage 5c)", resource_path)

    # --- Generate Pareto data ---
    logger.info("Generating Pareto data...")
    pareto = _generate_pareto_data(run_id)
    comparison["pareto"] = pareto

    # Write Pareto JSON
    pareto_path = config.RESULTS_DIR / f"run_{run_id}_pareto.json"
    with open(pareto_path, "w", encoding="utf-8") as f:
        json.dump(pareto, f, indent=2, default=str)
    logger.info("Pareto JSON written: %s", pareto_path)

    # --- RF seed variance ---
    rf_path = config.RF_MODEL_PATH
    if rf_path.exists():
        logger.info("Computing RF seed variance...")
        X_test, y_test, _ = io_utils.load_split("test")
        X_train, y_train, _ = io_utils.load_split("train")
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import f1_score as f1

        m_data = manifest.read()
        rf_section = m_data.get("rf", {})
        best_params = rf_section.get("best_params", {})

        seed_f1s = []
        for seed in config.RF_EVAL_SEEDS:
            rf_seed = RandomForestClassifier(
                n_estimators=config.RF_N_ESTIMATORS,
                criterion="gini",
                min_samples_leaf=best_params.get("min_samples_leaf", config.RF_MIN_SAMPLES_LEAF),
                max_features=best_params.get("max_features", "sqrt"),
                n_jobs=-1,
                random_state=seed,
            )
            rf_seed.fit(X_train, y_train)
            y_pred = rf_seed.predict(X_test)
            seed_f1s.append(float(f1(y_test, y_pred, zero_division=0)))

        comparison["rf_seed_variance"] = {
            "seeds": config.RF_EVAL_SEEDS,
            "f1_scores": seed_f1s,
            "mean_f1": float(np.mean(seed_f1s)),
            "std_f1": float(np.std(seed_f1s)),
        }
        logger.info("RF seed variance: F1=%.6f +/- %.6f", np.mean(seed_f1s), np.std(seed_f1s))

    # --- Generate graphs ---
    graph_fns = [
        ("metrics_comparison", _graph_metrics_comparison),
        ("confusion_matrices", _graph_confusion_matrices),
        ("inference_latency", _graph_inference_latency),
        ("training_time", _graph_training_time),
        ("pareto", _graph_pareto),
        ("model_size", _graph_model_size),
        ("resource_cpu", stage5c_resource.graph_cpu),
        ("resource_memory", stage5c_resource.graph_memory),
    ]
    for graph_name, graph_fn in graph_fns:
        result = graph_fn(comparison, run_id)
        if result is not None:
            comparison["graphs"][graph_name] = result

    # --- Write comparison JSON ---
    comp_path = config.RESULTS_DIR / f"run_{run_id}_comparison.json"
    with open(comp_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, default=str)
    logger.info("Comparison JSON written: %s", comp_path)

    elapsed = time.perf_counter() - t_start
    logger.info("Stage 5b completed in %.1f s", elapsed)

    return comparison
