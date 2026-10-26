"""
stage5_benchmark.py — Time-focused resource evaluation for DT and RF.

Removed: psutil, tracemalloc, cpu_*, mem_* fields.
Added: controlled training-time scaling, batch-size sweep, single-packet timing
with proper gc controls, for both DT and RF models.

Output: run_<id>_timing.json
"""

import os
import gc
import copy
import time
import json
import platform
import tempfile
import logging
import threading

import numpy as np
import joblib
import psutil

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import config
import io_utils
import manifest

logger = logging.getLogger(__name__)


def _get_process_tree():
    """Return a list of psutil.Process objects for current process and recursive descendants."""
    procs = []
    try:
        parent = psutil.Process()
        procs.append(parent)
        try:
            children = parent.children(recursive=True)
            procs.extend(children)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    except Exception:
        pass
    unique = {}
    for p in procs:
        if p.pid not in unique:
            unique[p.pid] = p
    return list(unique.values())


def _snapshot_cpu_times():
    """Sum user + system CPU seconds across current process and recursive children."""
    total = 0.0
    for p in _get_process_tree():
        try:
            ct = p.cpu_times()
            total += (ct.user + ct.system)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
        except Exception:
            continue
    return float(total)


def _snapshot_rss_bytes():
    """Sum RSS bytes across current process and recursive children."""
    total = 0
    for p in _get_process_tree():
        try:
            total += p.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
        except Exception:
            continue
    return int(total)


class ResourceSampler(threading.Thread):
    """Background sampler thread tracking peak RSS across the process tree."""
    def __init__(self, interval_s=0.05):
        super().__init__(daemon=True)
        self.interval_s = interval_s
        self._stop_event = threading.Event()
        self.peak_rss_bytes = 0
        self.sample_count = 0
        self._lock = threading.Lock()

    def run(self):
        while not self._stop_event.is_set():
            try:
                rss = _snapshot_rss_bytes()
                with self._lock:
                    if rss > self.peak_rss_bytes:
                        self.peak_rss_bytes = rss
                    self.sample_count += 1
            except Exception:
                pass
            self._stop_event.wait(self.interval_s)

    def stop(self):
        self._stop_event.set()


def profile_workload(workload_fn, workload_name, config_dict, n_repeats=3, sampling_interval_s=0.05):
    """
    Profile CPU time, CPU utilization, and process-tree RSS for a given workload.
    """
    repeats = []
    for _ in range(n_repeats):
        rss_before = _snapshot_rss_bytes()
        cpu_before = _snapshot_cpu_times()
        sampler = ResourceSampler(interval_s=sampling_interval_s)

        gc.collect()
        gc.disable()
        t_wall0 = time.perf_counter()
        sampler.start()
        try:
            workload_fn()
        finally:
            t_wall1 = time.perf_counter()
            gc.enable()
            sampler.stop()
            sampler.join(timeout=2.0)
            rss_after = _snapshot_rss_bytes()
            cpu_after = _snapshot_cpu_times()

        wall_time_s = float(t_wall1 - t_wall0)
        cpu_time_s_total = float(max(0.0, cpu_after - cpu_before))
        rss_peak = int(max(rss_before, sampler.peak_rss_bytes, rss_after))
        rss_peak_delta = int(max(0, rss_peak - rss_before))
        rss_after_delta = int(rss_after - rss_before)
        sample_count = sampler.sample_count

        if wall_time_s < sampling_interval_s:
            cpu_util = None
            reason = "duration_below_sampling_resolution"
        elif wall_time_s > 0:
            cpu_util = float(100.0 * cpu_time_s_total / wall_time_s)
            reason = None
        else:
            cpu_util = None
            reason = "zero_wall_time"

        repeats.append({
            "wall_time_s": wall_time_s,
            "cpu_time_s_total": cpu_time_s_total,
            "cpu_utilization_avg_pct": cpu_util,
            "rss_before_bytes": rss_before,
            "rss_peak_bytes": rss_peak,
            "rss_after_bytes": rss_after,
            "rss_peak_delta_bytes": rss_peak_delta,
            "rss_after_delta_bytes": rss_after_delta,
            "sample_count": sample_count,
            "unavailable_reason": reason,
        })

    # Summary across repeats
    cpu_times = [r["cpu_time_s_total"] for r in repeats]
    valid_utils = [r["cpu_utilization_avg_pct"] for r in repeats if r["cpu_utilization_avg_pct"] is not None]
    peak_deltas = [r["rss_peak_delta_bytes"] for r in repeats]
    peaks = [r["rss_peak_bytes"] for r in repeats]
    after_deltas = [r["rss_after_delta_bytes"] for r in repeats]

    summary = {
        "n_repeats": len(repeats),
        "cpu_time_median_s": float(np.median(cpu_times)),
        "cpu_utilization_median_pct": float(np.median(valid_utils)) if valid_utils else None,
        "cpu_utilization_p25_pct": float(np.percentile(valid_utils, 25)) if valid_utils else None,
        "cpu_utilization_p75_pct": float(np.percentile(valid_utils, 75)) if valid_utils else None,
        "rss_peak_median_bytes": int(np.median(peaks)),
        "rss_peak_delta_median_bytes": int(np.median(peak_deltas)),
        "rss_peak_delta_p25_bytes": int(np.percentile(peak_deltas, 25)),
        "rss_peak_delta_p75_bytes": int(np.percentile(peak_deltas, 75)),
        "rss_after_delta_median_bytes": int(np.median(after_deltas)),
    }

    return {
        "workload": workload_name,
        "configuration": config_dict,
        "repeats": repeats,
        "summary": summary,
    }


def _get_env_block():
    """Build environment descriptor block."""
    import sklearn
    env = {
        "cpu_model": platform.processor(),
        "cpu_cores": os.cpu_count(),
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "platform": platform.platform(),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "sklearn_version": sklearn.__version__,
    }
    # Try to read /proc/cpuinfo for more detail (Linux)
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    env["cpu_model_name"] = line.split(":")[1].strip()
                    break
    except Exception:
        pass
    return env


def _time_inference_batch(model, X, batch_sizes, n_repeats, n_jobs=1, sampling_interval_s=0.05):
    """Batch-size sweep: report µs/packet (median and IQR) and resource profiling."""
    results = {}
    resource_by_batch = {}
    n_total = len(X)

    for bs in batch_sizes:
        if bs > n_total:
            continue
        X_batch = np.ascontiguousarray(X[:bs], dtype=np.float32)

        # Warm-up (1 run)
        try:
            model.predict(X_batch)
        except Exception:
            pass

        # Timing loop
        times_per_pkt = []
        gc.collect()
        gc.disable()
        try:
            for _ in range(n_repeats):
                t0 = time.perf_counter()
                model.predict(X_batch)
                elapsed = time.perf_counter() - t0
                times_per_pkt.append(elapsed / bs * 1e6)  # µs/pkt
        finally:
            gc.enable()

        times_per_pkt = np.array(times_per_pkt)
        results[str(bs)] = {
            "median_us_per_pkt": float(np.median(times_per_pkt)),
            "q25_us_per_pkt": float(np.percentile(times_per_pkt, 25)),
            "q75_us_per_pkt": float(np.percentile(times_per_pkt, 75)),
        }

        # Resource profiling across 3 repeats
        def predict_batch_workload():
            model.predict(X_batch)

        res_obj = profile_workload(
            workload_fn=predict_batch_workload,
            workload_name="inference_batch",
            config_dict={
                "n_jobs": n_jobs,
                "batch_size": bs,
                "operations_per_measurement": 1,
            },
            n_repeats=3,
            sampling_interval_s=sampling_interval_s,
        )
        resource_by_batch[str(bs)] = res_obj

    results["resource"] = resource_by_batch
    return results


def _time_inference_single(model, X_test, n_warmup, n_samples, n_jobs=1, sampling_interval_s=0.05):
    """Single-packet timing with 2000 different rows, warm-up, and resource profiling."""
    n_total = len(X_test)
    rng = np.random.RandomState(42)
    indices = rng.choice(n_total, size=min(n_samples + n_warmup, n_total), replace=False)

    # Warm-up
    for i in indices[:n_warmup]:
        row = np.ascontiguousarray(X_test[i:i+1], dtype=np.float32)
        model.predict(row)

    # Timed runs
    times_us = []
    gc.collect()
    gc.disable()
    try:
        for i in indices[n_warmup:n_warmup + n_samples]:
            row = np.ascontiguousarray(X_test[i:i+1], dtype=np.float32)
            t0 = time.perf_counter()
            model.predict(row)
            elapsed = time.perf_counter() - t0
            times_us.append(elapsed * 1e6)
    finally:
        gc.enable()

    times_us = np.array(times_us)
    results = {
        "p50_us": float(np.percentile(times_us, 50)),
        "p95_us": float(np.percentile(times_us, 95)),
        "p99_us": float(np.percentile(times_us, 99)),
        "n_samples": len(times_us),
    }

    # Pre-extract rows for predictable resource workload
    test_rows = [
        np.ascontiguousarray(X_test[i:i+1], dtype=np.float32)
        for i in indices[n_warmup:n_warmup + n_samples]
    ]

    def single_packet_workload():
        for r in test_rows:
            model.predict(r)

    res_obj = profile_workload(
        workload_fn=single_packet_workload,
        workload_name="inference_single",
        config_dict={
            "n_jobs": n_jobs,
            "operations_per_measurement": len(test_rows),
            "n_samples": len(test_rows),
        },
        n_repeats=3,
        sampling_interval_s=sampling_interval_s,
    )
    results["resource"] = res_obj
    return results


def _time_training_scaling(model_class, model_params, X_train, y_train):
    """Training-time scaling: fit on subsamples of different sizes."""
    results = []
    n_total = len(y_train)

    for frac in config.TIMING_SCALING_FRACTIONS:
        n_rows = max(10, int(n_total * frac))
        n_repeats = config.TIMING_SCALING_REPEATS.get(frac, 1)

        if n_rows >= n_total:
            X_sub, y_sub = X_train, y_train
            n_rows = n_total
        else:
            rng = np.random.RandomState(42)
            idx = rng.choice(n_total, size=n_rows, replace=False)
            X_sub = np.ascontiguousarray(X_train[idx], dtype=np.float32)
            y_sub = y_train[idx]

        wall_times = []
        cpu_times = []
        for _ in range(n_repeats):
            clf = model_class(**model_params)
            gc.collect()
            t_cpu0 = time.process_time()
            t_wall0 = time.perf_counter()
            clf.fit(X_sub, y_sub)
            wall_times.append(time.perf_counter() - t_wall0)
            cpu_times.append(time.process_time() - t_cpu0)

        results.append({
            "fraction": frac,
            "n_rows": n_rows,
            "n_repeats": n_repeats,
            "wall_time_s": [round(t, 4) for t in wall_times],
            "cpu_time_s": [round(t, 4) for t in cpu_times],
            "wall_time_median_s": float(np.median(wall_times)),
            "cpu_time_median_s": float(np.median(cpu_times)),
        })

    return results


def _model_static_info(model, model_name):
    """Get static model descriptors (depth, leaves, size on disk)."""
    info = {}

    if hasattr(model, "get_depth"):
        info["depth"] = model.get_depth()
        info["leaves"] = model.get_n_leaves()
    elif hasattr(model, "estimators_"):
        depths = [t.get_depth() for t in model.estimators_]
        total_nodes = sum(t.tree_.node_count for t in model.estimators_)
        info["n_estimators"] = len(model.estimators_)
        info["mean_depth"] = float(np.mean(depths))
        info["total_nodes"] = total_nodes

    # Model size on disk — use temp file
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tf:
        tmp_path = tf.name
    try:
        joblib.dump(model, tmp_path)
        info["model_size_bytes"] = os.path.getsize(tmp_path)
        info["model_size_kb"] = round(info["model_size_bytes"] / 1024, 1)
    finally:
        os.unlink(tmp_path)

    return info


def _validate_timing_resource_records(timing):
    """
    Defensively validate that both models have expected resource records
    with numeric values or explicit unavailability reasons.
    """
    for name, model_data in timing.get("models", {}).items():
        # Check batch inference resource
        ib = model_data.get("inference_batch", {})
        if "resource" not in ib:
            logger.warning("Model '%s': missing 'resource' record in inference_batch", name)
        else:
            for bs_str, res in ib["resource"].items():
                if res.get("workload") != "inference_batch":
                    logger.warning("Model '%s' batch %s: unexpected workload name '%s'",
                                   name, bs_str, res.get("workload"))
                summary = res.get("summary", {})
                if not isinstance(summary.get("cpu_time_median_s"), (int, float)):
                    logger.warning("Model '%s' batch %s: non-numeric cpu_time_median_s", name, bs_str)
                if not isinstance(summary.get("rss_peak_median_bytes"), int):
                    logger.warning("Model '%s' batch %s: non-integer rss_peak_median_bytes", name, bs_str)

        # Check single-packet inference resource
        is_data = model_data.get("inference_single", {})
        if "resource" not in is_data:
            logger.warning("Model '%s': missing 'resource' record in inference_single", name)
        else:
            res = is_data["resource"]
            if res.get("workload") != "inference_single":
                logger.warning("Model '%s' single: unexpected workload name '%s'",
                               name, res.get("workload"))
            summary = res.get("summary", {})
            if not isinstance(summary.get("cpu_time_median_s"), (int, float)):
                logger.warning("Model '%s' single: non-numeric cpu_time_median_s", name)
            if not isinstance(summary.get("rss_peak_median_bytes"), int):
                logger.warning("Model '%s' single: non-integer rss_peak_median_bytes", name)


def run(run_id=None, clf=None, X_train=None, y_train=None, X_test=None, y_test=None):
    """
    Time-focused benchmark for DT and RF models.

    Can be called with run_id (file-based) or with direct arrays (legacy compat).

    Returns
    -------
    dict of timing results (also written to JSON).
    """
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier

    t_start = time.perf_counter()
    logger.info("=== STAGE 5: Time-Focused Resource Evaluation ===")

    # --- Load data if not provided ---
    if X_train is None or X_test is None:
        X_train, y_train, _ = io_utils.load_split("train")
        X_test, y_test, _ = io_utils.load_split("test")

    # Ensure float32 C-contiguous
    X_train = np.ascontiguousarray(X_train, dtype=np.float32)
    X_test = np.ascontiguousarray(X_test, dtype=np.float32)

    logger.info("Train: %d rows, Test: %d rows", len(y_train), len(y_test))

    # --- Load models ---
    models = {}
    for name in config.MODEL_NAMES:
        path = config.MODELS_DIR / f"{name}_model.joblib"
        if path.exists():
            models[name] = joblib.load(path)
            logger.info("Loaded %s from %s", name, path)
        else:
            logger.warning("%s model not found at %s — skipping", name, path)

    # --- Read training costs from manifest ---
    m = manifest.read()
    training_costs = {}
    cost_keys = [
        "tuning_time_s", "tuning_cpu_s", "fit_time_s", "fit_cpu_s",
        "total_training_cost_s", "n_rows_tuning", "n_rows_fit",
        "n_candidates", "n_fits_tuning", "n_jobs_tuning", "n_jobs_fit"
    ]
    for name in models:
        section = m.get(name, {})
        costs = {}
        for k in cost_keys:
            if k in section:
                costs[k] = section[k]
            else:
                logger.error("Missing manifest key for model '%s': '%s'", name, k)
                costs[k] = None
        training_costs[name] = costs

    timing = {
        "env": _get_env_block(),
        "notes": "Feature extraction cost is NOT included in these timings.",
        "resource_profiling": {
            "schema_version": 1,
            "library": "psutil",
            "scope": "model predict/testing only; model training, feature extraction, data loading, tuning, serialization, graphing and report generation are excluded",
            "process_scope": "current process plus recursive child processes",
            "cpu_definition": "aggregate process-tree user+system CPU seconds; average utilization = 100 * CPU seconds / wall seconds; may exceed 100 for parallel workloads",
            "memory_definition": "aggregate process-tree RSS bytes sampled during workload; shared pages can be counted by more than one process",
            "sampling_interval_s": 0.05,
            "status": "complete"
        },
        "models": {},
    }

    # Alternate model order across repeats for fairness
    model_names = list(models.keys())

    for name in model_names:
        model = models[name]
        logger.info("--- Timing %s ---", config.MODEL_DISPLAY_NAMES.get(name, name))

        model_timing = {
            "static": _model_static_info(model, name),
            "training_cost": training_costs.get(name, {}),
        }

        # Single-threaded timing model (never modify or re-save loaded model)
        timing_model = copy.deepcopy(model)
        if hasattr(timing_model, "n_jobs"):
            timing_model.n_jobs = 1

        # Primary Inference: batch sweep (single-threaded)
        logger.info("  Batch inference sweep (n_jobs=1)...")
        batch_res = _time_inference_batch(
            timing_model, X_test, config.TIMING_BATCH_SIZES, config.TIMING_BATCH_REPEATS, n_jobs=1,
        )
        batch_res["n_jobs"] = 1
        model_timing["inference_batch"] = batch_res

        # Primary Inference: single packet (single-threaded)
        logger.info("  Single-packet inference (n_jobs=1)...")
        single_res = _time_inference_single(
            timing_model, X_test, config.TIMING_SINGLE_WARMUP, config.TIMING_SINGLE_SAMPLES, n_jobs=1,
        )
        single_res["n_jobs"] = 1
        model_timing["inference_single"] = single_res

        # Training scaling
        logger.info("  Training-time scaling...")
        if name == "dt":
            alpha = getattr(model, "ccp_alpha", 0.0)
            model_timing["training_scaling"] = _time_training_scaling(
                DecisionTreeClassifier,
                {"criterion": "gini", "random_state": 42, "ccp_alpha": alpha},
                X_train, y_train,
            )
        elif name == "rf":
            rf_section = m.get("rf", {})
            best_params = rf_section.get("best_params", {})
            # n_jobs=1 run
            model_timing["training_scaling_1job"] = _time_training_scaling(
                RandomForestClassifier,
                {"criterion": "gini", "random_state": 42,
                 "n_estimators": config.RF_N_ESTIMATORS,
                 "min_samples_leaf": best_params.get("min_samples_leaf", config.RF_MIN_SAMPLES_LEAF),
                 "max_features": best_params.get("max_features", "sqrt"),
                 "n_jobs": 1},
                X_train, y_train,
            )
            # n_jobs=-1 run
            model_timing["training_scaling_all_cores"] = _time_training_scaling(
                RandomForestClassifier,
                {"criterion": "gini", "random_state": 42,
                 "n_estimators": config.RF_N_ESTIMATORS,
                 "min_samples_leaf": best_params.get("min_samples_leaf", config.RF_MIN_SAMPLES_LEAF),
                 "max_features": best_params.get("max_features", "sqrt"),
                 "n_jobs": -1},
                X_train, y_train,
            )

            # For RF only: additionally time the original model (all cores, as saved)
            orig_n_jobs = getattr(model, "n_jobs", -1)
            logger.info("  RF all-cores inference (n_jobs=%s)...", orig_n_jobs)
            batch_all = _time_inference_batch(
                model, X_test, config.TIMING_BATCH_SIZES, config.TIMING_BATCH_REPEATS, n_jobs=orig_n_jobs,
            )
            batch_all["n_jobs"] = orig_n_jobs
            model_timing["inference_batch_all_cores"] = batch_all

            single_all = _time_inference_single(
                model, X_test, config.TIMING_SINGLE_WARMUP, config.TIMING_SINGLE_SAMPLES, n_jobs=orig_n_jobs,
            )
            single_all["n_jobs"] = orig_n_jobs
            model_timing["inference_single_all_cores"] = single_all

        timing["models"][name] = model_timing

    # --- Defensive validation of resource records ---
    _validate_timing_resource_records(timing)

    # --- Write JSON ---
    if run_id is None:
        # Try to find latest run_id
        from datetime import datetime
        run_id = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = config.RESULTS_DIR / f"run_{run_id}_timing.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(timing, f, indent=2, default=str)
    logger.info("Timing JSON written: %s", json_path)

    elapsed = time.perf_counter() - t_start
    logger.info("Stage 5 completed in %.1f s", elapsed)

    return timing
