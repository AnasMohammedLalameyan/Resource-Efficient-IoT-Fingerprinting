"""
stage5c_resource.py — Fresh-process CPU and memory benchmark (testing stage, model-only).

Scope (Tier 1): scikit-learn predict() on pre-extracted feature rows.
PCAP parsing, feature extraction, data loading, training and tuning are EXCLUDED.

Method: every measurement runs in its OWN fresh Python process (subprocess):
  * no training data, no sampler thread, garbage collection left enabled
  * n_jobs=1, BLAS/OpenMP thread env vars = 1, process pinned to one logical CPU
  * CPU : loop predict() for >= RESOURCE_MIN_SECONDS, CPU time = process_time delta
          (>= 5 s makes the ~15.6 ms Windows timer resolution negligible)
  * Memory: RSS/USS ladder (baseline -> sklearn import -> model load -> input ->
          warm-up -> after loop) plus the OS-reported peak RSS
  * Repeats: RESOURCE_REPEATS fresh processes per (model, packets-per-call);
          model order alternates between repeats

The probe (--probe) imports only numpy / joblib / psutil, so it can be copied to a
Raspberry Pi unchanged with a model .joblib and the sample .npy (scikit-learn
version must match the one that pickled the model):

    OMP_NUM_THREADS=1 python stage5c_resource.py --probe --model dt_model.joblib \
        --sample run_<id>_probe_sample.npy --batch-size 1000 --min-seconds 5 --pin-cpu 2

Project modules (config, io_utils) are imported lazily so they never pollute the
probe process's memory baseline.

Outputs: run_<id>_resource.json, run_<id>_probe_sample.npy; graphs via stage 5b
(run_<id>_resource_cpu.png / _resource_memory.png); stage 6 calls add_report_section().
"""

import argparse
import gc
import json
import logging
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import joblib
import psutil

logger = logging.getLogger(__name__)

_MARK = "PROBE_JSON:"
_MB = 1024 * 1024
MODEL_COLORS = {"dt": "#2196F3", "rf": "#4CAF50"}


# ===========================================================================
# PROBE (fresh child process; must not import project modules)
# ===========================================================================

def _mem(p):
    gc.collect()
    rss = p.memory_info().rss
    try:
        uss = p.memory_full_info().uss
    except Exception:
        uss = None
    return rss, uss


def _peak_rss_bytes(p):
    try:
        pk = getattr(p.memory_info(), "peak_wset", None)  # Windows
        if pk:
            return int(pk)
    except Exception:
        pass
    try:
        import resource  # Linux / macOS
        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(ru if sys.platform == "darwin" else ru * 1024)
    except Exception:
        return None


def _pin(cpu):
    if cpu is None or cpu < 0:
        return None
    try:
        if cpu >= psutil.cpu_count(logical=True):
            return None
        psutil.Process().cpu_affinity([int(cpu)])
        return int(cpu)
    except Exception:
        return None


def probe(model_path, sample_path, batch_size, min_seconds, pin_cpu):
    p = psutil.Process()
    pinned = _pin(pin_cpu)
    ladder = {}

    def snap(name):
        rss, uss = _mem(p)
        ladder[name] = {"rss_bytes": int(rss),
                        "uss_bytes": None if uss is None else int(uss)}

    snap("0_baseline")                      # interpreter + numpy + joblib + psutil
    import sklearn
    import sklearn.ensemble
    import sklearn.tree
    snap("1_sklearn_imported")

    model = joblib.load(model_path)
    if hasattr(model, "n_jobs"):
        model.n_jobs = 1
    snap("2_model_loaded")

    X = np.ascontiguousarray(np.load(sample_path), dtype=np.float32)
    if hasattr(model, "n_features_in_") and X.shape[1] != model.n_features_in_:
        raise ValueError("Sample has %d features but model expects %d"
                         % (X.shape[1], model.n_features_in_))
    snap("3_input_loaded")

    n = len(X)
    B = int(batch_size)
    if B <= 1:                               # per-packet calls on distinct rows
        batches = [X[i:i + 1] for i in range(min(n, 2000))]
    elif B >= n:
        batches = [X]
    else:
        batches = [X[j:j + B] for j in range(0, n - B + 1, B)]

    warm = batches[:200] if B <= 1 else (batches * 3)[:3]
    for b in warm:
        model.predict(b)
    snap("4_after_warmup")

    predict = model.predict
    nb = len(batches)
    calls = 0
    pkts = 0
    i = 0
    cpu0 = time.process_time()
    w0 = time.perf_counter()
    deadline = w0 + float(min_seconds)
    while True:
        b = batches[i]
        predict(b)
        pkts += b.shape[0]
        calls += 1
        i += 1
        if i == nb:
            i = 0
        if time.perf_counter() >= deadline:
            break
    cpu1 = time.process_time()
    w1 = time.perf_counter()

    snap("5_after_loop")
    cpu_s = cpu1 - cpu0
    wall_s = w1 - w0

    return {
        "model_file": os.path.basename(model_path),
        "model_file_bytes": os.path.getsize(model_path),
        "batch_size": B,
        "n_calls": calls,
        "n_packets": pkts,
        "min_seconds": float(min_seconds),
        "wall_s": wall_s,
        "cpu_s": cpu_s,
        "cpu_us_per_pkt": cpu_s / pkts * 1e6,
        "wall_us_per_pkt": wall_s / pkts * 1e6,
        "cpu_over_wall": (cpu_s / wall_s) if wall_s > 0 else None,
        "pinned_cpu": pinned,
        "n_jobs": getattr(model, "n_jobs", None),
        "ladder": ladder,
        "peak_rss_bytes": _peak_rss_bytes(p),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "platform": platform.platform(),
    }


# ===========================================================================
# ORCHESTRATOR (parent process)
# ===========================================================================

def _project_imports():
    root = str(Path(__file__).resolve().parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)
    import config
    import io_utils
    return config, io_utils


def _run_probe(model_path, sample_path, batch_size, config, env):
    cmd = [
        sys.executable, str(Path(__file__).resolve()), "--probe",
        "--model", str(model_path),
        "--sample", str(sample_path),
        "--batch-size", str(batch_size),
        "--min-seconds", str(config.RESOURCE_MIN_SECONDS),
        "--pin-cpu", str(-1 if config.RESOURCE_PIN_CPU is None else config.RESOURCE_PIN_CPU),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env,
                          timeout=float(config.RESOURCE_MIN_SECONDS) * 4 + 300)
    if proc.returncode != 0:
        raise RuntimeError("Probe failed (rc=%s) for %s batch=%s:\n%s"
                           % (proc.returncode, model_path, batch_size, proc.stderr[-2000:]))
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith(_MARK):
            return json.loads(line[len(_MARK):])
    raise RuntimeError("Probe produced no result line for %s batch=%s" % (model_path, batch_size))


def _stats(vals):
    a = np.asarray(vals, dtype=float)
    return {"median": float(np.median(a)),
            "q25": float(np.percentile(a, 25)),
            "q75": float(np.percentile(a, 75)),
            "min": float(a.min()),
            "max": float(a.max())}


def _derive_mem(run):
    L = run["ladder"]
    d = {}
    for f, tag in (("rss_bytes", "rss"), ("uss_bytes", "uss")):
        try:
            d["sklearn_import_%s_mib" % tag] = (L["1_sklearn_imported"][f] - L["0_baseline"][f]) / _MB
            d["model_load_%s_mib" % tag] = (L["2_model_loaded"][f] - L["1_sklearn_imported"][f]) / _MB
            d["input_%s_mib" % tag] = (L["3_input_loaded"][f] - L["2_model_loaded"][f]) / _MB
            d["inference_growth_%s_mib" % tag] = (L["5_after_loop"][f] - L["3_input_loaded"][f]) / _MB
        except TypeError:       # USS unavailable (None)
            pass
    return d


def _aggregate(run_list, rates):
    cpu = _stats([r["cpu_us_per_pkt"] for r in run_list])
    med = cpu["median"]
    derived = [_derive_mem(r) for r in run_list]
    keys = sorted(set().union(*[d.keys() for d in derived]))
    memory = {k: _stats([d[k] for d in derived if k in d]) for k in keys}
    ladder_rss = {}
    for ck in run_list[0]["ladder"].keys():
        ladder_rss[ck] = float(np.median([r["ladder"][ck]["rss_bytes"] for r in run_list])) / _MB
    peaks = [r["peak_rss_bytes"] / _MB for r in run_list if r["peak_rss_bytes"] is not None]
    return {
        "n_runs": len(run_list),
        "pinned_cpu": run_list[0]["pinned_cpu"],
        "cpu_us_per_pkt": cpu,
        "wall_us_per_pkt_median": float(np.median([r["wall_us_per_pkt"] for r in run_list])),
        "cpu_over_wall_median": float(np.median([r["cpu_over_wall"] for r in run_list
                                                 if r["cpu_over_wall"] is not None])),
        "n_calls_median": float(np.median([r["n_calls"] for r in run_list])),
        "max_pps_per_core": 1e6 / med,
        "core_load_pct_at_rate": {str(rt): rt * med * 1e-6 * 100.0 for rt in rates},
        "peak_rss_mib": _stats(peaks) if peaks else None,
        "memory_mib": memory,
        "ladder_rss_mib": ladder_rss,
        "model_file_bytes": run_list[0]["model_file_bytes"],
    }


def run(run_id=None):
    config, io_utils = _project_imports()
    import sklearn

    t_start = time.perf_counter()
    if run_id is None:
        run_id = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    logger.info("=== STAGE 5c: Fresh-process CPU / memory benchmark (model-only) ===")

    models = {}
    for name in config.MODEL_NAMES:
        path = config.MODELS_DIR / ("%s_model.joblib" % name)
        if path.exists():
            models[name] = path
        else:
            logger.warning("%s model not found at %s — skipping", name, path)
    if not models:
        raise FileNotFoundError("No trained models found in %s" % config.MODELS_DIR)

    # Fixed random test sample shared by every probe (reusable on a Pi)
    X_test, y_test, _ = io_utils.load_split("test")
    n = len(X_test)
    k = min(int(config.RESOURCE_SAMPLE_ROWS), n)
    rng = np.random.RandomState(config.RESOURCE_SAMPLE_SEED)
    idx = np.sort(rng.choice(n, size=k, replace=False))
    sample = np.ascontiguousarray(X_test[idx], dtype=np.float32)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sample_path = config.RESULTS_DIR / ("run_%s_probe_sample.npy" % run_id)
    np.save(sample_path, sample)
    logger.info("Probe sample: %d of %d test rows -> %s", k, n, sample_path)
    del X_test, y_test, sample

    env_block = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_logical": psutil.cpu_count(logical=True),
        "cpu_physical": psutil.cpu_count(logical=False),
        "python": sys.version,
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
    }
    try:
        bat = psutil.sensors_battery()
        env_block["on_ac_power"] = None if bat is None else bool(bat.power_plugged)
    except Exception:
        env_block["on_ac_power"] = None
    if env_block["on_ac_power"] is False:
        logger.warning("Laptop is on battery: plug in AC and use a high-performance "
                       "power plan before trusting these numbers.")

    child_env = dict(os.environ)
    child_env.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                      "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"})

    names = list(models.keys())
    sizes = [int(b) for b in config.RESOURCE_BATCH_SIZES]
    runs = {nm: {str(b): [] for b in sizes} for nm in names}

    for rep in range(int(config.RESOURCE_REPEATS)):
        order = names if rep % 2 == 0 else names[::-1]      # alternate model order
        for b in sizes:
            for nm in order:
                res = _run_probe(models[nm], sample_path, b, config, child_env)
                runs[nm][str(b)].append(res)
                pk = res["peak_rss_bytes"]
                logger.info("  rep %d/%d %-2s batch=%-7d cpu=%.4f us/pkt  wall=%.4f us/pkt  peakRSS=%s MiB",
                            rep + 1, config.RESOURCE_REPEATS, nm, b,
                            res["cpu_us_per_pkt"], res["wall_us_per_pkt"],
                            "n/a" if pk is None else "%.1f" % (pk / _MB))

    rates = list(config.RESOURCE_REFERENCE_RATES_PPS)
    out = {
        "schema_version": 2,
        "scope": ("Tier 1, model-only: scikit-learn predict() on pre-extracted features. "
                  "PCAP parsing, feature extraction, data loading, training and tuning are excluded. "
                  "Host measurement, not a Raspberry Pi measurement."),
        "method": {
            "isolation": "fresh subprocess per measurement; no training data; no sampler thread; GC enabled",
            "cpu": "process_time delta / packets over >= min_seconds of continuous predict(); n_jobs=1; pinned CPU",
            "memory": "RSS/USS ladder (baseline, sklearn import, model load, input, warm-up, after loop) + OS peak RSS",
            "core_load": "core_load_pct = rate_pps * cpu_us_per_pkt * 1e-4 (100% = one fully busy core)",
        },
        "env": env_block,
        "config": {
            "min_seconds": config.RESOURCE_MIN_SECONDS,
            "repeats": config.RESOURCE_REPEATS,
            "batch_sizes": sizes,
            "sample_rows": k,
            "sample_seed": config.RESOURCE_SAMPLE_SEED,
            "pin_cpu_requested": config.RESOURCE_PIN_CPU,
            "reference_rates_pps": rates,
        },
        "models": {},
    }
    for nm in names:
        out["models"][nm] = {
            "by_batch": {str(b): _aggregate(runs[nm][str(b)], rates) for b in sizes},
            "runs": runs[nm],
        }

    json_path = config.RESULTS_DIR / ("run_%s_resource.json" % run_id)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    logger.info("Resource JSON written: %s", json_path)
    logger.info("Stage 5c completed in %.1f s", time.perf_counter() - t_start)
    return out


# ===========================================================================
# GRAPHS (called from stage 5b)
# ===========================================================================

def graph_cpu(comp, run_id):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        config, _ = _project_imports()

        res = comp.get("resource")
        if not res:
            logger.warning("No resource data for graph_cpu (run stage 5c).")
            return None
        models = [m for m in config.MODEL_NAMES if m in res["models"]]
        sizes = [str(b) for b in res["config"]["batch_sizes"]]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        width = 0.8 / len(models)
        x = np.arange(len(sizes))
        for i, m in enumerate(models):
            by = res["models"][m]["by_batch"]
            med = [by[s]["cpu_us_per_pkt"]["median"] for s in sizes]
            lo = [by[s]["cpu_us_per_pkt"]["median"] - by[s]["cpu_us_per_pkt"]["q25"] for s in sizes]
            hi = [by[s]["cpu_us_per_pkt"]["q75"] - by[s]["cpu_us_per_pkt"]["median"] for s in sizes]
            bars = ax1.bar(x + (i - (len(models) - 1) / 2) * width, med, width,
                           yerr=[lo, hi], capsize=3,
                           color=MODEL_COLORS.get(m, "#999"),
                           label=config.MODEL_DISPLAY_NAMES.get(m, m))
            for bar, v in zip(bars, med):
                ax1.text(bar.get_x() + bar.get_width() / 2, v, "%.3g" % v,
                         ha="center", va="bottom", fontsize=8)
        ax1.set_yscale("log")
        ax1.set_xticks(x)
        ax1.set_xticklabels(["1 (per-packet calls)" if s == "1" else "{:,}".format(int(s)) for s in sizes])
        ax1.set_xlabel("Packets per predict() call")
        ax1.set_ylabel("CPU time per packet (us, log scale)")
        ax1.set_title("CPU time per packet\n(fresh process, n_jobs=1, pinned CPU; median and IQR)")
        ax1.grid(axis="y", linestyle="--", alpha=0.4)
        ax1.legend()

        rates = np.logspace(1, 6, 100)
        styles = [(sizes[0], "-")]
        if len(sizes) > 1:
            styles.append((sizes[1], "--"))
        for m in models:
            by = res["models"][m]["by_batch"]
            for s, ls in styles:
                cost_us = by[s]["cpu_us_per_pkt"]["median"]
                lab = "per-packet calls" if s == "1" else "{:,}-packet batches".format(int(s))
                ax2.plot(rates, rates * cost_us * 1e-4, ls, color=MODEL_COLORS.get(m, "#999"),
                         label="%s, %s" % (config.MODEL_DISPLAY_NAMES.get(m, m), lab))
        ax2.axhline(100, color="red", linestyle=":", label="one core fully busy (100%)")
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.set_ylim(1e-4, 1e4)
        ax2.set_xlabel("Packet rate (packets/s)")
        ax2.set_ylabel("Load on one core (%)")
        ax2.set_title("Model-only core load vs packet rate\n(excludes capture and feature extraction)")
        ax2.grid(True, which="both", alpha=0.3)
        ax2.legend(fontsize=8)

        fig.suptitle("CPU cost of inference (testing stage, host, model-only)", fontsize=13, fontweight="bold")
        plt.tight_layout()
        path = config.RESULTS_DIR / ("run_%s_resource_cpu.png" % run_id)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info("Graph saved: %s", path)
        return path.name
    except Exception as e:
        logger.error("Failed to generate resource_cpu graph: %s", e)
        return None


def graph_memory(comp, run_id):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        config, _ = _project_imports()

        res = comp.get("resource")
        if not res:
            logger.warning("No resource data for graph_memory (run stage 5c).")
            return None
        models = [m for m in config.MODEL_NAMES if m in res["models"]]
        sizes = [str(b) for b in res["config"]["batch_sizes"]]
        small, large = sizes[0], sizes[-1]

        def med(m, s, key):
            e = res["models"][m]["by_batch"][s]["memory_mib"].get(key)
            return e["median"] if e else 0.0

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        cats = ["Model load",
                "Inference growth\n(per-packet calls)",
                "Inference growth\n({:,}-packet batches)".format(int(large))]
        width = 0.8 / len(models)
        x = np.arange(len(cats))
        for i, m in enumerate(models):
            vals = [med(m, large, "model_load_rss_mib"),
                    med(m, small, "inference_growth_rss_mib"),
                    med(m, large, "inference_growth_rss_mib")]
            bars = ax1.bar(x + (i - (len(models) - 1) / 2) * width, vals, width,
                           color=MODEL_COLORS.get(m, "#999"),
                           label=config.MODEL_DISPLAY_NAMES.get(m, m))
            for bar, v in zip(bars, vals):
                ax1.text(bar.get_x() + bar.get_width() / 2, max(v, 0), "%.2f" % v,
                         ha="center", va="bottom", fontsize=8)
        ax1.set_xticks(x)
        ax1.set_xticklabels(cats)
        ax1.set_ylabel("Resident memory added (MiB)")
        ax1.set_title("Memory attributable to the model and to inference\n(fresh process; median of repeats)")
        ax1.grid(axis="y", linestyle="--", alpha=0.4)
        ax1.legend()

        comp_labels = ["Python + NumPy/joblib runtime", "scikit-learn import", "Model load",
                       "Input batch ({:,} packets)".format(int(large)), "Inference / transient to peak"]
        for i, m in enumerate(models):
            by = res["models"][m]["by_batch"][large]
            L = by["ladder_rss_mib"]
            peak = by["peak_rss_mib"]["median"] if by.get("peak_rss_mib") else L["5_after_loop"]
            parts = [L["0_baseline"],
                     L["1_sklearn_imported"] - L["0_baseline"],
                     L["2_model_loaded"] - L["1_sklearn_imported"],
                     L["3_input_loaded"] - L["2_model_loaded"],
                     peak - L["3_input_loaded"]]
            parts = [max(0.0, v) for v in parts]
            bottom = 0.0
            for j, v in enumerate(parts):
                ax2.bar(i, v, 0.5, bottom=bottom, color="C%d" % j,
                        label=comp_labels[j] if i == 0 else None)
                bottom += v
            ax2.text(i, bottom, "%.0f MiB" % bottom, ha="center", va="bottom", fontsize=9)
        ax2.set_xticks(range(len(models)))
        ax2.set_xticklabels([config.MODEL_DISPLAY_NAMES.get(m, m) for m in models])
        ax2.set_ylabel("Peak resident memory (MiB)")
        ax2.set_title("Peak RSS of a minimal inference process\n(runtime dominates; model is a small slice)")
        ax2.grid(axis="y", linestyle="--", alpha=0.4)
        ax2.legend(fontsize=8, loc="lower right")

        fig.suptitle("Memory of inference (testing stage, host, model-only)", fontsize=13, fontweight="bold")
        plt.tight_layout()
        path = config.RESULTS_DIR / ("run_%s_resource_memory.png" % run_id)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info("Graph saved: %s", path)
        return path.name
    except Exception as e:
        logger.error("Failed to generate resource_memory graph: %s", e)
        return None


# ===========================================================================
# PDF SECTION (called from stage 6; ASCII-only text for the core PDF font)
# ===========================================================================

def add_report_section(pdf, run_data, body_text, simple_table):
    config, _ = _project_imports()
    res = run_data.get("resource")

    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "7b. CPU and Memory Benchmark (Testing Stage, Model-Only)",
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    if not res:
        body_text("No resource benchmark data available (run stage 5c).")
        return

    cfg = res["config"]
    env = res["env"]
    sizes = [str(b) for b in cfg["batch_sizes"]]
    rates = cfg["reference_rates_pps"]
    models = [m for m in config.MODEL_NAMES if m in res["models"]]
    pinned = res["models"][models[0]]["by_batch"][sizes[0]].get("pinned_cpu") if models else None

    body_text(
        "Scope: model-only. scikit-learn predict() on pre-extracted feature rows; PCAP parsing, "
        "feature extraction, data loading, training and tuning are excluded. This is a host "
        "measurement, not a Raspberry Pi measurement.\n"
        "Method: every measurement runs in a fresh Python process (no training data, no sampler "
        "thread, garbage collection enabled) with n_jobs=1, pinned to logical CPU %s. "
        "CPU time per packet = process CPU-time delta / packets over at least %s s of continuous "
        "prediction (this makes the ~15.6 ms Windows timer resolution negligible); median and IQR "
        "over %s fresh processes. Memory = resident set size (RSS) after each stage of a minimal "
        "inference process (interpreter, scikit-learn import, model load, input batch, prediction) "
        "plus the OS-reported peak RSS. Input: fixed random sample of %s test packets.\n"
        "Host: %s, %s logical / %s physical cores, %s. On AC power: %s."
        % (pinned, cfg["min_seconds"], cfg["repeats"], "{:,}".format(cfg["sample_rows"]),
           env.get("processor"), env.get("cpu_logical"), env.get("cpu_physical"),
           env.get("platform"), env.get("on_ac_power"))
    )

    def lab(s):
        return "1 (per-packet)" if s == "1" else "{:,}".format(int(s))

    rows = []
    for m in models:
        for s in sizes:
            a = res["models"][m]["by_batch"][s]
            c = a["cpu_us_per_pkt"]
            rows.append([config.MODEL_DISPLAY_NAMES.get(m, m), lab(s),
                         "%.4g" % c["median"], "%.4g - %.4g" % (c["q25"], c["q75"]),
                         "{:,.0f}".format(a["max_pps_per_core"])])
    body_text("CPU time per packet:")
    simple_table(["Model", "Packets/call", "CPU us/pkt (median)", "IQR (q25 - q75)",
                  "Max pkt/s per core"], rows, col_widths=[32, 28, 40, 50, 40])

    rows = []
    for m in models:
        for s in sizes:
            a = res["models"][m]["by_batch"][s]
            rows.append([config.MODEL_DISPLAY_NAMES.get(m, m), lab(s)] +
                        ["%.4g" % a["core_load_pct_at_rate"][str(r)] for r in rates])
    body_text("Load on one core (%) at reference packet rates (100% = one fully busy core; "
              "model-only, excludes capture and feature extraction):")
    simple_table(["Model", "Packets/call"] + ["{:,} pkt/s".format(r) for r in rates],
                 rows, col_widths=[32, 28] + [32] * len(rates))

    rows = []
    for m in models:
        for s in sizes:
            a = res["models"][m]["by_batch"][s]
            mem = a["memory_mib"]
            pk = a.get("peak_rss_mib")
            rows.append([config.MODEL_DISPLAY_NAMES.get(m, m), lab(s),
                         "%.2f" % mem.get("model_load_rss_mib", {}).get("median", 0.0),
                         "%.2f" % mem.get("inference_growth_rss_mib", {}).get("median", 0.0),
                         "n/a" if not pk else "%.1f" % pk["median"]])
    body_text("Memory (MiB), median over fresh processes:")
    simple_table(["Model", "Packets/call", "Model load", "Inference growth", "Peak RSS (process)"],
                 rows, col_widths=[32, 28, 40, 45, 45])
    imp = None
    if models:
        e = res["models"][models[0]]["by_batch"][sizes[-1]]["memory_mib"].get("sklearn_import_rss_mib")
        imp = e["median"] if e else None
    body_text("Peak RSS includes the Python, NumPy and scikit-learn runtime%s; the memory "
              "attributable to the model itself is the 'Model load' column. 1 packet/call includes "
              "scikit-learn per-call overhead."
              % ("" if imp is None else " (importing scikit-learn alone adds about %.0f MiB)" % imp))


# ===========================================================================
# CLI: probe entry point (used by run() and for manual runs on another device)
# ===========================================================================

def _main():
    ap = argparse.ArgumentParser(description="Fresh-process CPU/memory probe (child of stage 5c).")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--model")
    ap.add_argument("--sample")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--min-seconds", type=float, default=5.0)
    ap.add_argument("--pin-cpu", type=int, default=-1)
    a = ap.parse_args()
    if not a.probe:
        ap.print_help()
        sys.exit(1)
    print(_MARK + json.dumps(probe(a.model, a.sample, a.batch_size, a.min_seconds, a.pin_cpu)),
          flush=True)


if __name__ == "__main__":
    _main()