"""
config.py — Centralized paths and constants for the IoT fingerprinting pipeline.

Every stage file, orchestrator, and dashboard imports from here.
No hardcoded absolute paths or MAC/day literals should exist anywhere else.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root and directory layout
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent

PCAP_DIR               = PROJECT_ROOT / "unsw_pcap_files"
RAW_FEATURES_DIR       = PROJECT_ROOT / "data" / "raw_features"
SELECTED_FEATURES_DIR  = PROJECT_ROOT / "data" / "selected_features"
MODELS_DIR             = PROJECT_ROOT / "models"
LOGS_DIR               = PROJECT_ROOT / "logs"
RESULTS_DIR            = PROJECT_ROOT / "results"
DEPLOY_TEST_DIR        = PCAP_DIR  # test-day archives live in the same PCAP folder

def refresh_paths(tag=None, sample_frac=None):
    """Dynamically refresh directory layout and model paths when TAG or SAMPLE_FRAC changes."""
    global TAG, SAMPLE_FRAC, SELECTED_FEATURES_DIR, MODELS_DIR, LOGS_DIR, RESULTS_DIR, DT_MODEL_PATH, RF_MODEL_PATH
    if tag is not None:
        os.environ["IOT_TAG"] = str(tag).strip()
    if sample_frac is not None:
        os.environ["IOT_SAMPLE_FRAC"] = str(sample_frac).strip()
    
    TAG = os.environ.get("IOT_TAG", "").strip()
    SAMPLE_FRAC = float(os.environ.get("IOT_SAMPLE_FRAC", "1.0"))

    SELECTED_FEATURES_DIR  = PROJECT_ROOT / "data" / "selected_features"
    MODELS_DIR             = PROJECT_ROOT / "models"
    LOGS_DIR               = PROJECT_ROOT / "logs"
    RESULTS_DIR            = PROJECT_ROOT / "results"

    if TAG:
        SELECTED_FEATURES_DIR = SELECTED_FEATURES_DIR / TAG
        MODELS_DIR            = MODELS_DIR / TAG
        LOGS_DIR              = LOGS_DIR / TAG
        RESULTS_DIR           = RESULTS_DIR / TAG

    for _d in (RAW_FEATURES_DIR, SELECTED_FEATURES_DIR, MODELS_DIR, LOGS_DIR, RESULTS_DIR):
        _d.mkdir(parents=True, exist_ok=True)

    DT_MODEL_PATH  = MODELS_DIR / "dt_model.joblib"
    RF_MODEL_PATH  = MODELS_DIR / "rf_model.joblib"

refresh_paths()

# ---------------------------------------------------------------------------
# Train / test day split
# ---------------------------------------------------------------------------
# 20 day-archives are available.  16 earliest → train, 4 latest → test.
# Sorted calendar order of all available archives:
#   16-09-23, 16-09-24, 16-09-25, 16-09-26, 16-09-27, 16-09-28,
#   16-09-29, 16-09-30, 16-10-01, 16-10-02, 16-10-03, 16-10-04,
#   16-10-05, 16-10-06, 16-10-07, 16-10-08, 16-10-09, 16-10-10,
#   16-10-11, 16-10-12
TRAIN_DAYS = {
    "16-09-23", "16-09-24", "16-09-25", "16-09-26", "16-09-27", "16-09-28",
    "16-09-29", "16-09-30", "16-10-01", "16-10-02", "16-10-03", "16-10-04",
    "16-10-05", "16-10-06", "16-10-07", "16-10-08",
}
TEST_DAYS = {
    "16-10-09", "16-10-10", "16-10-11", "16-10-12",
}

# Archive filenames — one per day, all under PCAP_DIR
ARCHIVES = [f"{day}.tar.gz" for day in sorted(TRAIN_DAYS | TEST_DAYS)]

# ---------------------------------------------------------------------------
# MAC address → label mapping (verbatim from original pipeline)
# ---------------------------------------------------------------------------
MAC_LABEL = {
    "d0:52:a8:00:67:5e": "IoT",
    "44:65:0d:56:cc:d3": "IoT",
    "70:ee:50:18:34:43": "IoT",
    "f4:f2:6d:93:51:f1": "IoT",
    "00:16:6c:ab:6b:88": "IoT",
    "30:8c:fb:2f:e4:b2": "IoT",
    "00:62:6e:51:27:2e": "IoT",
    "e8:ab:fa:19:de:4f": "IoT",
    "ec:1a:59:79:f4:89": "IoT",
    "50:c7:bf:00:56:39": "IoT",
    "74:c6:3b:29:d7:1d": "IoT",
    "ec:1a:59:83:28:11": "IoT",
    "18:b4:30:25:be:e4": "IoT",
    "70:ee:50:03:b8:ac": "IoT",
    "00:24:e4:1b:6f:96": "IoT",
    "74:6a:89:00:2e:25": "IoT",
    "00:24:e4:20:28:c6": "IoT",
    "d0:73:d5:01:83:08": "IoT",
    "18:b7:9e:02:20:44": "IoT",
    "e0:76:d0:33:bb:85": "IoT",
    "70:5a:0f:e4:9b:c0": "IoT",
    "30:8c:fb:6b:6a:45": "IoT",
    "08:21:ef:3b:fc:e3": "non-IoT",
    "40:f3:08:ff:1e:da": "non-IoT",
    "74:2f:68:81:69:42": "non-IoT",
    "ac:bc:32:d4:6f:2f": "non-IoT",
    "b4:ce:f6:a7:a3:c2": "non-IoT",
    "d0:a6:37:df:a1:e1": "non-IoT",
    "f4:5c:89:93:cc:85": "non-IoT",
}

# ---------------------------------------------------------------------------
# 32 candidate feature columns (verbatim from original pipeline)
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "ip_len", "ip_ttl", "ip_proto", "ip_hdr_len", "ip_flag_df", "ip_flag_mf",
    "tcp_window_size", "tcp_flag_syn", "tcp_flag_ack", "tcp_flag_fin",
    "tcp_flag_psh", "tcp_flag_rst", "tcp_flag_urg", "tcp_dataofs",
    "tcp_options_len", "udp_length", "icmp_type", "icmp_code",
    "proto_type", "src_port_class", "dst_port_class",
    "has_dns", "has_mdns", "has_ntp", "has_ssdp", "has_dhcp", "has_http", "has_https",
    "eapol_type", "eapol_version",
    "payload_entropy", "pck_size",
]

# ---------------------------------------------------------------------------
# Feature selection & cleaning constants
# ---------------------------------------------------------------------------
NAN_FILL = -1
RF_IMPORTANCE_THRESHOLD = 0.001
SELECTION_RF_MAX_ROWS = 500_000

# ---------------------------------------------------------------------------
# Cross-validation & Decision Tree constants
# ---------------------------------------------------------------------------
GROUP_KFOLD_SPLITS = 5
CV_N_ALPHAS        = 15
CV_ONE_SE          = True
BENCHMARK_REPEATS  = 5

# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------
CSV_COMPRESS = True

# ---------------------------------------------------------------------------
# Model paths
# ---------------------------------------------------------------------------
DT_MODEL_PATH  = MODELS_DIR / "dt_model.joblib"
RF_MODEL_PATH  = MODELS_DIR / "rf_model.joblib"

MODEL_NAMES = ("dt", "rf")
MODEL_DISPLAY_NAMES = {"dt": "Decision Tree", "rf": "Random Forest"}

# ---------------------------------------------------------------------------
# RF training & tuning constants
# ---------------------------------------------------------------------------
RF_N_ESTIMATORS     = 50
RF_MIN_SAMPLES_LEAF = 5
RF_TUNE_FRAC        = 0.2
TUNE_FRAC           = RF_TUNE_FRAC   # shared by DT and RF tuning (single protocol)
RF_EVAL_SEEDS       = [42, 123, 999]
RF_GRID             = {
    "min_samples_leaf": [1, 5, 20, 100],
    "max_features": ["sqrt", 0.5],
}

# ---------------------------------------------------------------------------
# Timing & Resource Evaluation constants (WP5)
# ---------------------------------------------------------------------------
TIMING_BATCH_SIZES       = [1, 10, 100, 1_000, 10_000, 100_000]
TIMING_BATCH_REPEATS     = 5
TIMING_SINGLE_WARMUP     = 200
TIMING_SINGLE_SAMPLES    = 2_000
TIMING_SCALING_FRACTIONS = [0.1, 0.3, 1.0]
TIMING_SCALING_REPEATS   = {0.1: 3, 0.3: 3, 1.0: 1}

# ---------------------------------------------------------------------------
# Model comparison constants
# ---------------------------------------------------------------------------
COMPARISON_BENCH_ROWS    = 100_000   # fixed test sample for fair resource comparison
COMPARISON_REPEATS       = 3         # repeats for inference-side comparison
COMPARISON_TRAIN_REPEATS = 2         # repeats for training-side comparison

# ---------------------------------------------------------------------------
# Fresh-process CPU / memory benchmark (stage 5c) — testing stage, model-only
# ---------------------------------------------------------------------------
RESOURCE_MIN_SECONDS         = 5.0                    # continuous predict() time per measurement
RESOURCE_REPEATS             = 7                      # fresh processes per (model, batch size)
RESOURCE_BATCH_SIZES         = [1, 1_000, 100_000]    # packets per predict() call (1 = per-packet calls)
RESOURCE_SAMPLE_ROWS         = 100_000                # fixed random test sample used by every probe
RESOURCE_SAMPLE_SEED         = 42
RESOURCE_PIN_CPU             = 2                      # logical CPU to pin to (None = no pinning)
RESOURCE_REFERENCE_RATES_PPS = [100, 1_000, 10_000, 100_000]
