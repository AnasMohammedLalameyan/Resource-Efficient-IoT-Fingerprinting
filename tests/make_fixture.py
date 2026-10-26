"""
tests/make_fixture.py — Generate a synthetic raw_features dataset for fast unit tests.

Creates a small synthetic dataset mimicking the structure of raw_features:
  - columns: mac_src, day, label, + 32 FEATURE_COLS
  - days from config.TRAIN_DAYS and config.TEST_DAYS
  - MAC addresses from config.MAC_LABEL
  - NaNs in protocol-specific columns
  - Learnable signal for IoT vs non-IoT
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import config


def create_synthetic_raw_features(output_path, n_rows_per_day=500, random_state=42):
    """
    Generate synthetic raw_features.csv.gz.

    Parameters
    ----------
    output_path : Path or str
        Path to output CSV / CSV.GZ / Parquet file.
    n_rows_per_day : int
        Number of rows per day.
    random_state : int
        RNG seed.

    Returns
    -------
    Path to written file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(random_state)

    train_days = sorted(list(config.TRAIN_DAYS))[:4]  # Use 4 train days
    test_days  = sorted(list(config.TEST_DAYS))[:2]   # Use 2 test days
    days = train_days + test_days

    iot_macs = [m for m, l in config.MAC_LABEL.items() if l == "IoT"]
    non_iot_macs = [m for m, l in config.MAC_LABEL.items() if l == "non-IoT"]

    rows = []
    for day in days:
        for _ in range(n_rows_per_day):
            is_iot = rng.rand() > 0.4
            if is_iot:
                mac = rng.choice(iot_macs)
                label = "IoT"
            else:
                mac = rng.choice(non_iot_macs)
                label = "non-IoT"

            # Generate features
            feat = {}
            # Signal: IoT has smaller ip_len and lower entropy
            if is_iot:
                feat["ip_len"] = float(rng.randint(40, 300))
                feat["ip_ttl"] = float(rng.choice([64, 128]))
                feat["payload_entropy"] = float(round(rng.uniform(1.0, 4.0), 4))
                feat["src_port_class"] = float(rng.choice([1, 2]))
            else:
                feat["ip_len"] = float(rng.randint(200, 1500))
                feat["ip_ttl"] = float(rng.choice([54, 64, 255]))
                feat["payload_entropy"] = float(round(rng.uniform(4.0, 7.5), 4))
                feat["src_port_class"] = float(rng.choice([3, 4]))

            # Common features
            feat["ip_proto"] = float(rng.choice([6, 17, 1]))
            feat["ip_hdr_len"] = 20.0
            feat["ip_flag_df"] = float(rng.choice([0, 1]))
            feat["ip_flag_mf"] = 0.0
            feat["pck_size"] = feat["ip_len"] + 14.0
            feat["proto_type"] = feat["ip_proto"]
            feat["dst_port_class"] = float(rng.choice([1, 2, 3, 4]))

            # TCP features (NaN for UDP/ICMP)
            if feat["ip_proto"] == 6:
                feat["tcp_window_size"] = float(rng.choice([1024, 8192, 65535]))
                feat["tcp_flag_syn"] = float(rng.choice([0, 1], p=[0.9, 0.1]))
                feat["tcp_flag_ack"] = 1.0
                feat["tcp_flag_fin"] = 0.0
                feat["tcp_flag_psh"] = float(rng.choice([0, 1]))
                feat["tcp_flag_rst"] = 0.0
                feat["tcp_flag_urg"] = 0.0
                feat["tcp_dataofs"] = 32.0
                feat["tcp_options_len"] = 12.0
            else:
                for col in ["tcp_window_size", "tcp_flag_syn", "tcp_flag_ack", "tcp_flag_fin",
                            "tcp_flag_psh", "tcp_flag_rst", "tcp_flag_urg", "tcp_dataofs",
                            "tcp_options_len"]:
                    feat[col] = np.nan

            # UDP features
            if feat["ip_proto"] == 17:
                feat["udp_length"] = feat["ip_len"] - 20.0
            else:
                feat["udp_length"] = np.nan

            # ICMP features
            if feat["ip_proto"] == 1:
                feat["icmp_type"] = 8.0
                feat["icmp_code"] = 0.0
            else:
                feat["icmp_type"] = np.nan
                feat["icmp_code"] = np.nan

            # Protocol booleans
            feat["has_dns"] = float(rng.choice([0, 1], p=[0.8, 0.2]))
            feat["has_mdns"] = 0.0
            feat["has_ntp"] = float(rng.choice([0, 1], p=[0.95, 0.05]))
            feat["has_ssdp"] = 0.0
            feat["has_dhcp"] = 0.0
            feat["has_http"] = float(rng.choice([0, 1], p=[0.8, 0.2]))
            feat["has_https"] = float(rng.choice([0, 1], p=[0.7, 0.3]))
            feat["eapol_type"] = np.nan
            feat["eapol_version"] = np.nan

            row = {"mac_src": mac, "day": day, "label": label}
            for c in config.FEATURE_COLS:
                row[c] = feat.get(c, np.nan)
            rows.append(row)

    df = pd.DataFrame(rows)
    if output_path.suffix == ".parquet":
        df.to_parquet(output_path, index=False, engine="pyarrow")
    elif output_path.name.endswith(".csv.gz"):
        df.to_csv(output_path, index=False, compression="gzip")
    else:
        df.to_csv(output_path, index=False)

    return output_path


if __name__ == "__main__":
    out = PROJECT_ROOT / "tests" / "fixture_raw_features.csv.gz"
    create_synthetic_raw_features(out)
    print(f"Synthetic raw features fixture generated at: {out}")
