"""
stage1_extract.py — Archive reading + per-packet feature-dict extraction.

Reads raw .tar.gz archives, extracts packet features using the original
32-feature logic (port_class, payload_entropy, extract_features), and writes
a single combined CSV (with a 'day' column) to RAW_FEATURES_DIR.

Output: combined CSV file (optionally gzip-compressed) with columns:
  mac_src, day, label, <32 feature cols>
"""

import tarfile
import os
import io
import gc
import time
import logging

import numpy as np
import pandas as pd

from scapy.all import PcapReader, Ether, IP, TCP, UDP, ICMP, Raw
from scapy.layers.dns import DNS
from scapy.layers.dhcp import DHCP
from scapy.layers.eap import EAPOL

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helper functions (logic unchanged from original train_pipeline.py)
# ---------------------------------------------------------------------------

def port_class(port):
    if port is None or port == 0:
        return 0
    p = int(port)
    if p == 53:   return 1
    if p in (67, 68): return 2
    if p == 80:   return 3
    if p == 123:  return 4
    if p == 443:  return 5
    if p == 1900: return 6
    if p == 5353: return 7
    if 1 <= p <= 1023: return 8
    if 1024 <= p <= 49151: return 9
    return 10


def payload_entropy(pkt):
    if Raw in pkt:
        data = bytes(pkt[Raw].load)
    elif IP in pkt:
        data = bytes(pkt[IP].payload)
    else:
        return 0.0
    if not data:
        return 0.0
    counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
    probs = counts[counts > 0] / len(data)
    return float(-np.sum(probs * np.log2(probs)))


def extract_features(pkt, day_label):
    """Extract a feature dict from a single packet. Returns None if packet
    should be dropped (no Ethernet, no IP, or unknown MAC)."""
    if not pkt.haslayer(Ether):
        return None
    mac_src = pkt[Ether].src.lower()
    label = config.MAC_LABEL.get(mac_src)
    if label is None:
        return None
    if not pkt.haslayer(IP):
        return None
    ip = pkt[IP]
    row = {}
    row["mac_src"] = mac_src
    row["day"] = day_label
    row["label"] = label
    row["ip_len"] = int(ip.len) if ip.len is not None else np.nan
    row["ip_ttl"] = int(ip.ttl) if ip.ttl is not None else np.nan
    row["ip_proto"] = int(ip.proto) if ip.proto is not None else np.nan
    row["ip_hdr_len"] = int(ip.ihl) * 4 if ip.ihl is not None else np.nan
    flags = int(ip.flags) if ip.flags is not None else 0
    row["ip_flag_df"] = int(bool(flags & 0x2))
    row["ip_flag_mf"] = int(bool(flags & 0x1))
    if TCP in pkt:
        tcp = pkt[TCP]
        row["tcp_window_size"] = int(tcp.window) if tcp.window is not None else np.nan
        f = int(tcp.flags) if tcp.flags is not None else 0
        row["tcp_flag_syn"] = int(bool(f & 0x02))
        row["tcp_flag_ack"] = int(bool(f & 0x10))
        row["tcp_flag_fin"] = int(bool(f & 0x01))
        row["tcp_flag_psh"] = int(bool(f & 0x08))
        row["tcp_flag_rst"] = int(bool(f & 0x04))
        row["tcp_flag_urg"] = int(bool(f & 0x20))
        row["tcp_dataofs"] = int(tcp.dataofs) if tcp.dataofs is not None else np.nan
        opts = tcp.options if tcp.options else []
        row["tcp_options_len"] = len(opts)
        row["src_port_class"] = port_class(tcp.sport)
        row["dst_port_class"] = port_class(tcp.dport)
    else:
        row["tcp_window_size"] = np.nan
        for fl in ("syn", "ack", "fin", "psh", "rst", "urg"):
            row["tcp_flag_" + fl] = 0
        row["tcp_dataofs"] = np.nan
        row["tcp_options_len"] = 0
        row["src_port_class"] = 0
        row["dst_port_class"] = 0
    if UDP in pkt:
        udp = pkt[UDP]
        row["udp_length"] = int(udp.len) if udp.len is not None else np.nan
        if TCP not in pkt:
            row["src_port_class"] = port_class(udp.sport)
            row["dst_port_class"] = port_class(udp.dport)
    else:
        row["udp_length"] = np.nan
    if ICMP in pkt:
        icmp = pkt[ICMP]
        row["icmp_type"] = int(icmp.type) if icmp.type is not None else np.nan
        row["icmp_code"] = int(icmp.code) if icmp.code is not None else np.nan
    else:
        row["icmp_type"] = np.nan
        row["icmp_code"] = np.nan
    if TCP in pkt:   row["proto_type"] = 6
    elif UDP in pkt: row["proto_type"] = 17
    elif ICMP in pkt: row["proto_type"] = 1
    else:            row["proto_type"] = 0
    row["has_dns"] = int(pkt.haslayer(DNS))
    row["has_mdns"] = int(UDP in pkt and (pkt[UDP].dport == 5353 or pkt[UDP].sport == 5353))
    row["has_ntp"] = int(UDP in pkt and (pkt[UDP].dport == 123 or pkt[UDP].sport == 123))
    row["has_ssdp"] = int(UDP in pkt and (pkt[UDP].dport == 1900 or pkt[UDP].sport == 1900))
    row["has_dhcp"] = int(pkt.haslayer(DHCP) or (UDP in pkt and pkt[UDP].dport in (67, 68)))
    row["has_http"] = int(TCP in pkt and (pkt[TCP].dport == 80 or pkt[TCP].sport == 80))
    row["has_https"] = int(TCP in pkt and (pkt[TCP].dport == 443 or pkt[TCP].sport == 443))
    if pkt.haslayer(EAPOL):
        eapol = pkt[EAPOL]
        row["eapol_type"] = int(eapol.type) if hasattr(eapol, "type") and eapol.type is not None else np.nan
        row["eapol_version"] = int(eapol.version) if hasattr(eapol, "version") and eapol.version is not None else np.nan
    else:
        row["eapol_type"] = np.nan
        row["eapol_version"] = np.nan
    row["payload_entropy"] = payload_entropy(pkt)
    row["pck_size"] = len(pkt)
    return row


def extract_from_archive(archive_path):
    """Extract feature rows from all .pcap files inside a .tar.gz archive."""
    day = os.path.basename(archive_path).replace(".tar.gz", "")
    rows = []
    logger.info("Processing archive: %s", archive_path)
    with tarfile.open(archive_path, "r:gz") as tf:
        for member in tf.getmembers():
            if not member.name.endswith(".pcap"):
                continue
            f = tf.extractfile(member)
            if f is None:
                continue
            data = f.read()
            buf = io.BytesIO(data)
            try:
                reader = PcapReader(buf)
                for pkt in reader:
                    try:
                        r = extract_features(pkt, day)
                        if r is not None:
                            rows.append(r)
                    except Exception:
                        pass
            except Exception as e:
                logger.warning("  Warning reading %s: %s", member.name, e)
    logger.info("  -> %s rows from %s", f"{len(rows):,}", day)
    return rows


def _downcast_columns(df):
    """Downcast obviously-small integer columns to smaller dtypes to reduce
    CSV file size. Does NOT change float columns (e.g. payload_entropy) or
    any values — only storage dtype."""
    # Binary flag columns (0/1) -> uint8
    flag_cols = [
        "ip_flag_df", "ip_flag_mf",
        "tcp_flag_syn", "tcp_flag_ack", "tcp_flag_fin",
        "tcp_flag_psh", "tcp_flag_rst", "tcp_flag_urg",
        "has_dns", "has_mdns", "has_ntp", "has_ssdp", "has_dhcp",
        "has_http", "has_https",
    ]
    for c in flag_cols:
        if c in df.columns:
            df[c] = df[c].astype("uint8")

    # Small-range integer columns -> int16 or uint8
    small_int_cols = {
        "src_port_class": "uint8",
        "dst_port_class": "uint8",
        "proto_type": "uint8",
        "tcp_options_len": "uint8",
        "ip_proto": "uint8",
    }
    for c, dtype in small_int_cols.items():
        if c in df.columns:
            # Only downcast if no NaN (NaN requires float)
            if df[c].notna().all():
                df[c] = df[c].astype(dtype)

    return df


def run(archives=None, pcap_dir=None, output_dir=None, days_filter=None):
    """
    Run stage 1 extraction.

    Parameters
    ----------
    archives : list[str] or None
        Archive filenames to process. Defaults to config.ARCHIVES.
    pcap_dir : Path or None
        Directory containing archives. Defaults to config.PCAP_DIR.
    output_dir : Path or None
        Where to write the CSV. Defaults to config.RAW_FEATURES_DIR.
    days_filter : set or None
        If provided, only process archives whose day is in this set.

    Returns
    -------
    Path to the output CSV file.
    """
    t_start = time.perf_counter()
    archives = archives or config.ARCHIVES
    pcap_dir = pcap_dir or config.PCAP_DIR
    output_dir = output_dir or config.RAW_FEATURES_DIR

    logger.info("=== STAGE 1: Extraction ===")
    logger.info("PCAP directory: %s", pcap_dir)
    logger.info("Archives to process: %d", len(archives))

    all_rows = []
    total_dropped = 0  # approximate — we can't count dropped packets without additional tracking
    for archive_name in archives:
        day = archive_name.replace(".tar.gz", "")
        if days_filter and day not in days_filter:
            logger.info("  Skipping %s (not in days_filter)", day)
            continue
        path = os.path.join(str(pcap_dir), archive_name)
        if not os.path.exists(path):
            logger.warning("  Archive not found, skipping: %s", path)
            continue
        rows = extract_from_archive(path)
        all_rows.extend(rows)
        del rows
        gc.collect()

    df = pd.DataFrame(all_rows)
    del all_rows
    gc.collect()

    logger.info("Total rows after extraction: %s", f"{len(df):,}")

    # Downcast small integer columns for CSV size
    df = _downcast_columns(df)

    # Write CSV
    ext = ".csv.gz" if config.CSV_COMPRESS else ".csv"
    output_path = output_dir / f"raw_features{ext}"
    compression = "gzip" if config.CSV_COMPRESS else None
    df.to_csv(output_path, index=False, compression=compression)

    elapsed = time.perf_counter() - t_start
    logger.info("Output written to: %s", output_path)
    logger.info("Output shape: %s rows x %s columns", f"{len(df):,}", len(df.columns))
    logger.info("Stage 1 completed in %.1f s", elapsed)

    return output_path
