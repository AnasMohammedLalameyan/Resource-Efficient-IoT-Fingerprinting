"""
manifest.py — Centralized experiment manifest management.

Principle: Every stage reads only files and writes only files.
Manifest tracks feature selection, model hyperparameters, CV scores,
and environment versions atomically on disk at config.MODELS_DIR / "manifest.json".
"""

import os
import sys
import json
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import sklearn

import config


def get_manifest_path() -> Path:
    """Return path to manifest.json based on current config.MODELS_DIR."""
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return config.MODELS_DIR / "manifest.json"


def read() -> dict:
    """
    Read the manifest from disk.
    Returns empty dict if manifest does not exist or is invalid.
    """
    path = get_manifest_path()
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def update(section: str, data: dict) -> dict:
    """
    Update a section of the manifest atomically on disk.

    Stores data along with python, sklearn, numpy versions and timestamp.
    Uses atomic write (write to temp file then os.replace).

    Returns
    -------
    dict: The full updated manifest.
    """
    path = get_manifest_path()
    current = read()

    section_data = dict(data)
    section_data["_meta"] = {
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
    }

    current[section] = section_data

    # Atomic write to config.MODELS_DIR
    temp_dir = path.parent
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    with tempfile.NamedTemporaryFile("w", dir=temp_dir, delete=False, suffix=".tmp", encoding="utf-8") as tf:
        json.dump(current, tf, indent=2, default=str)
        temp_name = tf.name

    os.replace(temp_name, path)
    return current


def get_section(section: str, default=None):
    """Convenience helper to read a specific section."""
    m = read()
    return m.get(section, default)
