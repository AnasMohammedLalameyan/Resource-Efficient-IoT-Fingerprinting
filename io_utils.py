"""
io_utils.py — Table locating, loading, splitting, and format conversion.

Principle: Every stage reads only files and writes only files.
Centralized fast I/O utilities with Parquet preference and stratified sampling.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

import config


def find_table(dir_path: Union[str, Path], stem: str) -> Path:
    """
    Locate a data table preferring .parquet, then .csv.gz, then .csv.

    Parameters
    ----------
    dir_path : Path or str
        Directory to look in.
    stem : str
        Base filename without extension (e.g., 'raw_features' or 'selected_features').

    Returns
    -------
    Path to existing file.

    Raises
    ------
    FileNotFoundError if no matching file exists.
    """
    dir_path = Path(dir_path)
    for ext in (".parquet", ".csv.gz", ".csv"):
        candidate = dir_path / f"{stem}{ext}"
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"No table with stem '{stem}' (.parquet, .csv.gz, .csv) found in {dir_path}"
    )


def convert(input_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None) -> Path:
    """
    Convert a CSV or CSV.gz table to Parquet.

    Parameters
    ----------
    input_path : Path or str
        Path to input CSV or CSV.gz file.
    output_path : Path or str, optional
        Destination path. If None, replaces extension with .parquet.

    Returns
    -------
    Path to created Parquet file.
    """
    input_path = Path(input_path)
    if output_path is None:
        # If input is stem.csv.gz, strip both suffixes
        name = input_path.name
        if name.endswith(".csv.gz"):
            stem_name = name[:-7]
        elif name.endswith(".csv"):
            stem_name = name[:-4]
        else:
            stem_name = input_path.stem
        output_path = input_path.parent / f"{stem_name}.parquet"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Read table
    if input_path.suffix == ".parquet":
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)

    df.to_parquet(output_path, index=False, engine="pyarrow")
    return output_path


def load_split(
    split: str,
    cols: Optional[List[str]] = None,
    sample_frac: Optional[float] = None,
    random_state: int = 42,
    table_path: Optional[Union[str, Path]] = None,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load a train or test split from the selected features table.

    Parameters
    ----------
    split : {'train', 'test'}
        Which split to load.
    cols : list of str, optional
        Specific feature columns to load. If None, loads all features present
        (excluding 'mac_src', 'day', 'label').
    sample_frac : float, optional
        Fraction of rows to sample. Defaults to config.SAMPLE_FRAC.
        When < 1.0, applies seeded stratified sampling by (day, label).
    random_state : int, default 42
        Seed for reproducible sampling.
    table_path : Path or str, optional
        Path to selected features table. If None, auto-detected via find_table.

    Returns
    -------
    X : np.ndarray
        Float32 C-contiguous feature array.
    y : np.ndarray
        Int8 array (1 for IoT, 0 for non-IoT).
    meta : pd.DataFrame
        DataFrame with columns ['mac_src', 'day'].
    """
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got '{split}'")

    split_days = config.TRAIN_DAYS if split == "train" else config.TEST_DAYS

    if table_path is None:
        table_path = find_table(config.SELECTED_FEATURES_DIR, "selected_features")
    else:
        table_path = Path(table_path)

    # Read table
    is_parquet = table_path.suffix == ".parquet"
    if cols is not None:
        needed_cols = ["mac_src", "day", "label"] + [c for c in cols if c not in ("mac_src", "day", "label")]
        if is_parquet:
            df = pd.read_parquet(table_path, columns=needed_cols)
        else:
            df = pd.read_csv(table_path, usecols=needed_cols)
    else:
        if is_parquet:
            df = pd.read_parquet(table_path)
        else:
            df = pd.read_csv(table_path)

    # Filter to requested split days
    df = df[df["day"].isin(split_days)].copy()

    # Determine feature columns
    if cols is not None:
        feature_cols = [c for c in cols if c not in ("mac_src", "day", "label")]
    else:
        feature_cols = [c for c in df.columns if c not in ("mac_src", "day", "label")]

    # Apply sampling if sample_frac < 1.0
    if sample_frac is None:
        sample_frac = config.SAMPLE_FRAC

    if 0.0 < sample_frac < 1.0:
        # Seeded stratified sampling by (day, label)
        sampled_groups = []
        for (d, lbl), grp in df.groupby(["day", "label"], sort=False):
            n_grp = len(grp)
            n_sample = max(1, int(round(n_grp * sample_frac)))
            sampled_groups.append(grp.sample(n=min(n_sample, n_grp), random_state=random_state))
        df = pd.concat(sampled_groups, axis=0).sort_index()

    # Build X as float32 C-contiguous
    # Fill any remaining NaNs with config.NAN_FILL
    X_vals = df[feature_cols].fillna(config.NAN_FILL).values
    X = np.ascontiguousarray(X_vals, dtype=np.float32)

    # Build y as int8
    y = np.ascontiguousarray((df["label"] == "IoT").astype(np.int8).values, dtype=np.int8)

    # Build meta
    meta = df[["mac_src", "day"]].copy().reset_index(drop=True)

    return X, y, meta
