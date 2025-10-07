
from __future__ import annotations

from typing import Dict, Any, Iterable

import pandas as pd
import numpy as np

# Import column name definitions if available. Fallback to sensible defaults
try:
    from common.constants import COLUMN_NAMES
except ImportError:
    COLUMN_NAMES = {
        'timestamp': 'timestamp',
        'cell_entity': 'cell_entity',
        'ue_entity': 'ue_entity',
    }


def _basic_counts(cell: pd.DataFrame, ue: pd.DataFrame) -> Dict[str, Any]:
    """Compute simple row and entity counts.

    Parameters
    ----------
    cell: pandas.DataFrame
        Cell‐level records for the window.
    ue: pandas.DataFrame
        UE‐level records for the window.

    Returns
    -------
    dict
        Dictionary of count features.
    """
    feats: Dict[str, Any] = {}
    feats['cell_rows'] = int(len(cell)) if cell is not None else 0
    feats['ue_rows'] = int(len(ue)) if ue is not None else 0
    feats['total_rows'] = feats['cell_rows'] + feats['ue_rows']
    # Unique entity counts
    cell_ent = COLUMN_NAMES.get('cell_entity', 'cell_entity')
    ue_ent = COLUMN_NAMES.get('ue_entity', 'ue_entity')
    feats['unique_cells'] = int(cell[cell_ent].nunique()) if not cell.empty and cell_ent in cell.columns else 0
    feats['unique_ues'] = int(ue[ue_ent].nunique()) if not ue.empty and ue_ent in ue.columns else 0
    return feats


def _duplicate_counts(cell: pd.DataFrame, ue: pd.DataFrame) -> Dict[str, Any]:
    """Count duplicate entity/timestamp combinations.

    Identifies duplicates based on the combination of entity and
    timestamp columns. This can be used as a proxy for uniqueness
    checks in the absence of a dedicated uniqueness dimension.
    """
    feats: Dict[str, Any] = {}
    ts_col = COLUMN_NAMES.get('timestamp', 'timestamp')
    cell_ent = COLUMN_NAMES.get('cell_entity', 'cell_entity')
    ue_ent = COLUMN_NAMES.get('ue_entity', 'ue_entity')
    # Cell duplicates
    if cell is not None and not cell.empty and ts_col in cell.columns and cell_ent in cell.columns:
        dup_mask = cell.duplicated(subset=[cell_ent, ts_col], keep=False)
        feats['cell_duplicate_rows'] = int(dup_mask.sum())
    else:
        feats['cell_duplicate_rows'] = 0
    # UE duplicates
    if ue is not None and not ue.empty and ts_col in ue.columns and ue_ent in ue.columns:
        dup_mask = ue.duplicated(subset=[ue_ent, ts_col], keep=False)
        feats['ue_duplicate_rows'] = int(dup_mask.sum())
    else:
        feats['ue_duplicate_rows'] = 0
    return feats


def _numeric_stats(df: pd.DataFrame, prefix: str) -> Dict[str, Any]:
    """Compute summary statistics for numeric columns.

    Parameters
    ----------
    df: pandas.DataFrame
        The input DataFrame (cell or UE). Non‐numeric columns are
        ignored.
    prefix: str
        A prefix ('cell' or 'ue') prepended to feature names to
        indicate the origin of the metric.

    Returns
    -------
    dict
        A flat dictionary of statistics keyed by ``{prefix}_{col}_{stat}``.
        Supported statistics include mean, std, min, max, median,
        quantile at 0.25 and 0.75.
    """
    feats: Dict[str, Any] = {}
    if df is None or df.empty:
        return feats
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        series = pd.to_numeric(df[col], errors='coerce').dropna()
        if series.empty:
            continue
        key_base = f"{prefix}_{col}"
        feats[f"{key_base}_mean"] = float(series.mean())
        feats[f"{key_base}_std"] = float(series.std(ddof=0))
        feats[f"{key_base}_min"] = float(series.min())
        feats[f"{key_base}_max"] = float(series.max())
        feats[f"{key_base}_median"] = float(series.median())
        feats[f"{key_base}_q25"] = float(series.quantile(0.25))
        feats[f"{key_base}_q75"] = float(series.quantile(0.75))
    return feats


def make_feature_row(window_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a flattened feature representation for a single window.

    Parameters
    ----------
    window_data: dict
        Dictionary with keys ``cell_data``, ``ue_data`` and
        ``metadata``. ``cell_data`` and ``ue_data`` should be pandas
        DataFrames; ``metadata`` can be any mapping with auxiliary
        information. Missing data are handled gracefully.

    Returns
    -------
    dict
        A dictionary of engineered features. Metadata keys (except for
        nested dicts) are copied verbatim with the prefix ``meta_``.
        Numeric summaries and structural counts are included. The caller
        is responsible for saving these features to disk or combining
        them with labels for training.
    """
    cell = window_data.get('cell_data', pd.DataFrame())
    ue = window_data.get('ue_data', pd.DataFrame())
    metadata = window_data.get('metadata', {}) or {}
    feats: Dict[str, Any] = {}
    # Copy top‐level metadata values (primitives) with a prefix
    for key, val in metadata.items():
        if isinstance(val, (int, float, str)):
            feats[f"meta_{key}"] = val
    # Basic structural counts
    feats.update(_basic_counts(cell, ue))
    feats.update(_duplicate_counts(cell, ue))
    # Numeric summaries
    feats.update(_numeric_stats(cell, 'cell'))
    feats.update(_numeric_stats(ue, 'ue'))
    return feats


__all__ = ['make_feature_row']