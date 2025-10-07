
from __future__ import annotations

from typing import Dict, Any, Iterable

import pandas as pd
import numpy as np
from pathlib import Path
import json

# Import column name definitions if available. Fallback to sensible defaults
try:
    from common.constants import COLUMN_NAMES, REQUIRED_DIRS, UNRELIABLE_METRICS
    feats_dir = Path(REQUIRED_DIRS.get('artifacts', './artifacts'))
except ImportError:
    COLUMN_NAMES = {
        'timestamp': 'timestamp',
        'cell_entity': 'cell_entity',
        'ue_entity': 'ue_entity',
    }
    feats_dir = Path('./artifacts')
    UNRELIABLE_METRICS = {"TB.TotNbrDl", "TB.TotNbrUl"}

FEATURES_SCHEMA_PATH = feats_dir / "features" / "features_schema.json"

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

def save_feature_schema(columns):
    FEATURES_SCHEMA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FEATURES_SCHEMA_PATH, "w", encoding="utf-8") as f:
        json.dump({"columns": list(columns)}, f, indent=2)

def load_feature_schema():
    if FEATURES_SCHEMA_PATH.exists():
        with open(FEATURES_SCHEMA_PATH, "r", encoding="utf-8") as f:
            return json.load(f).get("columns", [])
    return []        

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
    num = df.select_dtypes(include=[np.number]).copy()
    if num.empty:
        return feats
    cols_keep = [c for c in num.columns if c not in UNRELIABLE_METRICS]
    num = num[cols_keep]
    if num.empty:
        return feats
    for col in num:
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

def _energy_consistency(cell: pd.DataFrame) -> Dict[str, Any]:
    feats: Dict[str, Any] = {}
    if cell is None or cell.empty:
        return feats
    ts_col = COLUMN_NAMES.get('timestamp', 'timestamp')
    if not {'PEE.AvgPower', 'PEE.Energy'}.issubset(set(cell.columns)) or ts_col not in cell.columns:
        return feats
    df = cell[[ts_col, 'PEE.AvgPower', 'PEE.Energy']].copy()
    df[ts_col] = pd.to_datetime(df[ts_col], errors='coerce')
    df = df.dropna(subset=[ts_col]).sort_values(ts_col)
    if df.empty:
        return feats
    dt = df[ts_col].diff().dt.total_seconds()
    e  = pd.to_numeric(df['PEE.Energy'], errors='coerce')
    p  = pd.to_numeric(df['PEE.AvgPower'], errors='coerce')
    de = e.diff()
    pred = p.shift(1) * dt  # power at previous step times delta t
    resid = (de - pred).replace([np.inf, -np.inf], np.nan).dropna()
    if resid.empty:
        return feats
    feats['cell_energy_resid_abs_mean'] = float(resid.abs().mean())
    feats['cell_energy_resid_abs_q75']  = float(resid.abs().quantile(0.75))
    return feats

def _ratio_features(cell: pd.DataFrame, ue: pd.DataFrame) -> Dict[str, Any]:
    feats: Dict[str, Any] = {}
    # Cell PRB utilization if available: Used / Tot (DL/UL)
    if cell is not None and not cell.empty:
        def safe_ratio(a, b):
            s = pd.to_numeric(cell.get(a), errors='coerce')
            t = pd.to_numeric(cell.get(b), errors='coerce')
            r = s / t
            return r.replace([np.inf, -np.inf], np.nan).dropna()

        for dl_ul in ('Dl', 'Ul'):
            used = f'RRU.PrbUsed{dl_ul}'
            tot  = f'RRU.PrbTot{dl_ul}_abs'
            avail   = f'RRU.PrbAvail{dl_ul}'
            if used in cell.columns and (tot in cell.columns or avail in cell.columns):
                denom = cell.get(tot) if tot in cell.columns else cell.get(avail)
                r = safe_ratio(used, denom)
                if not r.empty:
                    feats[f'cell_prb_util_{dl_ul.lower()}_mean'] = float(r.mean())
                    feats[f'cell_prb_util_{dl_ul.lower()}_q25']  = float(r.quantile(0.25))
                    feats[f'cell_prb_util_{dl_ul.lower()}_q75']  = float(r.quantile(0.75))

    # UE throughput per PRB (needs UE PRB used & UE THP)
    if ue is not None and not ue.empty:
        for dl_ul in ('Dl', 'Ul'):
            thp = f'DRB.UEThp{dl_ul}'
            prb = f'RRU.PrbUsed{dl_ul}'
            if thp in ue.columns and prb in ue.columns:
                t = pd.to_numeric(ue[thp], errors='coerce')
                p = pd.to_numeric(ue[prb], errors='coerce')
                r = t / p
                r = r.replace([np.inf, -np.inf], np.nan).dropna()
                if not r.empty:
                    feats[f'ue_thp_per_prb_{dl_ul.lower()}_mean'] = float(r.mean())
                    feats[f'ue_thp_per_prb_{dl_ul.lower()}_q25']  = float(r.quantile(0.25))
                    feats[f'ue_thp_per_prb_{dl_ul.lower()}_q75']  = float(r.quantile(0.75))
    # Energy per throughput (DL)
    if cell is not None and not cell.empty:
        if 'PEE.Energy_interval' in cell.columns and 'DRB.UEThpDl' in cell.columns:
            energy = pd.to_numeric(cell['PEE.Energy_interval'], errors='coerce')
            thp_dl = pd.to_numeric(cell['DRB.UEThpDl'], errors='coerce')
            r = energy / (thp_dl + 1e-9)  # avoid division by zero
            r = r.replace([np.inf, -np.inf], np.nan).dropna()
            if not r.empty:
                feats['cell_energy_per_thp_dl_mean'] = float(r.mean())
                feats['cell_energy_per_thp_dl_q75']  = float(r.quantile(0.75))
    return feats

def _null_density(df: pd.DataFrame, prefix: str) -> Dict[str, Any]:
    feats: Dict[str, Any] = {}
    if df is None or df.empty:
        return feats
    num = df.select_dtypes(include=[np.number])
    if num.empty:
        return feats
    null_frac = num.isna().mean().mean()  # overall numeric null fraction
    feats[f'{prefix}_null_frac'] = float(null_frac)
    return feats

def _time_cadence_features(df: pd.DataFrame, prefix: str, meas_sec: int | None) -> Dict[str, Any]:
    feats: Dict[str, Any] = {}
    if df is None or df.empty:
        return feats
    ts_col = COLUMN_NAMES.get('timestamp', 'timestamp')
    if ts_col not in df.columns:
        return feats

    ts = pd.to_datetime(df[ts_col], errors='coerce').dropna().sort_values()
    if ts.empty:
        return feats

    deltas = ts.diff().dropna().dt.total_seconds()
    if deltas.empty:
        return feats

    feats[f'{prefix}_dt_min'] = float(deltas.min())
    feats[f'{prefix}_dt_median'] = float(deltas.median())
    feats[f'{prefix}_dt_max'] = float(deltas.max())

    # expected interval
    expected = meas_sec if meas_sec else int(deltas.median())
    if expected <= 0:
        expected = int(deltas.median())

    # off-grid: not multiples of expected (within 10% slack)
    tol = max(1, int(0.1 * expected))
    off_grid = (~((deltas % expected).abs() <= tol) & ~(((expected - (deltas % expected)) % expected) <= tol)).sum()
    feats[f'{prefix}_offgrid_count'] = int(off_grid)

    # missing intervals (gaps) estimate
    missing = ((deltas // expected) - 1).clip(lower=0).sum()
    feats[f'{prefix}_missing_intervals'] = int(missing)

    # coverage ratio vs theoretical
    span = (ts.max() - ts.min()).total_seconds()
    theoretical = int(span // expected) + 1
    observed = len(ts)
    feats[f'{prefix}_coverage_ratio'] = float(observed / theoretical) if theoretical > 0 else 1.0
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
    ts = None
    if 'window_start_time' in metadata:
        ts = pd.to_datetime(metadata['window_start_time'], unit='s', errors='coerce')
    elif not cell.empty and COLUMN_NAMES.get('timestamp') in cell.columns:
        first_ts = cell[COLUMN_NAMES['timestamp']].iloc[0]
        ts = pd.to_datetime(first_ts, unit='s', errors='coerce')
    if ts is not None and not pd.isna(ts):
        feats['meta_hour_of_day'] = int(ts.hour)
        feats['meta_day_of_week'] = int(ts.dayofweek)  # Monday=0
    # Copy top‐level metadata values (primitives) with a prefix
    for key, val in metadata.items():
        if isinstance(val, (int, float, str)):
            feats[f"meta_{key}"] = val
    # Basic structural counts
    feats.update(_basic_counts(cell, ue))
    feats.update(_duplicate_counts(cell, ue))
    #timeliness features
    meas_sec = None
    try:
        from common.constants import MEAS_INTERVAL_SEC
        meas_sec = int(MEAS_INTERVAL_SEC)
    except Exception:
        pass

    feats.update(_time_cadence_features(cell, 'cell', meas_sec))
    feats.update(_time_cadence_features(ue, 'ue', meas_sec))

    #completeness features
    feats.update(_null_density(cell, 'cell'))
    feats.update(_null_density(ue, 'ue'))

    #Ratio features tied to Validity/Accuracy
    feats.update(_ratio_features(cell, ue))
    # consistency energy-power feature
    feats.update(_energy_consistency(cell))
    # Numeric summaries
    feats.update(_numeric_stats(cell, 'cell'))
    feats.update(_numeric_stats(ue, 'ue'))
    return feats

__all__ = ['make_feature_row']