
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from scoring.traditional_scorer import (
    score_windows_in_directory,
    load_window_from_disk,
)
from scoring.pca_consolidator import PCAConsolidator
from data_processing.feature_extractor import make_feature_row


def _flatten_dimension_results(dim_results: Dict[str, Dict]) -> Dict[str, float]:
    """Flatten per‐dimension metrics into a flat dict.

    Only APR and MPR (score) values are included. Additional entries
    can be added here if required for training.
    """
    row: Dict[str, float] = {}
    for dim_name, res in dim_results.items():
        # Use MPR as the primary score; APR is also included
        # MPR is stored under key 'score' in dimension results
        mpr = float(res.get('score', 0.0))
        apr = float(res.get('apr', 0.0))
        row[f'{dim_name}_mpr'] = mpr
        row[f'{dim_name}_apr'] = apr
    return row


def generate_labeled_dataset(windows_dir: Path, output_dir: Path) -> pd.DataFrame:
    """Generate labelled dataset from scored windows.

    Parameters
    ----------
    windows_dir: Path
        Directory containing window subdirectories with data (as
        described in ``traditional_scorer``).
    output_dir: Path
        Directory where the output dataset and PCA parameters will
        be written. Will be created if it does not exist.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing engineered features and the PCA‐based
        quality label. The DataFrame is also written to a parquet
        file in ``output_dir``.
    """
    windows_dir = Path(windows_dir)
    output_dir = Path(output_dir)
    # Step 1: score all windows
    scored = score_windows_in_directory(windows_dir)
    if not scored:
        raise RuntimeError(f"No windows scored in {windows_dir}")
    window_ids: List[str] = list(scored.keys())
    # Step 2: build DataFrame of per‐dimension metrics
    dim_rows: List[Dict[str, float]] = []
    for wid in window_ids:
        dim_rows.append(_flatten_dimension_results(scored[wid]))
    dim_df = pd.DataFrame(dim_rows, index=window_ids)
    # Step 3: fit PCA consolidator and compute labels
    consolidator = PCAConsolidator()
    consolidator.fit(dim_df)
    labels = consolidator.transform(dim_df)
    # Persist PCA parameters for deployment
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'pca_consolidator.json', 'w', encoding='utf-8') as fh:
        json.dump(consolidator.to_dict(), fh, indent=2)
    # Step 4: compute feature vectors and attach labels
    feature_rows: List[Dict[str, float]] = []
    for idx, wid in enumerate(window_ids):
        window_path = windows_dir / wid
        window_data = load_window_from_disk(window_path)
        feat_row = make_feature_row(window_data)
        feat_row['label'] = float(labels[idx])
        feat_row['window_id'] = wid
        feature_rows.append(feat_row)
    dataset = pd.DataFrame(feature_rows)
    # Write to disk
    dataset.to_parquet(output_dir / 'labeled_data.parquet', index=False)
    return dataset


__all__ = ['generate_labeled_dataset']