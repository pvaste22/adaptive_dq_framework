
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
from data_processing.feature_extractor import make_feature_row, load_feature_schema, save_feature_schema


def _next_versioned_path(base: Path) -> Path:
    if not base.exists():
        return base
    stem, suf = base.stem, base.suffix  # e.g., 'labeled_data', '.parquet'
    for i in range(1, 1000):
        cand = base.with_name(f"{stem}_v{i:03d}{suf}")
        if not cand.exists():
            return cand
    raise RuntimeError("Too many versions")


def _flatten_dimension_results(dim_results: dict) -> dict:
    """Flatten per‐dimension metrics into a flat dict.

    Only APR and MPR (score) values are included. Additional entries
    can be added here if required for training.
    """
    flat = {}
    for dim, res in dim_results.items():
        val = res.get("mpr", res.get("score"))
        try:
            flat[f"{dim}_mpr"] = float(val) if val is not None else float("nan")
        except Exception:
            flat[f"{dim}_mpr"] = float("nan")
    return flat



def generate_labeled_dataset_old(windows_dir: Path, output_dir: Path) -> pd.DataFrame:
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

    # step 2: Build per-dimension metric rows in the same order
    dim_rows: List[Dict[str, float]] = []
    paths: List[Path] = []
    for wid in window_ids:
        payload = scored[wid]
        paths.append(Path(payload["path"]))                 # use scorer-returned path
        dim_rows.append(_flatten_dimension_results(payload["result"]))

    dim_df = pd.DataFrame(dim_rows, index=window_ids)
    if dim_df.isna().all(axis=None):
        raise ValueError("All flattened MPR values are NaN/non-numeric. "
                     "Check that dimensions return numeric 'mpr'/'score'.")
    # Step 3: fit PCA consolidator and compute labels
    consolidator = PCAConsolidator(good_is_high=True, round_ndecimals=2)
    consolidator.fit(dim_df)
    labels = consolidator.transform(dim_df)
    # Persist PCA parameters for deployment
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'pca_consolidator.json', 'w', encoding='utf-8') as fh:
        json.dump(consolidator.to_dict(), fh, indent=2)
    # Step 4: compute feature vectors and attach labels
    feature_rows: List[Dict[str, float]] = []
    for idx, wid in enumerate(window_ids):
        window_data = load_window_from_disk(paths[idx])     # use returned path
        feat_row = make_feature_row(window_data)
        feat_row["label"] = float(labels[idx])
        feat_row["window_id"] = wid
        feature_rows.append(feat_row)
    dataset = pd.DataFrame(feature_rows)
    schema = load_feature_schema()
    if not schema:
        # first run → lock current order (or sorted(dataset.columns))
        schema = list(dataset.columns)
        save_feature_schema(schema)

    # enforce same order every run
    dataset = dataset.reindex(columns=schema)
    # Write to disk
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = _next_versioned_path(out_dir / "labeled_data.parquet")
    dataset.to_parquet(target, index=False)
    return dataset


def generate_labeled_dataset(
    windows_dir: str,
    output_parquet: str,
    pca_mode: str = "fit",                 # "fit" (train on full set) or "transform" (use saved model)
    pca_model_path: str = "artifacts/pca_consolidator.json",
) -> None:
    """
    Build labeled dataset from window scores using PCA consolidator.
    - pca_mode="fit": fit PCA on ALL windows here, save model, also output labels for these windows.
    - pca_mode="transform": load saved model and ONLY transform these windows (no fitting).
    """

    windows_dir = Path(windows_dir)
    if not windows_dir.exists():
        raise FileNotFoundError(f"{windows_dir} not found")

    # 1) Score windows (your existing scorer)
    scored = score_windows_in_directory(windows_dir)  # must return {window_id: {"path":..., "result": {...}}}
    if not scored:
        raise RuntimeError(f"No windows scored in {windows_dir}")

    # 2) Build PCA feature matrix (rows=windows, cols=dimension MPRs)
    window_ids = list(scored.keys())
    dim_rows, paths = [], []
    for wid in window_ids:
        payload = scored[wid]
        paths.append(Path(payload["path"]))
        dim_rows.append(_flatten_dimension_results(payload["result"]))

    dim_df = pd.DataFrame(dim_rows, index=window_ids)
    if dim_df.empty:
        raise RuntimeError("No numeric dimension scores to feed PCA.")

    # 3) PCA: train-once vs reuse
    pca_model_file = Path(pca_model_path)
    if pca_mode == "fit":
        pca_model_file.parent.mkdir(parents=True, exist_ok=True)
        cons = PCAConsolidator(good_is_high=True, round_ndecimals=2)  # 1=good, 0=bad + 2 dp
        cons.fit(dim_df)                                              # TRAIN on full set
        # (Optional) also produce labels for this same set
        labels = cons.transform(dim_df)
        with pca_model_file.open("w", encoding="utf-8") as fh:
            json.dump(cons.to_dict(), fh, indent=2)
        print(f"[PCA] Trained and saved model -> {pca_model_file}")
    elif pca_mode == "transform":
        if not pca_model_file.exists():
            raise FileNotFoundError(f"PCA model not found: {pca_model_file}")
        with pca_model_file.open("r", encoding="utf-8") as fh:
            cons = PCAConsolidator.from_dict(json.load(fh))
        labels = cons.transform(dim_df)                                # USE TRAINING STATS
        print(f"[PCA] Loaded model and transformed {len(window_ids)} windows.")
    else:
        raise ValueError("pca_mode must be 'fit' or 'transform'")

    # 4) Assemble final labeled rows (keep your feature extraction if you have it)
    # If you already build a feature row from raw window data, call that here.
    # Otherwise, we at least output window_id + label.
    rows = []
    for i, wid in enumerate(window_ids):
        # : window_data = load_window_from_disk(paths[i]); feat = make_feature_row(window_data)
        window_data = load_window_from_disk(paths[i])
        feat_row = make_feature_row(window_data)
        # Attach metadata
        feat_row["window_id"] = wid
        feat_row["label"] = float(labels[i])
        # else minimal row:
        rows.append(feat_row)

    dataset = pd.DataFrame(rows)

    # 5) Enforce consistent column order
    schema = load_feature_schema()
    if not schema:
        schema = [c for c in dataset.columns if c not in ["label", "window_id"]] + ["label", "window_id"]
        save_feature_schema(schema)
    dataset = dataset.reindex(columns=schema, fill_value=0.0)

    # 6) Save labeled dataset
    out_path = Path(output_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(out_path, index=False)
    print(f"[OK] Labeled dataset -> {out_path}")
    print(f"Shape: {dataset.shape}")  # Should be (n_windows, 200+ features)
    print(dataset.head())

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows_dir", required=True)
    ap.add_argument("--output_parquet", required=True)
    ap.add_argument("--pca_mode", choices=["fit", "transform"], default="fit")
    ap.add_argument("--pca_model_path", default="artifacts/pca_consolidator.json")
    args = ap.parse_args()

    generate_labeled_dataset(
        windows_dir=args.windows_dir,
        output_parquet=args.output_parquet,
        pca_mode=args.pca_mode,
        pca_model_path=args.pca_model_path,
    )

__all__ = ['generate_labeled_dataset']