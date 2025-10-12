
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import joblib
import numpy as np
import pandas as pd


def _load_schema(schema_path: Optional[str]) -> Optional[List[str]]:
    if schema_path is None:
        return None
    p = Path(schema_path)
    if not p.exists():
        raise FileNotFoundError(f"Feature schema file not found: {schema_path}")
    with p.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError("Feature schema must be a JSON list of column names")
    return data


class MLScorer:
    """Predict quality scores from engineered features using a trained ML model.

    Parameters
    ----------
    model_path : str
        Path to a pickled estimator supporting a ``predict`` method.
    scaler_path : str
        Path to a pickled scaler implementing ``transform`` for normalising
        feature inputs.
    feature_schema_path : str, optional
        Path to a JSON file containing the ordered list of feature
        names used during training.  If provided, incoming feature
        dictionaries are reindexed to match this order.  If not
        provided, the scorer assumes that the feature dict keys
        correspond exactly to the columns seen during training.
    """

    def __init__(self, *, model_path: str, scaler_path: str, feature_schema_path: Optional[str] = None) -> None:
        model_p = Path(model_path)
        scaler_p = Path(scaler_path)
        if not model_p.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not scaler_p.exists():
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
        self.model = joblib.load(model_p)
        self.scaler = joblib.load(scaler_p)
        self.schema: Optional[List[str]] = _load_schema(feature_schema_path)

    def _prepare_dataframe(self, feature_dict: Dict[str, Any]) -> pd.DataFrame:
        # Convert to single-row DataFrame
        df = pd.DataFrame([feature_dict])
        # If a schema is available, reorder and add missing columns
        if self.schema is not None:
            # Reindex ensures order and inserts NaN for missing columns
            df = df.reindex(columns=self.schema)
        return df

    def score_from_feature_dict(self, feature_dict: Dict[str, Any]) -> float:
        """Return a predicted quality score for a single feature dict.

        Parameters
        ----------
        feature_dict : dict
            Mapping from feature names to values (numeric).  Any features
            absent from the schema will be filled with NaN and then
            scaled using the stored scaler (which imputes by using
            training means if necessary).

        Returns
        -------
        float
            Predicted quality score.
        """
        df = self._prepare_dataframe(feature_dict)
        # Scale features; assume the scaler was fitted on the same column ordering
        X_scaled = self.scaler.transform(df)
        pred = self.model.predict(X_scaled)
        # model.predict returns an array; return scalar
        return float(pred[0])


__all__ = ["MLScorer"]