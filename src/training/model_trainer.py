
from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple, Optional

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import xgboost as xgb
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "xgboost is not installed. Install it or modify model_trainer to use a different estimator."
    ) from exc


def train_xgboost_model(
    dataset_path: str,
    model_output_path: str,
    scaler_output_path: str,
    *,
    label_column: str = "label",
    exclude_columns: Optional[Tuple[str, ...]] = ("window_id",),
    test_size: float = 0.2,
    random_state: int = 42,
    **xgb_params,
) -> Tuple[xgb.XGBRegressor, StandardScaler]:
    """Train an XGBoost regressor on a labelled feature dataset.

    Parameters
    ----------
    dataset_path : str
        Path to a parquet or CSV file containing a labelled dataset.  The
        table must include a ``label_column`` for the target variable.  Any
        columns listed in ``exclude_columns`` will be dropped before
        training.
    model_output_path : str
        Path where the trained model will be saved via ``joblib``.
    scaler_output_path : str
        Path where the fitted ``StandardScaler`` will be saved via ``joblib``.
    label_column : str, optional
        Name of the target column.  Defaults to ``"label"``.
    exclude_columns : tuple of str, optional
        Columns to drop before training.  Useful for removing identifiers
        such as ``window_id``.  Defaults to ``("window_id",)``.
    test_size : float, optional
        Fraction of the dataset to hold out for validation.  Defaults to
        0.2 (20%).
    random_state : int, optional
        Random seed used for reproducible splits and model training.
    **xgb_params
        Additional keyword arguments passed directly to the XGBoost
        regressor.  For example, ``n_estimators``, ``max_depth`` and
        ``learning_rate``.

    Returns
    -------
    model : xgboost.XGBRegressor
        The fitted XGBoost regressor.
    scaler : StandardScaler
        The fitted scaler used for feature normalisation.

    Notes
    -----
    The function will persist both the model and the scaler to disk.
    Directories for ``model_output_path`` and ``scaler_output_path`` are
    created if they do not already exist.
    """
    # Resolve file paths and ensure they exist
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    # Load dataset.  Allow parquet or CSV formats.
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix in {".csv", ".txt"}:
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported dataset format: {path.suffix}")

    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in dataset")
    # Separate features and label
    X = df.drop(columns=list(exclude_columns) + [label_column], errors="ignore")
    y = df[label_column].astype(float)
    # Fit scaler on training data only
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Optionally split off a validation set for quick metrics
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state
    )
    # Default XGBoost parameters, can be overridden by xgb_params
    default_params = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "reg:squarederror",
        "n_jobs": 4,
        "random_state": random_state,
    }
    default_params.update(xgb_params)
    model = xgb.XGBRegressor(**default_params)
    model.fit(X_train, y_train)
    # Evaluate quickly on the holdout set for sanity
    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    mse = mean_squared_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    print(f"Validation MAE: {mae:.5f}, MSE: {mse:.5f}, R2: {r2:.4f}")
    # Persist model and scaler
    model_path = Path(model_output_path)
    scaler_path = Path(scaler_output_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    return model, scaler


__all__ = ["train_xgboost_model"]