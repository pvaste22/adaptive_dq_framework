# compare_baselines_vs_model.py
import json, numpy as np, pandas as pd, xgboost as xgb, joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

import os, glob, time
from pathlib import Path

"""def resolve_run_dir():
    env = os.getenv("MODEL_RUN_DIR")
    if env:
        p = Path(env)
        if (p / "meta.json").exists():
            return p
    candidates = [
        Path("data/artifacts/models/latest"),
        Path("data/artifacts/models/runs/latest"),
    ]
    for c in candidates:
        if (c / "meta.json").exists():
            return c
    metas = glob.glob("data/artifacts/models/**/meta.json", recursive=True)
    if not metas:
        raise FileNotFoundError("No meta.json found under data/artifacts/models")
    metas.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return Path(metas[0]).parent
"""
run_dir = Path("data/artifacts/models/latest").resolve()
print("latest symlink: ",run_dir)
meta = json.loads((run_dir / "meta.json").read_text())


df = pd.read_parquet(meta["data_file"])
y = df[meta["label_col"]].astype(float)
X = df.drop(columns=[meta["label_col"]]+meta["drop_cols"], errors="ignore")
X = X[meta["features"]]

X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
if meta.get("use_scaler", False):
    scaler = joblib.load(run_dir / "scaler.joblib") if meta.get("use_scaler", False) else None
    X_val = scaler.transform(X_val)

bst = xgb.Booster(); bst.load_model(str(run_dir / "model.bin"))
pred = bst.predict(xgb.DMatrix(X_val))

res = {
  "baseline_mae_mean": float(mean_absolute_error(y_val, np.full_like(y_val, y.mean()))),
  "baseline_mae_median": float(mean_absolute_error(y_val, np.full_like(y_val, y.median()))),
  "model_mae": float(mean_absolute_error(y_val, pred))
}
print(res)
