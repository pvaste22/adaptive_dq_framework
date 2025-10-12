
import json, numpy as np, pandas as pd
from sklearn.metrics import mean_absolute_error
from pathlib import Path

import os, json, glob, time


def resolve_run_dir():
    # 1) explicit env override (optional)
    env = os.getenv("MODEL_RUN_DIR")
    if env:
        p = Path(env)
        if (p / "meta.json").exists():
            return p

    # 2) common symlink locations
    candidates = [
        Path("data/artifacts/models/latest"),
        Path("data/artifacts/models/runs/latest"),
    ]
    for c in candidates:
        if (c / "meta.json").exists():
            return c

    # 3) scan all meta.json and pick newest by mtime
    metas = glob.glob("data/artifacts/models/**/meta.json", recursive=True)
    if not metas:
        raise FileNotFoundError("No meta.json found under data/artifacts/models")
    metas.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return Path(metas[0]).parent

run_dir = resolve_run_dir()
meta = json.loads((run_dir / "meta.json").read_text())
print(f"[bootstrap] using run dir: {run_dir}")
# Load your val split predictions by re-running predict on the saved best model
import xgboost as xgb, joblib
from sklearn.model_selection import train_test_split

df = pd.read_parquet(meta["data_file"])
y = df[meta["label_col"]].astype(float)
X = df.drop(columns=[meta["label_col"]]+meta["drop_cols"], errors="ignore")

# enforce feature order
X = X[meta["features"]]

# scaler (if used)
if meta.get("use_scaler", False):
    scaler = joblib.load(run_dir/"scaler.joblib")
    # use same split seed
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val = scaler.transform(X_val)
else:
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

bst = xgb.Booster(); bst.load_model(str(run_dir/"model.bin"))
pred = bst.predict(xgb.DMatrix(X_val))

mae = mean_absolute_error(y_val, pred)

# bootstrap 1000x
rng = np.random.default_rng(42)
idx = np.arange(len(y_val))
boot = []
for _ in range(1000):
    b = rng.choice(idx, size=len(idx), replace=True)
    boot.append(mean_absolute_error(y_val.iloc[b].values, pred[b]))
low, high = np.percentile(boot, [2.5, 97.5])
print({"val_mae": float(mae), "mae_ci_95": [float(low), float(high)]})