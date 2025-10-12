import os, time, json, hashlib
from pathlib import Path
import joblib, optuna, pandas as pd, numpy as np, xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import subprocess

from common.constants import MAIN_CONFIG  # loads config/config.yaml  :contentReference[oaicite:2]{index=2}

CFG = MAIN_CONFIG.get("ML_Training", {})
DATA_FILE   = Path(CFG.get("data_file", "data/processed/training/v0/train_labels.parquet"))
LABEL_COL   = CFG.get("label_col", "label")
DROP_COLS   = list(CFG.get("drop_cols", ["window_id"]))
USE_SCALER  = bool(CFG.get("use_scaler", True))
MAX_TRIALS  = int(CFG.get("max_trials", 50))
TARGET_MAE  = float(CFG.get("min_target_mae", 0.05))
OUT_ROOT    = Path(CFG.get("out_root", "./data/artifacts/models/runs"))
VER = str(CFG.get("model_version", "0.0.0"))
UPDATE_LATEST = bool(CFG.get("update_latest_symlink", True))



if not DATA_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

# --- load dataset ---
df = pd.read_parquet(DATA_FILE) if DATA_FILE.suffix==".parquet" else pd.read_csv(DATA_FILE)
if LABEL_COL not in df.columns:
    raise ValueError(f"Label column '{LABEL_COL}' not found")
y = df[LABEL_COL].astype(float)
X = df.drop(columns=[LABEL_COL]+DROP_COLS, errors="ignore")
X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
bad_cols = [c for c in X.columns if X[c].isna().all() or X[c].nunique(dropna=True) <= 1]
if bad_cols:
    print("Dropping non-informative cols:", bad_cols)
    X = X.drop(columns=bad_cols)

# --- split + optional scaling ---
X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = None
if USE_SCALER:
    med = X_tr.median(numeric_only=True)
    X_tr = X_tr.fillna(med)
    X_va  = X_va.fillna(med)
    scaler = StandardScaler().fit(X_tr)
    X_tr = scaler.transform(X_tr)
    X_va = scaler.transform(X_va)
else:
    scaler = None    

dtr = xgb.DMatrix(X_tr, label=y_tr)
dva = xgb.DMatrix(X_va, label=y_va)

def objective(trial):
    params = {
        "verbosity": 0,
        "objective": "reg:squarederror",
        "booster": trial.suggest_categorical("booster", ["gbtree","dart"]),
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
    }
    if CFG.get("tree_method"): params["tree_method"] = CFG["tree_method"]
    params["seed"] = 42
    params["random_state"] = 42

    num_round = trial.suggest_int("num_boost_round", 200, 1200, step=50)
    esr = 50

    model = xgb.train(
        params, dtr, num_boost_round=num_round,
        evals=[(dva, "valid")],
        early_stopping_rounds=esr
    )
    preds = model.predict(dva, iteration_range=(0, model.best_iteration + 1))
    mae = mean_absolute_error(y_va, preds)
    val_rmse = float(np.sqrt(mean_squared_error(y_va, preds)))
    val_r2  = float(r2_score(y_va, preds))

    print(f"[VAL] MAE: {mae:.5f} | R²: {val_r2:.5f}")
    print(f"[VAL] RMSE: {val_rmse:.5f}")
    trial.set_user_attr("model", model)
    trial.set_user_attr("r2", val_r2)

    return mae

study = optuna.create_study(direction="minimize")
for _ in range(MAX_TRIALS):
    study.optimize(objective, n_trials=1)
    if study.best_value <= TARGET_MAE:
        print(f"[OK] Target MAE reached: {study.best_value:.5f} ≤ {TARGET_MAE:.5f}")
        break

best = study.best_trial
best_model = best.user_attrs["model"]
y_pred = best_model.predict(dva, iteration_range=(0, best_model.best_iteration + 1))
val_mae  = float(mean_absolute_error(y_va, y_pred))
val_r2   = float(r2_score(y_va, y_pred))
val_rmse = float(np.sqrt(mean_squared_error(y_va, y_pred)))
print(f"[VAL*] MAE: {val_mae:.5f} | R²: {val_r2:.5f} | RMSE: {val_rmse:.5f}")

# --- make run folder ---
ts = time.strftime("%Y%m%d_%H%M%S")
finger = hashlib.md5(str(best.params).encode()).hexdigest()[:10]
RUN_DIR = OUT_ROOT / f"v{VER}" / f"{ts}_{finger}"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
RUN_DIR.mkdir(parents=True, exist_ok=True)

# --- persist artifacts ---
bin_path    = RUN_DIR / "model.bin"
pkl_path    = RUN_DIR / "model.joblib"
scaler_path = RUN_DIR / "scaler.joblib"
meta_path   = RUN_DIR / "meta.json"

best_model.save_model(str(bin_path))     # native .bin
joblib.dump(best_model, pkl_path)        # joblib Booster
if scaler is not None:
    joblib.dump(scaler, scaler_path)

meta = {
    "best_params": best.params,
    "optuna_best_trial_mae": float(study.best_value),
    "optuna_best_trial_r2":  float(best.user_attrs.get("r2", float("nan"))),
    "final_val_mae": val_mae,
    "final_val_r2":  val_r2,
    "final_val_rmse": val_rmse,
    "data_file": str(DATA_FILE),
    "features": list(X.columns),
    "drop_cols": DROP_COLS,
    "label_col": LABEL_COL,
    "use_scaler": USE_SCALER,
    "run_dir": str(RUN_DIR),
    "timestamp": ts,
    "fingerprint": finger
}

def _file_md5(p: Path) -> str:
    import hashlib
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()

git_commit = subprocess.getoutput("git rev-parse --short HEAD") or "NA"
data_md5   = _file_md5(DATA_FILE)
feat_md5   = hashlib.md5("|".join(X.columns).encode()).hexdigest()

meta.update({
    "version": VER,
    "git_commit": git_commit,
    "data_md5": data_md5,
    "features_md5": feat_md5
})

with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)
assert Path(meta_path).exists(), f"meta.json not found at {meta_path}"
print("[OK] meta.json:", Path(meta_path).resolve())    

print("\nSaved artifacts:")
print("  ", bin_path)
print("  ", pkl_path)
if scaler is not None:
    print("  ", scaler_path)
print("  ", meta_path)

if UPDATE_LATEST:
    latest = OUT_ROOT / "latest"
    try:
        target_rel = RUN_DIR.relative_to(OUT_ROOT) 
    except Exception:
        target_rel = RUN_DIR.resolve()
    try:
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        latest.symlink_to(target_rel, target_is_directory=True)
        print("↪ updated symlink:", latest, "->", RUN_DIR)
    except Exception as e:
        print("symlink update skipped:", e)

