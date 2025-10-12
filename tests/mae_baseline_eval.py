import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

df = pd.read_parquet("data/processed/training/v0/train_labels.parquet")
y = df["label"].astype(float)

# 80/20 split (same random_state you use)
_, y_val = train_test_split(y, test_size=0.2, random_state=42)

# mean/median baselines
mae_mean   = mean_absolute_error(y_val, np.full_like(y_val, y.mean()))
mae_median = mean_absolute_error(y_val, np.full_like(y_val, y.median()))
print({"baseline_mae_mean": float(mae_mean), "baseline_mae_median": float(mae_median)})