import pandas as pd, numpy as np
df = pd.read_parquet("data/processed/training/v0/train_labels.parquet")
y = df["label"].astype(float)

summary = {
   "min": y.min(), "q25": y.quantile(.25), "median": y.median(),
   "mean": y.mean(), "q75": y.quantile(.75), "max": y.max(),
   "std": y.std(), "iqr": float(y.quantile(.75)-y.quantile(.25))
}
print(summary)

# optional: margins around policy thresholds
thresholds = [0.40, 0.70]  # change to what you’ll use
margins = np.diff(thresholds).tolist()
print({"thresholds": thresholds, "min_decision_margin": min(margins) if margins else None})