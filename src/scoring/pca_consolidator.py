
from __future__ import annotations

import json
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


class PCAConsolidator:
    """Consolidates multiple quality metrics into a single scalar via PCA.

    This class provides ``fit`` and ``transform`` methods similar to
    scikit‐learn estimators. During fitting, it learns the mean,
    standard deviation and principal component on a training DataFrame.
    ``transform`` can then be applied to the same or new data with
    matching columns to produce a normalised quality label.
    """

    def __init__(self) -> None:
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.pc1_: Optional[np.ndarray] = None
        self.columns_: Optional[List[str]] = None
        self.pc1_min_: Optional[float] = None
        self.pc1_max_: Optional[float] = None

    def fit(self, df: pd.DataFrame) -> 'PCAConsolidator':
        """Learn normalisation parameters and principal component.

        Parameters
        ----------
        df: pandas.DataFrame
            DataFrame where each column corresponds to a quality metric
            (e.g. APR/MPR values) and each row corresponds to a
            separate data window.

        Returns
        -------
        self: PCAConsolidator
            Fitted instance.
        """
        if df is None or df.empty:
            raise ValueError("Input DataFrame for PCAConsolidator.fit must not be empty")
        # Keep only numeric columns
        numeric_df = df.select_dtypes(include=[np.number]).copy()
        if numeric_df.empty:
            raise ValueError("No numeric columns found for PCA consolidation")
        self.columns_ = numeric_df.columns.tolist()
        # Replace NaNs with column means for stability
        numeric_df = numeric_df.fillna(numeric_df.mean())
        X = numeric_df.to_numpy(dtype=float)
        # Compute mean and std per feature
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0, ddof=0)
        # Avoid division by zero: if std is zero (constant column), set to 1
        std_safe = np.where(self.std_ == 0, 1.0, self.std_)
        X_std = (X - self.mean_) / std_safe
        # Compute covariance matrix of standardised data (rowvar=False => columns are variables)
        cov = np.cov(X_std, rowvar=False)
        # Compute eigenvalues and eigenvectors
        eigvals, eigvecs = np.linalg.eigh(cov)
        # Sort eigenvectors by descending eigenvalue
        idx = np.argsort(eigvals)[::-1]  
        eigvecs = eigvecs[:, idx]  
        self.pc1_ = eigvecs[:, 0] # first principal component (real part)
        pc1_scores = X_std @ self.pc1_
        self.pc1_min_ = float(pc1_scores.min())
        self.pc1_max_ = float(pc1_scores.max())
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform new data into a consolidated quality score.

        Parameters
        ----------
        df: pandas.DataFrame
            New data with the same set of columns used during fit.

        Returns
        -------
        numpy.ndarray
            One‐dimensional array of normalised PC1 scores on [0, 1].
        """
        if any(v is None for v in [self.mean_, self.std_, self.pc1_, self.columns_, self.pc1_min_, self.pc1_max_]):
            raise RuntimeError("PCAConsolidator has not been fitted")
        # Select and reorder columns to match training
        X_raw = df.reindex(columns=self.columns_)
        impute_means = pd.Series(self.mean_, index=self.columns_)
        X_raw = X_raw.fillna(impute_means)
        X = X_raw.to_numpy(dtype=float)
        std_safe = np.where(self.std_ == 0, 1.0, self.std_)
        X_std = (X - self.mean_) / std_safe
        # Project onto principal component
        # Normalise scores to [0, 1]
        pc1_scores = X_std @ self.pc1_
        den = (self.pc1_max_ - self.pc1_min_)
        if den == 0:
            z = np.zeros_like(pc1_scores)     # all same, avoid div-by-zero
        else:
            z = (pc1_scores - self.pc1_min_) / den
        return z
        #return normalised

    def to_dict(self) -> dict:
        return {
            "columns": self.columns_,
            "mean": self.mean_.tolist(),
            "std": self.std_.tolist(),
            "pc1": self.pc1_.tolist(),
            "pc1_min": self.pc1_min_,
            "pc1_max": self.pc1_max_,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PCAConsolidator":
        obj = cls()
        obj.columns_  = list(d["columns"])
        obj.mean_     = np.array(d["mean"], dtype=float)
        obj.std_      = np.array(d["std"], dtype=float)
        obj.pc1_      = np.array(d["pc1"], dtype=float)
        obj.pc1_min_  = float(d["pc1_min"])
        obj.pc1_max_  = float(d["pc1_max"])
        return obj


__all__ = ['PCAConsolidator']