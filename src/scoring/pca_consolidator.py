
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
        self.eigvec_: Optional[np.ndarray] = None
        self.columns_: Optional[List[str]] = None

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
        eigvals, eigvecs = np.linalg.eig(cov)
        # Sort eigenvectors by descending eigenvalue
        idx = np.argsort(eigvals)[::-1]
        self.eigvec_ = eigvecs[:, idx[0]].real  # first principal component (real part)
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
        if self.mean_ is None or self.std_ is None or self.eigvec_ is None or self.columns_ is None:
            raise RuntimeError("PCAConsolidator has not been fitted")
        # Select and reorder columns to match training
        X_raw = df[self.columns_].copy()
        X_raw = X_raw.fillna(X_raw.mean())
        X = X_raw.to_numpy(dtype=float)
        std_safe = np.where(self.std_ == 0, 1.0, self.std_)
        X_std = (X - self.mean_) / std_safe
        # Project onto principal component
        scores = X_std.dot(self.eigvec_)
        # Normalise scores to [0, 1]
        min_val = np.min(scores)
        max_val = np.max(scores)
        if max_val - min_val > 0:
            normalised = (scores - min_val) / (max_val - min_val)
        else:
            normalised = np.zeros_like(scores)
        return normalised

    def to_dict(self) -> Dict[str, object]:
        """Serialise consolidator parameters to a dictionary.

        Returns
        -------
        dict
            Dictionary with keys ``mean``, ``std``, ``eigvec`` and
            ``columns``. Values are lists of floats.
        """
        if self.mean_ is None or self.std_ is None or self.eigvec_ is None or self.columns_ is None:
            raise RuntimeError("PCAConsolidator has not been fitted")
        return {
            'mean': self.mean_.tolist(),
            'std': self.std_.tolist(),
            'eigvec': self.eigvec_.tolist(),
            'columns': self.columns_,
        }

    @classmethod
    def from_dict(cls, params: Dict[str, object]) -> 'PCAConsolidator':
        """Reconstruct a PCAConsolidator from a serialised dictionary.

        Parameters
        ----------
        params: dict
            Dictionary produced by :meth:`to_dict`.

        Returns
        -------
        PCAConsolidator
            New instance with parameters loaded.
        """
        obj = cls()
        obj.mean_ = np.array(params['mean'], dtype=float)
        obj.std_ = np.array(params['std'], dtype=float)
        obj.eigvec_ = np.array(params['eigvec'], dtype=float)
        obj.columns_ = list(params['columns'])
        return obj


__all__ = ['PCAConsolidator']