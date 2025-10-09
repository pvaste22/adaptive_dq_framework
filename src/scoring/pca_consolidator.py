
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

    def __init__(self, good_is_high: bool = True, round_ndecimals: int = 2):
        # config
        self.good_is_high = good_is_high
        self.round_ndecimals = round_ndecimals
        # learned params (after fit)
        self.col_means_ = None          # np.ndarray (len = n_features)
        self.col_stds_  = None          # np.ndarray (len = n_features)
        self.keep_cols_ = None          # list[str] kept (non-near-constant) columns
        self.pc1_vec_   = None          # np.ndarray (len = n_features_kept)
        self.pc1_min_   = None          # float (training min PC1 score)
        self.pc1_max_   = None          # float (training max PC1 score)

    # ------------------- FIT ON FULL TRAINING SET ------------------- #
    def fit(self, X: pd.DataFrame) -> "PCAConsolidator":
        """
        Fit on FULL training windows (NOT a small test batch).
        Stores: means/stds, kept columns, PC1 vector (orientation-locked), training min/max PC1.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("fit expects a pandas.DataFrame")
        # numeric only
        X = X.select_dtypes(include=[np.number]).copy()
        if X.empty:
            raise ValueError("fit: no numeric columns to use.")

        # drop near-constant columns (avoid z-score blow-ups)
        stds = X.std(axis=0, ddof=0)
        keep_mask = stds > 1e-9
        if not keep_mask.any():
            # fallback: keep all but use epsilon std
            keep_mask[:] = True
            stds = stds.replace(0, 1e-9)

        Xk = X.loc[:, keep_mask.index[keep_mask]]
        mu = Xk.mean(axis=0).values
        sd = Xk.std(axis=0, ddof=0).values
        sd[sd == 0.0] = 1e-9  # epsilon

        Z = (Xk - mu) / sd

        # PCA (cov -> eigen; take largest eigenvector)
        C = np.cov(Z.values, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(C)
        pc1 = eigvecs[:, -1]  # last = largest eigenvalue

        # orientation lock so that higher raw PC1 = WORSE quality
        # proxy of "worse" = (1 - mean(row)) since features are MPR in [0,1]
        row_mean = Xk.mean(axis=1).values
        worse_proxy = 1.0 - row_mean
        pc1_scores_tmp = Z.values @ pc1
        corr = np.corrcoef(pc1_scores_tmp, worse_proxy)[0, 1]
        if np.isnan(corr):
            corr = 1.0  # neutral
        sign = np.sign(corr) if corr != 0 else 1.0
        pc1 *= sign

        pc1_scores = Z.values @ pc1
        pc1_min = float(np.nanmin(pc1_scores))
        pc1_max = float(np.nanmax(pc1_scores))
        if not np.isfinite(pc1_min) or not np.isfinite(pc1_max) or (pc1_max - pc1_min) <= 0:
            # guard
            pc1_min, pc1_max = -1.0, 1.0

        # store
        self.keep_cols_ = list(Xk.columns)
        self.col_means_ = mu
        self.col_stds_  = sd
        self.pc1_vec_   = pc1
        self.pc1_min_   = pc1_min
        self.pc1_max_   = pc1_max
        return self

    # ------------------- TRANSFORM ANY (TEST) WINDOWS ------------------- #
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform new windows using TRAINING stats (no re-fit, no per-batch min-max).
        Returns label in [0,1]; by default **1 = good, 0 = bad** (good_is_high=True).
        """
        required = [self.keep_cols_, self.col_means_, self.col_stds_, self.pc1_vec_, self.pc1_min_, self.pc1_max_]
        if any(v is None for v in required):
            raise RuntimeError("transform called before fit (or model not loaded).")

        if not isinstance(X, pd.DataFrame):
            raise TypeError("transform expects a pandas.DataFrame")

        X = X.select_dtypes(include=[np.number]).copy()
        # align columns to training-kept columns
        # missing cols -> fill with training means, extra cols -> drop
        for c in self.keep_cols_:
            if c not in X.columns:
                X[c] = self.col_means_[self.keep_cols_.index(c)]
        Xk = X.loc[:, self.keep_cols_]

        # z-score with TRAINING stats
        Z = (Xk.values - self.col_means_) / self.col_stds_
        pc1_scores = Z @ self.pc1_vec_

        # absolute normalization with TRAINING range
        denom = (self.pc1_max_ - self.pc1_min_)
        if denom <= 0:
            denom = 1e-9
        badness = (pc1_scores - self.pc1_min_) / denom
        badness = np.clip(badness, 0.0, 1.0)

        # output orientation: 1 = good (requested) or 1 = bad
        labels = 1.0 - badness if self.good_is_high else badness

        if self.round_ndecimals is not None:
            labels = np.round(labels, self.round_ndecimals)
        return labels

    # ------------------- SAVE / LOAD MODEL ------------------- #
    def to_dict(self) -> dict:
        return {
            "good_is_high": self.good_is_high,
            "round_ndecimals": self.round_ndecimals,
            "keep_cols_": self.keep_cols_,
            "col_means_": None if self.col_means_ is None else self.col_means_.tolist(),
            "col_stds_":  None if self.col_stds_  is None else self.col_stds_.tolist(),
            "pc1_vec_":   None if self.pc1_vec_   is None else self.pc1_vec_.tolist(),
            "pc1_min_":   self.pc1_min_,
            "pc1_max_":   self.pc1_max_,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PCAConsolidator":
        obj = cls(
            good_is_high=d.get("good_is_high", True),
            round_ndecimals=d.get("round_ndecimals", 2),
        )
        obj.keep_cols_ = d.get("keep_cols_")
        obj.col_means_ = np.array(d["col_means_"]) if d.get("col_means_") is not None else None
        obj.col_stds_  = np.array(d["col_stds_"])  if d.get("col_stds_")  is not None else None
        obj.pc1_vec_   = np.array(d["pc1_vec_"])   if d.get("pc1_vec_")   is not None else None
        obj.pc1_min_   = d.get("pc1_min_")
        obj.pc1_max_   = d.get("pc1_max_")
        return obj


__all__ = ['PCAConsolidator']