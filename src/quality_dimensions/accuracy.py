"""
Accuracy measures the degree to which data correctly represents real-world objects, events, or agreed-upon sources of truth. 
DAMA defines it as "the degree to which data correctly describes the 'real world' object or event being described," 
making it perhaps the most fundamental quality dimension since inaccurate data, regardless of other quality attributes, 
renders information unfit for use.
"""

"""
Accuracy dimension calculator.

Detects outliers and impossible values using z-score and domain rules.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from scipy import stats
from .base_dimension import BaseDimension
from ...common.constants import BAND_LIMITS

class AccuracyDimension(BaseDimension):
    """Calculate accuracy score for data quality assessment."""
    
    def __init__(self):
        """Initialize accuracy dimension calculator."""
        super().__init__("Accuracy")
        self.z_score_threshold = self.thresholds.get('z_score_threshold', 3.0)
        self.band_limits = BAND_LIMITS
        
    def calculate(self, window_data: pd.DataFrame, 
                 baselines: Optional[Dict[str, Any]] = None) -> float:
        """
        Calculate accuracy score based on outliers and impossible values.
        
        Accuracy = 1 - (outlier_ratio + impossible_ratio) / 2
        
        Args:
            window_data: DataFrame containing window data
            baselines: Statistical baselines for outlier detection
            
        Returns:
            Accuracy score [0, 1]
        """
        if not self.validate_window(window_data):
            return 0.0
            
        outlier_scores = []
        impossible_scores = []
        
        # Check each metric for outliers and impossible values
        for col in window_data.columns:
            if col in ['timestamp', 'Viavi.Cell.Name', 'Viavi.UE.Name', 'band']:
                continue
                
            col_data = window_data[col].dropna()
            if len(col_data) == 0:
                continue
                
            # Check for outliers using z-score
            outlier_ratio = self._detect_outliers_zscore(col_data, col, baselines)
            outlier_scores.append(1.0 - outlier_ratio)
            
            # Check for impossible values
            impossible_ratio = self._detect_impossible_values(col_data, col, window_data)
            impossible_scores.append(1.0 - impossible_ratio)
        
        # Combine scores
        if outlier_scores and impossible_scores:
            outlier_score = np.mean(outlier_scores)
            impossible_score = np.mean(impossible_scores)
            accuracy_score = (outlier_score + impossible_score) / 2
        else:
            accuracy_score = 1.0
            
        self.logger.debug(
            f"Accuracy: {accuracy_score:.3f} "
            f"(outlier_score: {np.mean(outlier_scores) if outlier_scores else 1.0:.3f}, "
            f"impossible_score: {np.mean(impossible_scores) if impossible_scores else 1.0:.3f})"
        )
        
        return self.normalize_score(accuracy_score)
    
    def _detect_outliers_zscore(self, data: pd.Series, metric: str, 
                               baselines: Optional[Dict] = None) -> float:
        """
        Detect outliers using z-score method.
        
        Args:
            data: Series of metric values
            metric: Metric name
            baselines: Statistical baselines with mean/std
            
        Returns:
            Ratio of outliers
        """
        if len(data) < 3:  # Need at least 3 points for z-score
            return 0.0
            
        # Use baselines if available, otherwise calculate from current window
        if baselines and 'statistical_baselines' in baselines:
            stats_baseline = baselines['statistical_baselines'].get(metric, {})
            mean_val = stats_baseline.get('mean')
            std_val = stats_baseline.get('std')
            
            if mean_val is not None and std_val is not None and std_val > 0:
                z_scores = np.abs((data - mean_val) / std_val)
            else:
                # Fallback to modified z-score using MAD
                median = data.median()
                mad = np.median(np.abs(data - median))
                if mad == 0:
                    return 0.0
                z_scores = 0.6745 * (data - median) / mad
        else:
            # Calculate z-scores from current window
            z_scores = np.abs(stats.zscore(data))
        
        outlier_count = np.sum(z_scores > self.z_score_threshold)
        return outlier_count / len(data)
    
    def _detect_impossible_values(self, data: pd.Series, metric: str, 
                                 window_data: pd.DataFrame) -> float:
        """
        Detect physically impossible values based on domain rules.
        
        Args:
            data: Series of metric values
            metric: Metric name
            window_data: Full window data for context
            
        Returns:
            Ratio of impossible values
        """
        impossible_count = 0
        
        # Check for negative values where not allowed
        if metric in ['DRB.UEThpDl', 'DRB.UEThpUl', 'RRU.PrbUsedDl', 
                     'RRU.PrbUsedUl', 'PEE.AvgPower', 'RRC.ConnMean']:
            impossible_count += np.sum(data < 0)
        
        # Check PRB limits based on band
        if 'Prb' in metric and 'band' in window_data.columns:
            for band in window_data['band'].unique():
                if band in self.band_limits:
                    band_data = data[window_data['band'] == band]
                    max_prb = self.band_limits[band]['prb_count']
                    impossible_count += np.sum(band_data > max_prb)
        
        # Check CQI range [0, 15]
        if 'Cqi' in metric:
            impossible_count += np.sum((data < 0) | (data > 15))
        
        # Check RRC connections
        if metric == 'RRC.ConnMean' and 'RRC.ConnMax' in window_data.columns:
            mean_vals = data
            max_vals = window_data.loc[data.index, 'RRC.ConnMax']
            impossible_count += np.sum(mean_vals > max_vals)
        
        return impossible_count / len(data) if len(data) > 0 else 0.0