"""
Consistency measures the absence of contradictions when comparing data representations across different systems, time periods,
 or contexts. According to DAMA, it represents "the absence of difference, when comparing two or more representations of a thing 
 against a definition." 
"""

"""
Consistency dimension calculator.

Validates relationships between metrics and checks logical constraints.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from .base_dimension import BaseDimension
from ...common.constants import EXPECTED_CORRELATIONS

class ConsistencyDimension(BaseDimension):
    """Calculate consistency score for data quality assessment."""
    
    def __init__(self):
        """Initialize consistency dimension calculator."""
        super().__init__("Consistency")
        self.expected_correlations = EXPECTED_CORRELATIONS
        self.correlation_tolerance = self.thresholds.get('correlation_tolerance', 0.30)
        
    def calculate(self, window_data: pd.DataFrame, 
                 baselines: Optional[Dict[str, Any]] = None) -> float:
        """
        Calculate consistency score based on relationships and constraints.
        
        Consistency = weighted average of:
        - Logical constraint violations
        - Correlation deviations
        - Energy-power consistency
        
        Args:
            window_data: DataFrame containing window data
            baselines: Correlation baselines for comparison
            
        Returns:
            Consistency score [0, 1]
        """
        if not self.validate_window(window_data):
            return 0.0
            
        consistency_scores = []
        weights = []
        
        # Check logical constraints (weight: 0.4)
        constraint_score = self._check_logical_constraints(window_data)
        consistency_scores.append(constraint_score)
        weights.append(0.4)
        
        # Check expected correlations (weight: 0.3)
        correlation_score = self._check_correlations(window_data, baselines)
        if correlation_score is not None:
            consistency_scores.append(correlation_score)
            weights.append(0.3)
        
        # Check energy-power consistency (weight: 0.3)
        energy_score = self._check_energy_consistency(window_data)
        if energy_score is not None:
            consistency_scores.append(energy_score)
            weights.append(0.3)
        
        # Calculate weighted average
        if consistency_scores:
            # Normalize weights if not all checks were performed
            weights = np.array(weights[:len(consistency_scores)])
            weights = weights / weights.sum()
            consistency_score = np.average(consistency_scores, weights=weights)
        else:
            consistency_score = 1.0
            
        self.logger.debug(
            f"Consistency: {consistency_score:.3f} "
            f"(constraint: {constraint_score:.3f}, "
            f"correlation: {correlation_score if correlation_score else 'N/A':.3f}, "
            f"energy: {energy_score if energy_score else 'N/A':.3f})"
        )
        
        return self.normalize_score(consistency_score)
    
    def _check_logical_constraints(self, window_data: pd.DataFrame) -> float:
        """
        Check logical constraints between metrics.
        
        Args:
            window_data: DataFrame containing window data
            
        Returns:
            Score based on constraint violations
        """
        violations = 0
        total_checks = 0
        
        # PRB Used <= PRB Available
        if 'RRU.PrbUsedDl' in window_data.columns and 'RRU.PrbAvailDl' in window_data.columns:
            mask = window_data[['RRU.PrbUsedDl', 'RRU.PrbAvailDl']].notna().all(axis=1)
            if mask.any():
                violations += np.sum(
                    window_data.loc[mask, 'RRU.PrbUsedDl'] > 
                    window_data.loc[mask, 'RRU.PrbAvailDl']
                )
                total_checks += mask.sum()
        
        if 'RRU.PrbUsedUl' in window_data.columns and 'RRU.PrbAvailUl' in window_data.columns:
            mask = window_data[['RRU.PrbUsedUl', 'RRU.PrbAvailUl']].notna().all(axis=1)
            if mask.any():
                violations += np.sum(
                    window_data.loc[mask, 'RRU.PrbUsedUl'] > 
                    window_data.loc[mask, 'RRU.PrbAvailUl']
                )
                total_checks += mask.sum()
        
        # RRC.ConnMean <= RRC.ConnMax
        if 'RRC.ConnMean' in window_data.columns and 'RRC.ConnMax' in window_data.columns:
            mask = window_data[['RRC.ConnMean', 'RRC.ConnMax']].notna().all(axis=1)
            if mask.any():
                violations += np.sum(
                    window_data.loc[mask, 'RRC.ConnMean'] > 
                    window_data.loc[mask, 'RRC.ConnMax']
                )
                total_checks += mask.sum()
        
        # Average MIMO layers <= Max MIMO layers
        if 'CARR.AverageLayersDl' in window_data.columns and 'RRU.MaxLayerDlMimo' in window_data.columns:
            mask = window_data[['CARR.AverageLayersDl', 'RRU.MaxLayerDlMimo']].notna().all(axis=1)
            if mask.any():
                violations += np.sum(
                    window_data.loc[mask, 'CARR.AverageLayersDl'] > 
                    window_data.loc[mask, 'RRU.MaxLayerDlMimo']
                )
                total_checks += mask.sum()
        
        if total_checks > 0:
            violation_ratio = violations / total_checks
            return 1.0 - violation_ratio
        return 1.0
    
    def _check_correlations(self, window_data: pd.DataFrame, 
                          baselines: Optional[Dict] = None) -> Optional[float]:
        """
        Check if correlations match expected patterns.
        
        Args:
            window_data: DataFrame containing window data
            baselines: Correlation baselines
            
        Returns:
            Score based on correlation deviations or None if not calculable
        """
        correlation_scores = []
        
        for corr_name, expected_corr in self.expected_correlations.items():
            metric1, metric2 = expected_corr['metrics']
            expected_value = expected_corr['expected']
            
            # Check if both metrics exist
            if metric1 not in window_data.columns or metric2 not in window_data.columns:
                continue
                
            # Get non-null pairs
            mask = window_data[[metric1, metric2]].notna().all(axis=1)
            if mask.sum() < 10:  # Need at least 10 points for correlation
                continue
                
            # Calculate correlation
            actual_corr = window_data.loc[mask, [metric1, metric2]].corr().iloc[0, 1]
            
            # Compare with baseline if available
            if baselines and 'correlation_baselines' in baselines:
                baseline_corr = baselines['correlation_baselines'].get(corr_name)
                if baseline_corr is not None:
                    expected_value = baseline_corr
            
            # Calculate deviation score
            deviation = abs(actual_corr - expected_value)
            score = max(0, 1.0 - (deviation / self.correlation_tolerance))
            correlation_scores.append(score)
            
        if correlation_scores:
            return np.mean(correlation_scores)
        return None
    
    def _check_energy_consistency(self, window_data: pd.DataFrame) -> Optional[float]:
        """
        Check energy-power consistency: Energy ≈ Power × Time.
        
        Args:
            window_data: DataFrame containing window data
            
        Returns:
            Score based on energy consistency or None if not calculable
        """
        if 'PEE.Energy_interval' not in window_data.columns or \
           'PEE.AvgPower' not in window_data.columns:
            return None
            
        # Get non-null pairs
        mask = window_data[['PEE.Energy_interval', 'PEE.AvgPower']].notna().all(axis=1)
        if mask.sum() == 0:
            return None
            
        # Calculate expected energy from power (assuming 60-second intervals)
        interval_hours = 60 / 3600  # 1 minute in hours
        expected_energy = (window_data.loc[mask, 'PEE.AvgPower'] / 1000) * interval_hours
        actual_energy = window_data.loc[mask, 'PEE.Energy_interval']
        
        # Calculate relative error
        relative_errors = np.abs(actual_energy - expected_energy) / (expected_energy + 1e-10)
        
        # Score based on median relative error
        median_error = np.median(relative_errors)
        tolerance = self.thresholds.get('energy_tolerance', 0.20)  # 20% tolerance
        
        if median_error <= tolerance:
            score = 1.0 - (median_error / tolerance)
        else:
            score = max(0, 1.0 - median_error)
            
        return score