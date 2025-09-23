"""
Module: Quality Dimensions
Phase: 2
Author: Pranjal V
Created: 23/09/2025
Purpose: Completeness dimension calculator.
Implements checks C1-C4 for data completeness assessment.
"""
"""
Completeness dimension calculator.
Pure scoring implementation without judgments.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from quality_dimensions.base_dimension import BaseDimension
from common.constants import (
    EXPECTED_ENTITIES,
    MEAS_INTERVAL_SEC,
    EXPECTED_PATTERNS,
    DATA_QUIRKS
)


class CompletenessDimension(BaseDimension):
    """Calculates completeness score for data windows."""
    
    def __init__(self):
        """Initialize completeness dimension."""
        super().__init__('completeness')
        
        # Load expected values from configuration
        self.expected_cells = EXPECTED_ENTITIES['cells']
        self.expected_ues = EXPECTED_ENTITIES['ues']
        self.measurement_interval = MEAS_INTERVAL_SEC
        
        # Expected patterns from data analysis
        self.cqi_expected_nan_rate = EXPECTED_PATTERNS.get('cqi_no_measurement_rate', 0.60)
        self.mimo_expected_zero_rate = EXPECTED_PATTERNS.get('mimo_zero_rate', 0.86)
        
        # Define mandatory fields
        self.mandatory_fields_cell = [
            self.timestamp_col,
            self.cell_entity_col,
            'RRU.PrbAvailDl',
            'RRU.PrbAvailUl'
        ]
        self.mandatory_fields_ue = [
            self.timestamp_col,
            self.ue_entity_col
        ]
        self.logger.info(f"Completeness dimension initialized: {self.expected_cells} cells, {self.expected_ues} UEs expected")
        self.logger.debug(f"Expected patterns - CQI NaN: {self.cqi_expected_nan_rate:.1%}, MIMO zeros: {self.mimo_expected_zero_rate:.1%}")
    
    def calculate_score(self, window_data: Dict) -> Dict:
        """
        Calculate completeness score.
        
        Args:
            window_data: Window data dictionary
            
        Returns:
            Dictionary with score and measurement details
        """
        is_valid, error_msg = self.validate_window_data(window_data)
        if not is_valid:
            self.logger.error(f"Window validation failed: {error_msg}")
            return self.format_result(0.0, {'validation_error': error_msg})
        cell_data = window_data.get('cell_data', pd.DataFrame())
        ue_data = window_data.get('ue_data', pd.DataFrame())
        window_id = window_data.get('metadata', {}).get('window_id', 'unknown')
        self.logger.debug(f"Processing window {window_id}: {len(cell_data)} cell records, {len(ue_data)} UE records")
        
        # Calculate individual completeness components
        c1_score, c1_details = self._score_mandatory_fields(cell_data, ue_data)
        c2_score, c2_details = self._score_entity_presence(cell_data, ue_data)
        c3_score, c3_details = self._score_time_continuity(cell_data, ue_data)
        c4_score, c4_details = self._score_timestamp_alignment(cell_data, ue_data)
        c5_score, c5_details = self._score_field_completeness(cell_data, ue_data)
        
        # Combine scores (simple average)
        overall_score = np.mean([c1_score, c2_score, c3_score, c4_score, c5_score])
        self.logger.debug(f"Component scores - C1:{c1_score:.2f} C2:{c2_score:.2f} C3:{c3_score:.2f} "
                     f"C4:{c4_score:.2f} C5:{c5_score:.2f} Overall:{overall_score:.2f}")
    
        
        return {
            'score': overall_score,
            'details': {
                'mandatory_fields_score': c1_score,
                'entity_presence_score': c2_score,
                'time_continuity_score': c3_score,
                'timestamp_alignment_score': c4_score,
                'field_completeness_score': c5_score,
                'measurements': {
                    'mandatory_fields': c1_details,
                    'entity_presence': c2_details,
                    'time_continuity': c3_details,
                    'timestamp_alignment': c4_details,
                    'field_completeness': c5_details
                }
            }
        }
    
    def _score_mandatory_fields(self, cell_data: pd.DataFrame, 
                               ue_data: pd.DataFrame) -> Tuple[float, Dict]:
        """
        Score based on presence of mandatory fields.
        
        Returns:
            Tuple of (score, measurement details)
        """
        details = {}
        
        # Check cell mandatory fields
        cell_nulls = 0
        for field in self.mandatory_fields_cell:
            if field in cell_data.columns:
                null_count = cell_data[field].isnull().sum()
                cell_nulls += null_count
                details[f'cell_{field}_nulls'] = int(null_count)
                if null_count > 0:
                    self.logger.debug(f"Mandatory cell field '{field}' has {null_count} nulls")
            else:
                self.logger.warning(f"Mandatory cell field '{field}' missing from data")
                details[f'cell_{field}'] = 'missing_column'
                cell_nulls += len(cell_data)
        # Check UE mandatory fields
        ue_nulls = 0
        for field in self.mandatory_fields_ue:
            if field in ue_data.columns:
                null_count = ue_data[field].isnull().sum()
                ue_nulls += null_count
                details[f'ue_{field}_nulls'] = int(null_count)
                if null_count > 0:
                    self.logger.debug(f"Mandatory UE field '{field}' has {null_count} nulls")
            else:
                self.logger.warning(f"Mandatory UE field '{field}' missing from data")
                details[f'ue_{field}'] = 'missing_column'
                ue_nulls += len(ue_data)
        
        # Score: 1.0 if no nulls, 0.0 if any nulls in mandatory fields
        score = 1.0 if (cell_nulls == 0 and ue_nulls == 0) else 0.0
        self.logger.debug(f"C1 Mandatory fields score: {score:.2f} (cell_nulls: {cell_nulls}, ue_nulls: {ue_nulls})")
        
        return score, details
    
    def _score_entity_presence(self, cell_data: pd.DataFrame,
                              ue_data: pd.DataFrame) -> Tuple[float, Dict]:
        """
        Score based on percentage of entities reporting.
        
        Returns:
            Tuple of (score, measurement details)
        """
        details = {}
        scores = []
        
        # Check cells per timestamp
        if self.timestamp_col in cell_data.columns and self.cell_entity_col in cell_data.columns:
            cells_per_timestamp = cell_data.groupby(self.timestamp_col)[self.cell_entity_col].nunique()
            for timestamp, count in cells_per_timestamp.items():
                ratio = count / self.expected_cells
                scores.append(ratio)
            details['avg_cell_presence_ratio'] = float(np.mean(scores)) if scores else 0.0
        
        # Check UEs per timestamp
        if self.timestamp_col in ue_data.columns and self.ue_entity_col in ue_data.columns:
            ues_per_timestamp = ue_data.groupby(self.timestamp_col)[self.ue_entity_col].nunique()
            for timestamp, count in ues_per_timestamp.items():
                ratio = count / self.expected_ues
                scores.append(ratio)
            details['avg_ue_presence_ratio'] = float(np.mean(scores)) if scores else 0.0
        
        score = np.mean(scores) if scores else 0.0
        return score, details
    
    def _score_time_continuity(self, cell_data: pd.DataFrame,
                              ue_data: pd.DataFrame) -> Tuple[float, Dict]:
        """
        Score based on time series continuity.
        
        Returns:
            Tuple of (score, measurement details)
        """
        details = {}
        gap_count = 0
        
        # Check cell data gaps
        if self.timestamp_col in cell_data.columns:
            timestamps = pd.to_datetime(cell_data[self.timestamp_col], unit='s')
            unique_timestamps = sorted(timestamps.unique())
            
            for i in range(1, len(unique_timestamps)):
                gap = (unique_timestamps[i] - unique_timestamps[i-1]).total_seconds()
                if abs(gap - self.measurement_interval) > 5:  # 5 second tolerance
                    gap_count += 1
            
            details['cell_time_gaps'] = gap_count
        
        # Check UE data gaps  
        if self.timestamp_col in ue_data.columns:
            timestamps = pd.to_datetime(ue_data[self.timestamp_col], unit='s')
            unique_timestamps = sorted(timestamps.unique())
            
            ue_gaps = 0
            for i in range(1, len(unique_timestamps)):
                gap = (unique_timestamps[i] - unique_timestamps[i-1]).total_seconds()
                if abs(gap - self.measurement_interval) > 5:
                    ue_gaps += 1
                    gap_count += 1
            
            details['ue_time_gaps'] = ue_gaps
        
        # Score decreases with more gaps
        score = max(0, 1.0 - (gap_count * 0.2))
        details['total_gaps'] = gap_count
        
        return score, details
    
    def _score_timestamp_alignment(self, cell_data: pd.DataFrame,
                                  ue_data: pd.DataFrame) -> Tuple[float, Dict]:
        """
        Score based on timestamp alignment between datasets.
        
        Returns:
            Tuple of (score, measurement details)
        """
        details = {}
        
        if self.timestamp_col not in cell_data.columns or self.timestamp_col not in ue_data.columns:
            return 0.0, {'error': 'missing timestamp columns'}
        
        cell_timestamps = set(cell_data[self.timestamp_col].unique())
        ue_timestamps = set(ue_data[self.timestamp_col].unique())
        
        intersection = cell_timestamps & ue_timestamps
        
        # Score based on alignment
        score = len(intersection) / len(cell_timestamps) if len(cell_timestamps) > 0 else 0.0
        
        details['common_timestamps'] = len(intersection)
        details['cell_only_timestamps'] = len(cell_timestamps - ue_timestamps)
        details['ue_only_timestamps'] = len(ue_timestamps - cell_timestamps)
        
        return score, details
    
    def _score_field_completeness(self, cell_data: pd.DataFrame,
                                 ue_data: pd.DataFrame) -> Tuple[float, Dict]:
        """
        Score based on field-level completeness with pattern awareness.
        
        Returns:
            Tuple of (score, measurement details)
        """
        details = {}
        all_scores = []
        
        # Skip unreliable TB counters
        skip_fields = ['TB.TotNbrDl', 'TB.TotNbrUl'] if DATA_QUIRKS.get('tb_counters_unreliable') else []
        if skip_fields:
            self.logger.debug(f"Skipping unreliable fields: {skip_fields}")
        
        # Calculate completeness for each field
        for col in cell_data.columns:
            if col not in skip_fields:
                null_ratio = cell_data[col].isnull().sum() / len(cell_data) if len(cell_data) > 0 else 0
                completeness = 1.0 - null_ratio
                
                # Adjust for MIMO fields (zeros are expected)
                if col in ['CARR.AverageLayersDl', 'RRU.MaxLayerDlMimo']:
                    zero_ratio = (cell_data[col] == 0).sum() / len(cell_data) if len(cell_data) > 0 else 0
                    if abs(zero_ratio - self.mimo_expected_zero_rate) < 0.15:
                        completeness = 1.0  # Expected pattern
                        self.logger.debug(f"MIMO field {col}: {zero_ratio:.1%} zeros matches expected pattern")
                
                all_scores.append(completeness)
        
        for col in ue_data.columns:
            if col not in skip_fields:
                null_ratio = ue_data[col].isnull().sum() / len(ue_data) if len(ue_data) > 0 else 0
                completeness = 1.0 - null_ratio
                
                # Adjust for CQI fields (NaN is expected)
                if col in ['DRB.UECqiDl', 'DRB.UECqiUl']:
                    if abs(null_ratio - self.cqi_expected_nan_rate) < 0.15:
                        completeness = 1.0  # Expected pattern
                        self.logger.debug(f"CQI field {col}: {null_ratio:.1%} NaN matches expected pattern")
                
                all_scores.append(completeness)
        
        score = np.mean(all_scores) if all_scores else 0.0
        details['avg_field_completeness'] = float(score)
        details['fields_evaluated'] = len(all_scores)
        self.logger.debug(f"C5 Field completeness: {score:.2f} ({len(all_scores)} fields evaluated)")
    
        
        return score, details