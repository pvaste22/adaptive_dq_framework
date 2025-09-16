"""
Module: Data Processing
Phase: 1
Author: Pranjal V
Created: 06/09/2025
Purpose: Data loader for VIAVI O-RAN dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
import re

from common.logger import get_phase1_logger
from common.constants import (
    CELL_DTYPES, UE_DTYPES, COLUMN_NAMES, 
    PATTERNS, BAND_LIMITS, VIAVI_CONFIG, DATA_FILES,
    EXPECTED_ENTITIES, MEAS_INTERVAL_SEC
)
from common.exceptions import DataLoadingError, DataValidationError

class DataLoader:
    """Loads and preprocesses VIAVI O-RAN dataset."""
    
    def __init__(self):
        """Initialize DataLoader using constants directly."""
        self.logger = get_phase1_logger('data_loader')
        self.band_limits = BAND_LIMITS
        self.dataset_config = VIAVI_CONFIG
        self.column_names = COLUMN_NAMES
        self.expected_entities = EXPECTED_ENTITIES
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load both CellReports and UEReports with necessary corrections."""
        self.logger.info("Loading VIAVI O-RAN dataset...")
        
        # Load raw data
        cell_data = self._load_cell_reports()
        ue_data = self._load_ue_reports()
        
        # Apply necessary data corrections (not quality assessment)
        cell_data = self._correct_cell_data(cell_data)
        ue_data = self._correct_ue_data(ue_data)
        
        self.logger.info(f"Loaded {len(cell_data)} cell records, {len(ue_data)} UE records")
        
        # Log correction statistics
        if 'cqi_correction_stats' in ue_data.attrs:
            self.logger.info(f"UE CQI corrections: {ue_data.attrs['cqi_correction_stats']}")
        
        return cell_data, ue_data
    
    def _load_cell_reports(self) -> pd.DataFrame:
        """Load CellReports.csv with proper data types."""
        file_path = DATA_FILES['cell_reports']
        
        if not file_path.exists():
            raise DataLoadingError(f"Cell reports file not found: {file_path}")
        
        self.logger.info(f"Loading cell reports from {file_path}")
        
        # Load with specified dtypes for memory efficiency
        df = pd.read_csv(file_path, dtype=CELL_DTYPES, low_memory=False)
        
        # Convert timestamp from Unix to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        
        return df
    
    def _load_ue_reports(self) -> pd.DataFrame:
        """Load UEReports.csv with proper data types."""
        file_path = DATA_FILES['ue_reports']
        
        if not file_path.exists():
            raise DataLoadingError(f"UE reports file not found: {file_path}")
        
        self.logger.info(f"Loading UE reports from {file_path}")
        
        df = pd.read_csv(file_path, dtype=UE_DTYPES, low_memory=False)
        
        # Convert timestamp from Unix to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        
        return df
    
    def _correct_cell_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply necessary data corrections for cell data.
        Only corrections, not quality assessment.
        """
        df = df.copy()
        
        # Extract band information - required for framework operations
        if 'Band' not in df.columns and self.column_names['cell_entity'] in df.columns:
            df['Band'] = df[self.column_names['cell_entity']].str.extract(
                PATTERNS['band_extractor']
            )[0]
            self.logger.info(f"Extracted band information: {df['Band'].value_counts().to_dict()}")
        
        # No CQI columns in cell data - those are UE-specific metrics
        
        return df
    
    def _correct_ue_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply necessary data corrections for UE data.
        Only corrections, not quality assessment.
        """
        df = df.copy()
        correction_stats = {}
        
        # Handle CQI=0 as no measurement - critical for UE data
        for col in ['DRB.UECqiDl', 'DRB.UECqiUl']:
            if col in df.columns:
                zeros_count = (df[col] == 0).sum()
                if zeros_count > 0:
                    df[col] = df[col].replace(0, np.nan)
                    correction_stats[col] = zeros_count
                    total_count = len(df)
                    zero_percent = (zeros_count / total_count) * 100
                    self.logger.info(
                        f"Replaced {zeros_count} zeros in {col} with NaN "
                        f"({zero_percent:.1f}% of records had no measurement)"
                    )
        
        # Store correction statistics as metadata
        df.attrs['cqi_correction_stats'] = correction_stats
        
        # Note: We do NOT flag TB counters here - that's for quality assessment phase
        
        return df
    
    def get_data_summary(self, cell_data: pd.DataFrame, ue_data: pd.DataFrame) -> Dict:
        """
        Generate comprehensive summary statistics for loaded data.
        
        Args:
            cell_data: Cell reports DataFrame
            ue_data: UE reports DataFrame
            
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'cell_data': self._summarize_dataframe(cell_data, 'cell'),
            'ue_data': self._summarize_dataframe(ue_data, 'ue'),
            'data_alignment': self._check_data_alignment(cell_data, ue_data),
            'corrections_applied': self._summarize_corrections(cell_data, ue_data)
        }
        
        return summary
    
    def _summarize_dataframe(self, df: pd.DataFrame, data_type: str) -> Dict:
        """Generate summary for a single dataframe."""
        entity_col = self.column_names.get(f'{data_type}_entity')
        
        summary = {
            'total_records': len(df),
            'shape': df.shape,
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'columns': list(df.columns),
            'dtypes': df.dtypes.astype(str).to_dict()
        }
        
        # Entity statistics
        if entity_col and entity_col in df.columns:
            summary['unique_entities'] = df[entity_col].nunique()
            summary['entity_list'] = sorted(df[entity_col].unique().tolist())[:10]  # First 10
            if data_type == 'cell':
                summary['expected_entities'] = self.expected_entities.get('cells', 52)
            else:
                summary['expected_entities'] = self.expected_entities.get('ues', 48)
            summary['entity_completeness'] = summary['unique_entities'] / summary['expected_entities']
        
        # Timestamp statistics
        if 'timestamp' in df.columns:
            summary['unique_timestamps'] = df['timestamp'].nunique()
            summary['date_range'] = (
                str(df['timestamp'].min()),
                str(df['timestamp'].max())
            )
            summary['time_span_hours'] = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600
            
            # Check timestamp interval consistency
            timestamps = df['timestamp'].unique()
            if len(timestamps) > 1:
                intervals = np.diff(sorted(timestamps))
                expected_interval = pd.Timedelta(seconds=MEAS_INTERVAL_SEC)
                summary['consistent_intervals'] = all(abs(i - expected_interval) < pd.Timedelta(seconds=5) for i in intervals)
        
        # Band statistics (for cell data)
        if 'Band' in df.columns:
            summary['bands_present'] = sorted(df['Band'].dropna().unique().tolist())
            summary['band_distribution'] = df['Band'].value_counts().to_dict()
        
        # Missing value analysis
        missing_counts = df.isnull().sum()
        summary['missing_values'] = {
            'total_missing': int(missing_counts.sum()),
            'columns_with_missing': missing_counts[missing_counts > 0].to_dict(),
            'complete_columns': list(missing_counts[missing_counts == 0].index)
        }
        
        # Key metrics statistics (sample) - adjusted for correct columns
        key_metrics = {
            'cell': ['RRU.PrbUsedDl', 'RRU.PrbUsedUl', 'DRB.UEThpDl', 'DRB.UEThpUl', 
                     'RRC.ConnMean', 'PEE.AvgPower'],
            'ue': ['DRB.UECqiDl', 'DRB.UECqiUl', 'DRB.UEThpDl', 'DRB.UEThpUl', 
                   'RRU.PrbUsedDl', 'RRU.PrbUsedUl']
        }
        
        metrics_stats = {}
        for metric in key_metrics.get(data_type, []):
            if metric in df.columns:
                metric_data = df[metric].dropna()
                if len(metric_data) > 0:
                    metrics_stats[metric] = {
                        'mean': float(metric_data.mean()),
                        'std': float(metric_data.std()),
                        'min': float(metric_data.min()),
                        'max': float(metric_data.max()),
                        'nulls': int(df[metric].isnull().sum()),
                        'null_percent': float(df[metric].isnull().sum() / len(df) * 100)
                    }
        summary['key_metrics_stats'] = metrics_stats
        
        return summary
    
    def _check_data_alignment(self, cell_data: pd.DataFrame, ue_data: pd.DataFrame) -> Dict:
        """Check alignment between cell and UE data."""
        alignment = {
            'timestamp_overlap': {},
            'record_count_ratio': {},
            'time_synchronization': {}
        }
        
        if 'timestamp' in cell_data.columns and 'timestamp' in ue_data.columns:
            cell_timestamps = set(cell_data['timestamp'].unique())
            ue_timestamps = set(ue_data['timestamp'].unique())
            
            common_timestamps = cell_timestamps.intersection(ue_timestamps)
            
            alignment['timestamp_overlap'] = {
                'cell_unique_timestamps': len(cell_timestamps),
                'ue_unique_timestamps': len(ue_timestamps),
                'common_timestamps': len(common_timestamps),
                'overlap_percentage': (len(common_timestamps) / max(len(cell_timestamps), len(ue_timestamps))) * 100 if cell_timestamps else 0,
                'cell_only_timestamps': len(cell_timestamps - ue_timestamps),
                'ue_only_timestamps': len(ue_timestamps - cell_timestamps)
            }
            
            # Check if timestamps are synchronized
            if common_timestamps:
                alignment['time_synchronization']['synchronized'] = True
                alignment['time_synchronization']['common_range'] = (
                    str(min(common_timestamps)),
                    str(max(common_timestamps))
                )
            else:
                alignment['time_synchronization']['synchronized'] = False
                alignment['time_synchronization']['issue'] = "No overlapping timestamps"
        
        # Record count analysis
        alignment['record_count_ratio'] = {
            'cell_records': len(cell_data),
            'ue_records': len(ue_data),
            'ratio': len(cell_data) / len(ue_data) if len(ue_data) > 0 else 0,
            'expected_ratio': self.expected_entities['cells'] / self.expected_entities['ues']
        }
        
        return alignment
    
    def _summarize_corrections(self, cell_data: pd.DataFrame, ue_data: pd.DataFrame) -> Dict:
        """Summarize data corrections that were applied."""
        corrections = {
            'cqi_corrections': {},
            'band_extraction': {},
            'timestamp_conversion': True
        }
        
        # CQI corrections from UE data only
        if hasattr(ue_data, 'attrs') and 'cqi_correction_stats' in ue_data.attrs:
            corrections['cqi_corrections']['ue'] = ue_data.attrs['cqi_correction_stats']
        
        # Band extraction from cell data
        if 'Band' in cell_data.columns:
            corrections['band_extraction'] = {
                'applied': True,
                'bands_found': list(cell_data['Band'].dropna().unique())
            }
        else:
            corrections['band_extraction']['applied'] = False
        
        return corrections