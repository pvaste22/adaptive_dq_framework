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

from ..common.logger import get_phase1_logger
from ..common.constants import (
    CELL_DTYPES, UE_DTYPES, COLUMN_NAMES, 
    EXPECTED_ENTITIES, EXPECTED_PATTERNS, 
    PATTERNS, DATA_QUIRKS, BAND_LIMITS
)
from ..common.exceptions import DataLoadingError, DataValidationError

class DataLoader:
    """Loads and preprocesses VIAVI O-RAN dataset."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = get_phase1_logger('data_loader')
        self.band_limits = BAND_LIMITS
        self.expected_patterns = EXPECTED_PATTERNS
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load both CellReports and UEReports."""
        self.logger.info("Loading VIAVI O-RAN dataset...")
        
        # Load raw data
        cell_data = self._load_cell_reports()
        ue_data = self._load_ue_reports()
        
        # Apply preprocessing
        cell_data = self._preprocess_cell_data(cell_data)
        ue_data = self._preprocess_ue_data(ue_data)
        
        # Validate data
        if self.config.get('validate_on_load', True):
            self._validate_data(cell_data, ue_data)
        
        self.logger.info(f"Loaded {len(cell_data)} cell records, {len(ue_data)} UE records")
        return cell_data, ue_data
    
    def _load_cell_reports(self) -> pd.DataFrame:
        """Load CellReports.csv with proper data types."""
        file_path = Path(self.config['cell_file'])
        if not file_path.exists():
            raise DataLoadingError(f"Cell reports file not found: {file_path}")
        
        df = pd.read_csv(file_path, dtype=CELL_DTYPES, low_memory=False)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        return df
    
    def _load_ue_reports(self) -> pd.DataFrame:
        """Load UEReports.csv with proper data types."""
        file_path = Path(self.config['ue_file'])
        if not file_path.exists():
            raise DataLoadingError(f"UE reports file not found: {file_path}")
        
        df = pd.read_csv(file_path, dtype=UE_DTYPES, low_memory=False)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        return df
    
    def _preprocess_cell_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing to cell data."""
        df = df.copy()
        
        # Extract band/site/cell from cell names
        pattern = PATTERNS['cell_name_parser']
        df[['Site', 'Band', 'Cell_ID']] = df[COLUMN_NAMES['cell_entity']].str.extract(pattern)
        
        # Note: Unit conversions moved to unit_converter.py
        # Just flag data issues here
        df['Data_quality_flags'] = self._generate_quality_flags_cell(df)
        
        return df
    
    def _preprocess_ue_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing to UE data."""
        df = df.copy()
        
        # Mark CQI interpretation (0 = no measurement)
        if DATA_QUIRKS['cqi_zero_is_no_measurement']:
            df['CQI_measured_dl'] = df['DRB.UECqiDl'] > 0
            df['CQI_measured_ul'] = df['DRB.UECqiUl'] > 0
        
        # Flag unreliable TB counters
        if DATA_QUIRKS['tb_counters_unreliable']:
            df['TB_reliable'] = False
        
        df['Data_quality_flags'] = self._generate_quality_flags_ue(df)
        
        return df
    
    def _generate_quality_flags_cell(self, df: pd.DataFrame) -> pd.Series:
        """Generate quality flags for cell data."""
        flags = []
        
        for idx, row in df.iterrows():
            cell_flags = []
            
            # Physical law violations
            if 'RRU.PrbUsedDl' in df.columns and 'RRU.PrbAvailDl' in df.columns:
                if row['RRU.PrbUsedDl'] > row['RRU.PrbAvailDl']:
                    cell_flags.append('PRB_DL_EXCEEDED')
            
            # Band limit violations
            if 'Band' in df.columns and row['Band'] in self.band_limits:
                limit = self.band_limits[row['Band']]['prb_count']
                if row.get('RRU.PrbUsedDl', 0) > limit:
                    cell_flags.append('BAND_LIMIT_VIOLATION')
            
            flags.append('|'.join(cell_flags) if cell_flags else 'OK')
        
        return pd.Series(flags, index=df.index)
    
    def _generate_quality_flags_ue(self, df: pd.DataFrame) -> pd.Series:
        """Generate quality flags for UE data."""
        flags = []
        
        for idx, row in df.iterrows():
            ue_flags = []
            
            # TB counter reliability
            if DATA_QUIRKS['tb_counters_unreliable']:
                if row.get('TB.TotNbrDl', 0) == 0 and row.get('DRB.UEThpDl', 0) > 0:
                    ue_flags.append('TB_COUNTER_UNRELIABLE')
            
            # CQI range validation
            if row.get('DRB.UECqiDl', 0) > 15:
                ue_flags.append('CQI_DL_OUT_OF_RANGE')
            
            flags.append('|'.join(ue_flags) if ue_flags else 'OK')
        
        return pd.Series(flags, index=df.index)
    
    def _validate_data(self, cell_data: pd.DataFrame, ue_data: pd.DataFrame):
        """Validate loaded data."""
        # Check entity counts
        unique_cells = cell_data[COLUMN_NAMES['cell_entity']].nunique()
        unique_ues = ue_data[COLUMN_NAMES['ue_entity']].nunique()
        
        if unique_cells != EXPECTED_ENTITIES['cells']:
            self.logger.warning(f"Expected {EXPECTED_ENTITIES['cells']} cells, found {unique_cells}")
        if unique_ues != EXPECTED_ENTITIES['ues']:
            self.logger.warning(f"Expected {EXPECTED_ENTITIES['ues']} UEs, found {unique_ues}")
        
        # Validate expected patterns
        if 'DRB.UECqiDl' in ue_data.columns:
            cqi_zero_rate = (ue_data['DRB.UECqiDl'] == 0).mean()
            expected = self.expected_patterns['cqi_zero_rate']
            if abs(cqi_zero_rate - expected) > 0.10:
                self.logger.warning(f"CQI zero rate {cqi_zero_rate:.3f} deviates from expected {expected:.3f}")
        
        self.logger.info("Data validation complete")