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
import sys
import os
import json
from datetime import datetime 


project_root = Path(__file__).parent.parent.parent 
src_path = project_root / 'src'
sys.path.insert(0, str(src_path))

from common.logger import get_phase1_logger
from common.constants import (
    CELL_DTYPES, UE_DTYPES, COLUMN_NAMES, 
    PATTERNS, BAND_LIMITS, VIAVI_CONFIG
)
from common.exceptions import DataLoadingError, DataValidationError

class DataLoader:
    """Loads and preprocesses VIAVI O-RAN dataset."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = get_phase1_logger('data_loader')
        self.band_limits = BAND_LIMITS
        self.dataset_config = VIAVI_CONFIG
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load both CellReports and UEReports."""
        self.logger.info("Loading VIAVI O-RAN dataset...")
        
        # Load raw data
        cell_data = self._load_cell_reports()
        ue_data = self._load_ue_reports()
        
        
        self.logger.info(f"Loaded {len(cell_data)} cell records, {len(ue_data)} UE records")
        return cell_data, ue_data
    
    def _load_cell_reports(self) -> pd.DataFrame:
        """Load CellReports.csv with proper data types."""
        file_path = Path(self.dataset_config['cell_reports_file'])
        if not file_path.exists():
            raise DataLoadingError(f"Cell reports file not found: {file_path}")
        
        # Define data types for memory efficiency
        df = pd.read_csv(file_path, dtype=CELL_DTYPES, low_memory=False)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        return df
    
    def _load_ue_reports(self) -> pd.DataFrame:
        """Load UEReports.csv with proper data types."""
        file_path = Path(self.dataset_config['ue_reports_file'])
        if not file_path.exists():
            raise DataLoadingError(f"UE reports file not found: {file_path}")
        
        df = pd.read_csv(file_path, dtype=UE_DTYPES, low_memory=False)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        return df
    

   