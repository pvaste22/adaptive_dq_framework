"""
Module: Data Processing
Phase: 1
Author: Pranjal V
Created: 06/09/2025
Purpose: Unit converter for VIAVI dataset
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple

from ..common.logger import get_phase1_logger
from ..common.constants import (
    BAND_SPECS, QUALITY_THRESHOLDS, DATA_QUIRKS,
    COLUMN_NAMES, PATTERNS
)
from ..common.exceptions import UnitConversionError

class UnitConverter:
    """Handles critical unit conversions for VIAVI dataset."""
    
    def __init__(self, config: Dict):
        self.logger = get_phase1_logger('unit_converter')
        self.config = config
        self.band_limits = {band: spec['prb_count'] for band, spec in BAND_SPECS.items()}
        self.validation_thresholds = QUALITY_THRESHOLDS
        
    def convert_prb_percentage_to_absolute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert PrbTot from percentage to absolute values."""
        df = df.copy()
        
        if not DATA_QUIRKS['prb_tot_is_percentage']:
            return df
        
        self.logger.info("Converting PrbTot from percentage to absolute...")
        
        # PrbTot is percentage (0-100), convert to absolute
        if 'RRU.PrbTotDl' in df.columns and 'RRU.PrbAvailDl' in df.columns:
            df['RRU.PrbTotDl_abs'] = (df['RRU.PrbTotDl'] / 100.0) * df['RRU.PrbAvailDl']
        if 'RRU.PrbTotUl' in df.columns and 'RRU.PrbAvailUl' in df.columns:
            df['RRU.PrbTotUl_abs'] = (df['RRU.PrbTotUl'] / 100.0) * df['RRU.PrbAvailUl']
        
        df['prb_conversion_applied'] = True
        self.logger.info("PrbTot conversion completed")
        
        return df
    
    def convert_cumulative_energy_to_interval(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert cumulative energy to per-interval energy."""
        df = df.copy()
        
        if not DATA_QUIRKS['energy_is_cumulative']:
            return df
        
        self.logger.info("Converting cumulative energy to per-interval...")
        
        # Sort by entity and timestamp
        entity_col = COLUMN_NAMES['cell_entity']
        df = df.sort_values([entity_col, 'timestamp'])
        
        # Calculate interval energy by differencing
        df['PEE.Energy_interval'] = df.groupby(entity_col)['PEE.Energy'].diff()
        
        # First timestamp per entity has no previous value
        first_timestamps = df.groupby(entity_col).head(1).index
        df.loc[first_timestamps, 'PEE.Energy_interval'] = np.nan
        
        # Validate against expected formula
        interval_seconds = self.config.get('measurement_interval_seconds', 60)
        interval_hours = interval_seconds / 3600.0
        df['PEE.Energy_expected'] = (df['PEE.AvgPower'] / 1000.0) * interval_hours
        
        # Calculate validation error
        valid_mask = pd.notna(df['PEE.Energy_interval']) & (df['PEE.Energy_interval'] > 0)
        if valid_mask.any():
            error_percent = np.abs(
                df.loc[valid_mask, 'PEE.Energy_interval'] - 
                df.loc[valid_mask, 'PEE.Energy_expected']
            ) / df.loc[valid_mask, 'PEE.Energy_expected']
            
            df['Energy_validation_error'] = np.nan
            df.loc[valid_mask, 'Energy_validation_error'] = error_percent
        
        df['energy_conversion_applied'] = True
        self.logger.info("Energy conversion completed")
        
        return df
    
    def handle_qos_flow_semantics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle QosFlow 1-second semantics in 60-second intervals."""
        df = df.copy()
        
        if not DATA_QUIRKS['qos_flow_has_1s_semantics']:
            return df
        
        self.logger.info("Processing QosFlow 1-second semantics...")
        
        # Flag the semantic mismatch
        if 'QosFlow.TotPdcpPduVolumeDl' in df.columns:
            df['qos_flow_1s_semantics'] = True
        
        return df
    
    def standardize_units_comprehensive(self, 
                                      cell_data: pd.DataFrame,
                                      ue_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Apply all unit conversions."""
        self.logger.info("Starting comprehensive unit standardization...")
        
        # Extract band if not present
        if 'Band' not in cell_data.columns and COLUMN_NAMES['cell_entity'] in cell_data.columns:
            cell_data['Band'] = cell_data[COLUMN_NAMES['cell_entity']].str.extract(
                PATTERNS['band_extractor']
            )[0]
        
        # Apply conversions to cell data
        if self.config.get('prb_percentage_to_absolute', True):
            cell_data = self.convert_prb_percentage_to_absolute(cell_data)
        
        if self.config.get('energy_cumulative_to_interval', True):
            cell_data = self.convert_cumulative_energy_to_interval(cell_data)
        
        if self.config.get('qos_flow_validation', True):
            cell_data = self.handle_qos_flow_semantics(cell_data)
        
        # Process UE data (minimal conversions)
        # TB counters marked as unreliable, CQI=0 interpretation handled in data_loader
        
        # Add metadata
        cell_data['unit_conversion_version'] = '1.0'
        ue_data['unit_conversion_version'] = '1.0'
        
        self.logger.info("Unit standardization completed")
        
        return cell_data, ue_data