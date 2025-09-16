"""
Module: Data Processing
Phase: 1
Author: Pranjal V
Created: 06/09/2025
Purpose: Unit converter for VIAVI dataset - pure conversions only
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple

from common.logger import get_phase1_logger
from common.constants import (
    BAND_SPECS, DATA_QUIRKS, COLUMN_NAMES, 
    PATTERNS, CONVERSION_CONFIG, MEAS_INTERVAL_SEC
)
from common.exceptions import UnitConversionError

class UnitConverter:
    """Handles critical unit conversions for VIAVI dataset."""
    
    def __init__(self):
        """Initialize UnitConverter using constants directly."""
        self.logger = get_phase1_logger('unit_converter')
        self.band_limits = {band: spec['prb_count'] for band, spec in BAND_SPECS.items()}
        self.conversion_config = CONVERSION_CONFIG
        self.measurement_interval = MEAS_INTERVAL_SEC
        
    def convert_prb_percentage_to_absolute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert PrbTot from percentage to absolute values."""
        df = df.copy()
        
        if not DATA_QUIRKS.get('prb_tot_is_percentage', True):
            self.logger.info("PrbTot conversion not needed - already in absolute values")
            return df
        
        self.logger.info("Converting PrbTot from percentage to absolute...")
        
        conversions_applied = 0
        
        # Convert DL PRBs
        if 'RRU.PrbTotDl' in df.columns and 'RRU.PrbAvailDl' in df.columns:
            # PrbTot is percentage (0-100), convert to absolute
            df['RRU.PrbTotDl_abs'] = (df['RRU.PrbTotDl'] / 100.0) * df['RRU.PrbAvailDl']
            conversions_applied += 1
            self.logger.debug(f"Converted PrbTotDl to absolute values")
        
        # Convert UL PRBs
        if 'RRU.PrbTotUl' in df.columns and 'RRU.PrbAvailUl' in df.columns:
            df['RRU.PrbTotUl_abs'] = (df['RRU.PrbTotUl'] / 100.0) * df['RRU.PrbAvailUl']
            conversions_applied += 1
            self.logger.debug(f"Converted PrbTotUl to absolute values")
        
        if conversions_applied > 0:
            df['prb_conversion_applied'] = True
            self.logger.info(f"PrbTot conversion completed - {conversions_applied} columns converted")
        else:
            self.logger.warning("No PRB columns found for conversion")
        
        return df
    
    def convert_cumulative_energy_to_interval(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert cumulative energy to per-interval energy."""
        df = df.copy()
        
        if not DATA_QUIRKS.get('energy_is_cumulative', True):
            self.logger.info("Energy conversion not needed - already interval-based")
            return df
        
        if 'PEE.Energy' not in df.columns:
            self.logger.warning("PEE.Energy column not found - skipping energy conversion")
            return df
        
        self.logger.info("Converting cumulative energy to per-interval...")
        
        # Sort by entity and timestamp for proper differencing
        entity_col = COLUMN_NAMES.get('cell_entity', 'Viavi.Cell.Name')
        
        if entity_col not in df.columns:
            self.logger.warning(f"Entity column {entity_col} not found - cannot perform energy conversion")
            return df
        
        df = df.sort_values([entity_col, 'timestamp'])
        
        # Calculate interval energy by differencing
        df['PEE.Energy_interval'] = df.groupby(entity_col)['PEE.Energy'].diff()
        
        # First timestamp per entity has no previous value - use the cumulative value
        first_timestamps = df.groupby(entity_col).head(1).index
        df.loc[first_timestamps, 'PEE.Energy_interval'] = df.loc[first_timestamps, 'PEE.Energy']
        
        # Optional: Calculate expected energy for future quality checks (but don't validate here)
        if 'PEE.AvgPower' in df.columns:
            interval_hours = self.measurement_interval / 3600.0
            df['PEE.Energy_expected'] = (df['PEE.AvgPower'] / 1000.0) * interval_hours
            self.logger.debug("Added expected energy column for future validation")
        
        df['energy_conversion_applied'] = True
        self.logger.info("Energy conversion completed")
        
        return df
    
    def handle_qos_flow_semantics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle QosFlow 1-second semantics in 60-second intervals."""
        df = df.copy()
        
        if not DATA_QUIRKS.get('qos_flow_has_1s_semantics', True):
            return df
        
        self.logger.info("Processing QosFlow 1-second semantics...")
        
        # Flag the semantic mismatch for downstream processing
        qos_columns = ['QosFlow.TotPdcpPduVolumeDl', 'QosFlow.TotPdcpPduVolumeUl']
        qos_present = [col for col in qos_columns if col in df.columns]
        
        if qos_present:
            df['qos_flow_1s_semantics'] = True
            self.logger.warning(
                f"QosFlow columns {qos_present} use 1-second semantics "
                f"in {self.measurement_interval}-second intervals"
            )
        else:
            self.logger.info("No QosFlow columns found")
        
        return df
    
    def standardize_units_comprehensive(self, 
                                      cell_data: pd.DataFrame,
                                      ue_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Apply all unit conversions based on configuration."""
        self.logger.info("Starting comprehensive unit standardization...")
        
        # Track what conversions were applied
        conversion_summary = {
            'cell_conversions': [],
            'ue_conversions': []
        }
        
        # Apply conversions to cell data based on configuration
        if self.conversion_config.get('prb_percentage_to_absolute', True):
            cell_data = self.convert_prb_percentage_to_absolute(cell_data)
            if 'prb_conversion_applied' in cell_data.columns:
                conversion_summary['cell_conversions'].append('prb_percentage_to_absolute')
        
        if self.conversion_config.get('energy_cumulative_to_interval', True):
            cell_data = self.convert_cumulative_energy_to_interval(cell_data)
            if 'energy_conversion_applied' in cell_data.columns:
                conversion_summary['cell_conversions'].append('energy_cumulative_to_interval')
        
        if self.conversion_config.get('qos_flow_validation', True):
            cell_data = self.handle_qos_flow_semantics(cell_data)
            if 'qos_flow_1s_semantics' in cell_data.columns:
                conversion_summary['cell_conversions'].append('qos_flow_semantics_flagged')
        
        # Add metadata
        cell_data['unit_conversion_version'] = '1.0'
        ue_data['unit_conversion_version'] = '1.0'
        
        # Store conversion summary as attributes for tracking
        cell_data.attrs['conversions_applied'] = conversion_summary['cell_conversions']
        ue_data.attrs['conversions_applied'] = conversion_summary.get('ue_conversions', [])
        
        # Log summary
        self.logger.info(f"Unit standardization completed")
        self.logger.info(f"Cell conversions applied: {conversion_summary['cell_conversions']}")
        
        return cell_data, ue_data
    
    def get_conversion_summary(self, cell_data: pd.DataFrame, ue_data: pd.DataFrame) -> Dict:
        """
        Get summary of what conversions were applied.
        
        Args:
            cell_data: Converted cell data
            ue_data: Converted UE data
            
        Returns:
            Summary of applied conversions
        """
        summary = {
            'cell_data': {
                'conversions_applied': cell_data.attrs.get('conversions_applied', []),
                'new_columns_created': [],
                'flags_added': []
            },
            'ue_data': {
                'conversions_applied': ue_data.attrs.get('conversions_applied', []),
                'new_columns_created': [],
                'flags_added': []
            }
        }
        
        # Check what new columns were created in cell data
        if 'RRU.PrbTotDl_abs' in cell_data.columns:
            summary['cell_data']['new_columns_created'].append('RRU.PrbTotDl_abs')
        if 'RRU.PrbTotUl_abs' in cell_data.columns:
            summary['cell_data']['new_columns_created'].append('RRU.PrbTotUl_abs')
        if 'PEE.Energy_interval' in cell_data.columns:
            summary['cell_data']['new_columns_created'].append('PEE.Energy_interval')
        if 'PEE.Energy_expected' in cell_data.columns:
            summary['cell_data']['new_columns_created'].append('PEE.Energy_expected')
        
        # Check flags
        if 'prb_conversion_applied' in cell_data.columns:
            summary['cell_data']['flags_added'].append('prb_conversion_applied')
        if 'energy_conversion_applied' in cell_data.columns:
            summary['cell_data']['flags_added'].append('energy_conversion_applied')
        if 'qos_flow_1s_semantics' in cell_data.columns:
            summary['cell_data']['flags_added'].append('qos_flow_1s_semantics')
        
        return summary