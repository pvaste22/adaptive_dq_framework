"""
Module: Data Processing
Phase: 1
Author: Pranjal V
Created: 06/09/2025
Purpose: Window generator for 5-minute data windows
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta

from ..common.logger import get_phase1_logger
from ..common.constants import (
    WINDOW_SPECS, EXPECTED_ENTITIES, COLUMN_NAMES,
    QUALITY_THRESHOLDS
)
from ..common.exceptions import WindowGenerationError
from ..common.utils import validate_window_completeness, save_artifact

class WindowGenerator:
    """Generates 5-minute sliding windows from O-RAN data."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = get_phase1_logger('window_generator')
        
        # Window parameters from config
        self.window_size_minutes = config.get('window_size_minutes', 5)
        self.overlap_percent = config.get('overlap_percent', 50)
        self.min_completeness = config.get('min_completeness', 0.95)
        
        # Calculate step size
        self.step_minutes = self.window_size_minutes * (1 - self.overlap_percent / 100)
        
        # Expected counts
        self.expected_counts = {
            'cells': EXPECTED_ENTITIES['cells'],
            'ues': EXPECTED_ENTITIES['ues'],
            'timestamps_per_window': self.window_size_minutes,
            'cell_records_per_window': WINDOW_SPECS['expected_records']['cells_per_window'],
            'ue_records_per_window': WINDOW_SPECS['expected_records']['ues_per_window'],
            'total_records_per_window': WINDOW_SPECS['expected_records']['total_per_window']
        }
        
    def generate_windows(self, 
                        cell_data: pd.DataFrame, 
                        ue_data: pd.DataFrame) -> List[Dict]:
        """Generate 5-minute windows from cell and UE data."""
        self.logger.info("Starting window generation...")
        
        # Validate input
        self._validate_input_data(cell_data, ue_data)
        
        # Get time boundaries
        start_time, end_time = self._get_time_boundaries(cell_data, ue_data)
        
        # Generate window time slots
        window_times = self._generate_window_time_slots(start_time, end_time)
        
        # Create windows
        windows = []
        for window_start, window_end in window_times:
            window = self._create_single_window(
                cell_data, ue_data, window_start, window_end
            )
            if window is not None:
                windows.append(window)
        
        self.logger.info(f"Generated {len(windows)} valid windows")
        
        # Save metadata if configured
        if self.config.get('save_metadata', True):
            self._save_window_metadata(windows)
        
        return windows
    
    def _validate_input_data(self, cell_data: pd.DataFrame, ue_data: pd.DataFrame):
        """Validate input data."""
        timestamp_col = COLUMN_NAMES['timestamp']
        
        if timestamp_col not in cell_data.columns or timestamp_col not in ue_data.columns:
            raise WindowGenerationError(f"Timestamp column {timestamp_col} not found")
        
        if len(cell_data) == 0 or len(ue_data) == 0:
            raise WindowGenerationError("Empty data provided")
        
        # Check entity columns
        cell_entity_col = COLUMN_NAMES['cell_entity']
        ue_entity_col = COLUMN_NAMES['ue_entity']
        
        if cell_entity_col not in cell_data.columns:
            raise WindowGenerationError(f"Cell entity column {cell_entity_col} not found")
        if ue_entity_col not in ue_data.columns:
            raise WindowGenerationError(f"UE entity column {ue_entity_col} not found")
    
    def _get_time_boundaries(self, 
                           cell_data: pd.DataFrame, 
                           ue_data: pd.DataFrame) -> Tuple[datetime, datetime]:
        """Get overlapping time boundaries."""
        timestamp_col = COLUMN_NAMES['timestamp']
        
        cell_start = cell_data[timestamp_col].min()
        cell_end = cell_data[timestamp_col].max()
        ue_start = ue_data[timestamp_col].min()
        ue_end = ue_data[timestamp_col].max()
        
        overlap_start = max(cell_start, ue_start)
        overlap_end = min(cell_end, ue_end)
        
        if overlap_start >= overlap_end:
            raise WindowGenerationError("No overlapping time period")
        
        return overlap_start, overlap_end
    
    def _generate_window_time_slots(self, 
                                  start_time: datetime, 
                                  end_time: datetime) -> List[Tuple[datetime, datetime]]:
        """Generate sliding window time slots."""
        window_size = timedelta(minutes=self.window_size_minutes)
        step_size = timedelta(minutes=self.step_minutes)
        
        windows = []
        current_start = start_time
        
        while current_start + window_size <= end_time:
            current_end = current_start + window_size
            windows.append((current_start, current_end))
            current_start += step_size
        
        return windows
    
    def _create_single_window(self, 
                            cell_data: pd.DataFrame,
                            ue_data: pd.DataFrame, 
                            window_start: datetime,
                            window_end: datetime) -> Optional[Dict]:
        """Create a single window."""
        timestamp_col = COLUMN_NAMES['timestamp']
        
        # Extract data for window
        cell_mask = (
            (cell_data[timestamp_col] >= window_start) & 
            (cell_data[timestamp_col] < window_end)
        )
        ue_mask = (
            (ue_data[timestamp_col] >= window_start) & 
            (ue_data[timestamp_col] < window_end)
        )
        
        window_cell_data = cell_data[cell_mask].copy()
        window_ue_data = ue_data[ue_mask].copy()
        
        # Validate completeness
        completeness = validate_window_completeness(
            len(window_cell_data), 
            len(window_ue_data)
        )
        
        if completeness['total_completeness'] < self.min_completeness:
            self.logger.debug(f"Window {window_start} below completeness threshold")
            return None
        
        # Create window
        window_id = f"window_{window_start.strftime('%Y%m%d_%H%M%S')}"
        
        return {
            'window_id': window_id,
            'start_time': window_start,
            'end_time': window_end,
            'cell_data': window_cell_data,
            'ue_data': window_ue_data,
            'metadata': {
                'completeness': completeness,
                'cell_count': len(window_cell_data),
                'ue_count': len(window_ue_data),
                'unique_cells': window_cell_data[COLUMN_NAMES['cell_entity']].nunique(),
                'unique_ues': window_ue_data[COLUMN_NAMES['ue_entity']].nunique()
            }
        }
    
    def _save_window_metadata(self, windows: List[Dict]):
        """Save window metadata."""
        if not windows:
            return
        
        metadata_list = []
        for window in windows:
            meta = {
                'window_id': window['window_id'],
                'start_time': window['start_time'],
                'end_time': window['end_time'],
                **window['metadata']
            }
            metadata_list.append(meta)
        
        metadata_df = pd.DataFrame(metadata_list)
        save_artifact(metadata_df, 'window_metadata', 'phase1', versioned=True)
        
        self.logger.info(f"Saved metadata for {len(windows)} windows")