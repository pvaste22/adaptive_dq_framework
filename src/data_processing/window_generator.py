"""
Module: Data Processing
Phase: 1
Author: Pranjal V
Created: 06/09/2025
Purpose: Window generator for 5-minute data windows
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Generator, Tuple
from pathlib import Path
import sys
import os
import json
from datetime import datetime 


project_root = Path(__file__).parent.parent.parent 
src_path = project_root / 'src'
sys.path.insert(0, str(src_path))

from common.logger import get_phase1_logger
from common.constants import (
    WINDOW_SPECS, EXPECTED_ENTITIES, COLUMN_NAMES, MEAS_INTERVAL_SEC
)
from common.exceptions import WindowGenerationError

#this file will need to be modified when deploying on kafka stream to generate windows for deployment data
class WindowGenerator:
    """Creates 5-minute sliding windows from VIAVI time-series data."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = get_phase1_logger('window_generator')
        self.window_specs = WINDOW_SPECS
        self.expected_entities = EXPECTED_ENTITIES
        
    def generate_windows(self, cell_data: pd.DataFrame, ue_data: pd.DataFrame) -> List[Dict]:
        """Generate 5-minute windows from cell and UE data."""
        
        self.logger.info("Generating 5-minute sliding windows...")
        
        # Prepare data
        cell_data = cell_data.sort_values('timestamp')
        ue_data = ue_data.sort_values('timestamp')
        
        # Get time boundaries
        start_time = max(cell_data['timestamp'].min(), ue_data['timestamp'].min())
        end_time = min(cell_data['timestamp'].max(), ue_data['timestamp'].max())
        
        # Calculate window parameters
        window_size = timedelta(minutes=self.window_specs['size_minutes'])
        overlap_minutes = self.window_specs['size_minutes'] * (self.window_specs['overlap_percent'] / 100.0)
        step_size = timedelta(minutes=self.window_specs['size_minutes'] - overlap_minutes)
        
        windows = []
        current_time = start_time
        window_id = 0
        
        while current_time + window_size <= end_time:
            window_end = current_time + window_size
            
            # Extract window data
            window_cell = self._extract_window_data(cell_data, current_time, window_end)
            window_ue = self._extract_window_data(ue_data, current_time, window_end)
            
            # Validate window
            validation_result = self._validate_window_basic(window_cell, window_ue)
            
            if validation_result['valid']:
                window = {
                    'window_id': f"window_{window_id:06d}_{current_time.strftime('%Y%m%d_%H%M%S')}",
                    'start_time': current_time,
                    'end_time': window_end,
                    'cell_data': window_cell,
                    'ue_data': window_ue,
                    'metadata': self._create_window_metadata(window_cell, window_ue, window_id)
                }
                windows.append(window)
                window_id += 1
            else:
                self.logger.debug(f"Skipping window at {current_time}: {validation_result['reason']}")
            
            current_time += step_size
        
        self.logger.info(f"Generated {len(windows)} valid windows from {window_id} attempts")
        return windows
    
    def _extract_window_data(self, data: pd.DataFrame, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Extract data for a specific time window."""
        mask = (data['timestamp'] >= start_time) & (data['timestamp'] < end_time)
        return data[mask].copy()
    
    def _validate_window_basic(self, window_cell: pd.DataFrame, window_ue: pd.DataFrame) -> Dict:
        """Basic window validation for Phase 1."""
        
        # Check minimum completeness
        cell_count = len(window_cell)
        ue_count = len(window_ue)
        total_count = cell_count + ue_count
        
        expected_total = self.window_specs['expected_records']['total_per_window']
        completeness = total_count / expected_total if expected_total > 0 else 0
        
        min_completeness = self.config.get('min_window_completeness', 0.8)
        
        if completeness < min_completeness:
            return {
                'valid': False,
                'reason': f"Low completeness: {completeness:.2f} < {min_completeness}",
                'completeness': completeness
            }
        
        # Check timestamp consistency (should have 5 unique timestamps)
        expected_timestamps = self.window_specs['size_minutes']
        
        if len(window_cell) > 0:
            cell_timestamps = window_cell['timestamp'].nunique()
            if cell_timestamps < expected_timestamps * 0.8:  # Allow some tolerance
                return {
                    'valid': False,
                    'reason': f"Insufficient cell timestamps: {cell_timestamps} < {expected_timestamps}",
                    'completeness': completeness
                }
        
        if len(window_ue) > 0:
            ue_timestamps = window_ue['timestamp'].nunique()
            if ue_timestamps < expected_timestamps * 0.8:  # Allow some tolerance
                return {
                    'valid': False,
                    'reason': f"Insufficient UE timestamps: {ue_timestamps} < {expected_timestamps}",
                    'completeness': completeness
                }
        
        return {
            'valid': True,
            'reason': "Valid window",
            'completeness': completeness
        }
    
    def _create_window_metadata(self, window_cell: pd.DataFrame, window_ue: pd.DataFrame, window_id: int) -> Dict:
        """Create metadata for the window."""
        
        cell_entity_col = COLUMN_NAMES['cell_entity']
        ue_entity_col = COLUMN_NAMES['ue_entity']
        
        metadata = {
            'window_id': window_id,
            'generation_time': datetime.now().isoformat(),
            'record_counts': {
                'cells': len(window_cell),
                'ues': len(window_ue),
                'total': len(window_cell) + len(window_ue)
            },
            'entity_counts': {
                'unique_cells': window_cell[cell_entity_col].nunique() if len(window_cell) > 0 and cell_entity_col in window_cell.columns else 0,
                'unique_ues': window_ue[ue_entity_col].nunique() if len(window_ue) > 0 and ue_entity_col in window_ue.columns else 0
            },
            'completeness': {
                'cell_completeness': len(window_cell) / self.window_specs['expected_records']['cells_per_window'],
                'ue_completeness': len(window_ue) / self.window_specs['expected_records']['ues_per_window'],
                'total_completeness': (len(window_cell) + len(window_ue)) / self.window_specs['expected_records']['total_per_window']
            },
            'timestamp_range': {
                'cell_start': window_cell['timestamp'].min().isoformat() if len(window_cell) > 0 else None,
                'cell_end': window_cell['timestamp'].max().isoformat() if len(window_cell) > 0 else None,
                'ue_start': window_ue['timestamp'].min().isoformat() if len(window_ue) > 0 else None,
                'ue_end': window_ue['timestamp'].max().isoformat() if len(window_ue) > 0 else None
            }
        }
        
        return metadata
    
    def save_windows(self, windows: List[Dict], output_path: Path):
        """Save windows to disk in organized structure."""
        
        self.logger.info(f"Saving {len(windows)} windows to {output_path}")
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        for window in windows:
            window_dir = output_path / window['window_id']
            window_dir.mkdir(exist_ok=True)
            
            # Save data
            if len(window['cell_data']) > 0:
                window['cell_data'].to_parquet(window_dir / 'cell_data.parquet', compression='snappy')
            
            if len(window['ue_data']) > 0:
                window['ue_data'].to_parquet(window_dir / 'ue_data.parquet', compression='snappy')
            
            # Save metadata
            import json
            metadata_to_save = {
                'window_id': window['window_id'],
                'start_time': window['start_time'].isoformat(),
                'end_time': window['end_time'].isoformat(),
                'metadata': window['metadata']
            }
            
            with open(window_dir / 'metadata.json', 'w') as f:
                json.dump(metadata_to_save, f, indent=2, default=str)
        
        # Create summary file
        summary = {
            'total_windows': len(windows),
            'generation_time': datetime.now().isoformat(),
            'time_span': {
                'start': min(w['start_time'] for w in windows).isoformat() if windows else None,
                'end': max(w['end_time'] for w in windows).isoformat() if windows else None
            },
            'completeness_stats': {
                'mean': np.mean([w['metadata']['completeness']['total_completeness'] for w in windows]),
                'min': np.min([w['metadata']['completeness']['total_completeness'] for w in windows]),
                'max': np.max([w['metadata']['completeness']['total_completeness'] for w in windows])
            } if windows else None
        }
        
        with open(output_path / 'windows_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"Windows saved successfully to {output_path}")
    
    def get_window_statistics(self, windows: List[Dict]) -> Dict:
        """Calculate statistics across all windows."""
        
        if not windows:
            return {'error': 'No windows provided'}
        
        stats = {
            'total_windows': len(windows),
            'time_span': {
                'start': min(w['start_time'] for w in windows),
                'end': max(w['end_time'] for w in windows),
                'duration_hours': (max(w['end_time'] for w in windows) - min(w['start_time'] for w in windows)).total_seconds() / 3600
            },
            'completeness_stats': {
                'mean': np.mean([w['metadata']['completeness']['total_completeness'] for w in windows]),
                'std': np.std([w['metadata']['completeness']['total_completeness'] for w in windows]),
                'min': np.min([w['metadata']['completeness']['total_completeness'] for w in windows]),
                'max': np.max([w['metadata']['completeness']['total_completeness'] for w in windows])
            },
            'record_stats': {
                'mean_total_records': np.mean([w['metadata']['record_counts']['total'] for w in windows]),
                'mean_cell_records': np.mean([w['metadata']['record_counts']['cells'] for w in windows]),
                'mean_ue_records': np.mean([w['metadata']['record_counts']['ues'] for w in windows])
            }
        }
        
        return stats