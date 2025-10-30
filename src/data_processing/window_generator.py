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
from typing import List, Dict, Generator, Tuple, Optional
from pathlib import Path
import json

from common.logger import get_phase1_logger
from common.constants import (
    WINDOW_SPECS, EXPECTED_ENTITIES, COLUMN_NAMES, MEAS_INTERVAL_SEC
)
from common.exceptions import WindowGenerationError

class WindowGenerator:
    """Creates 5-minute sliding windows from VIAVI time-series data."""
    
    def __init__(self):
        """Initialize WindowGenerator using constants directly."""
        self.logger = get_phase1_logger('window_generator')
        self.window_specs = WINDOW_SPECS
        self.expected_entities = EXPECTED_ENTITIES
        self.column_names = COLUMN_NAMES
        
    def generate_windows(self, cell_data: pd.DataFrame, ue_data: pd.DataFrame) -> List[Dict]:
        """Generate 5-minute windows from cell and UE data."""
        
        self.logger.info(f"Generating {self.window_specs['size_minutes']}-minute sliding windows...")
        
        # Validate input data
        if cell_data.empty and ue_data.empty:
            raise WindowGenerationError("Both cell and UE data are empty")
        
        # Prepare data
        cell_data = cell_data.sort_values('timestamp')
        ue_data = ue_data.sort_values('timestamp')
        
        for d in (cell_data, ue_data):
            if not pd.api.types.is_datetime64_any_dtype(d["timestamp"]):
                d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True, errors="coerce")
            d.dropna(subset=["timestamp"], inplace=True)

        # Get time boundaries
        start_time = max(
            cell_data['timestamp'].min() if not cell_data.empty else pd.Timestamp.max,
            ue_data['timestamp'].min() if not ue_data.empty else pd.Timestamp.max
        )
        end_time = min(
            cell_data['timestamp'].max() if not cell_data.empty else pd.Timestamp.min,
            ue_data['timestamp'].max() if not ue_data.empty else pd.Timestamp.min
        )
        
        if start_time >= end_time:
            raise WindowGenerationError("No overlapping time range between cell and UE data")
        
        # Calculate window parameters
        window_size = timedelta(minutes=self.window_specs['size_minutes'])
        overlap_minutes = self.window_specs['size_minutes'] * (self.window_specs['overlap_percent'] / 100.0)
        step_size = timedelta(minutes=self.window_specs['size_minutes'] - overlap_minutes)
        
        self.logger.info(f"Window parameters: size={window_size}, step={step_size}, overlap={overlap_minutes} minutes")
        
        windows = []
        current_time = start_time
        window_id = 0
        skipped_windows = 0
        
        while current_time + window_size <= end_time:
            window_end = current_time + window_size
            
            # Extract window data
            window_cell = self._extract_window_data(cell_data, current_time, window_end)
            window_ue = self._extract_window_data(ue_data, current_time, window_end)
            
            # Validate window
            validation_result = self._validate_window(window_cell, window_ue)
            
            window = {
                'window_id': f"window_{window_id:06d}_{current_time.strftime('%Y%m%d_%H%M%S')}",
                'start_time': current_time,
                'end_time': window_end,
                'cell_data': window_cell,
                'ue_data': window_ue,
                'metadata': self._create_window_metadata(
                    window_cell, window_ue, 
                    current_time, window_end,
                    validation_result['completeness']
                    )
            }
            windows.append(window)
            window_id += 1
            
            current_time += step_size
        
        self.logger.info(f"Generated {len(windows)} valid windows, skipped {skipped_windows} incomplete windows")
        
        if len(windows) == 0:
            self.logger.warning("No valid windows generated - check data completeness")
        
        return windows
    
    def _extract_window_data(self, data: pd.DataFrame, 
                            start_time: datetime, 
                            end_time: datetime) -> pd.DataFrame:
        """Extract data for a specific time window."""
        if data.empty:
            return pd.DataFrame()
        
        mask = (data['timestamp'] >= start_time) & (data['timestamp'] < end_time)
        return data[mask].copy()
    
    def _validate_window(self, window_cell: pd.DataFrame, 
                        window_ue: pd.DataFrame) -> Dict:
        """Validate window completeness and quality."""
        
        # Calculate completeness
        cell_count = len(window_cell)
        ue_count = len(window_ue)
        total_count = cell_count + ue_count
        
        expected_total = self.window_specs['expected_records']['total_per_window']
        completeness = total_count / expected_total if expected_total > 0 else 0
           
        # Check timestamp consistency
        expected_timestamps = self.window_specs['size_minutes']
        min_required_timestamps = int(expected_timestamps * 0.8)  # Allow 20% missing
        
        if not window_cell.empty:
            cell_timestamps = window_cell['timestamp'].nunique()
            if cell_timestamps < min_required_timestamps:
                return {
                    'valid': False,
                    'reason': f"Insufficient cell timestamps: {cell_timestamps} < {min_required_timestamps}",
                    'completeness': completeness
                }
        
        if not window_ue.empty:
            ue_timestamps = window_ue['timestamp'].nunique()
            if ue_timestamps < min_required_timestamps:
                return {
                    'valid': False,
                    'reason': f"Insufficient UE timestamps: {ue_timestamps} < {min_required_timestamps}",
                    'completeness': completeness
                }
        
        # Check for minimum data
        if cell_count == 0 and ue_count == 0:
            return {
                'valid': False,
                'reason': "Empty window - no data",
                'completeness': 0
            }
        
        return {
            'valid': True,
            'reason': "Valid window",
            'completeness': completeness
        }
    
    def _create_window_metadata(self, window_cell: pd.DataFrame, 
                               window_ue: pd.DataFrame,
                               start_time: datetime,
                               end_time: datetime,
                               completeness: float) -> Dict:
        """Create metadata for the window."""
        
        cell_entity_col = self.column_names.get('cell_entity')
        ue_entity_col = self.column_names.get('ue_entity')
        
        metadata = {
            'generation_time': datetime.now().isoformat(),
            'time_range': {
                'start': start_time.isoformat(),
                'end': end_time.isoformat(),
                'duration_minutes': self.window_specs['size_minutes']
            },
            'record_counts': {
                'cells': len(window_cell),
                'ues': len(window_ue),
                'total': len(window_cell) + len(window_ue)
            },
            'entity_counts': {
                'unique_cells': window_cell[cell_entity_col].nunique() if not window_cell.empty and cell_entity_col in window_cell.columns else 0,
                'unique_ues': window_ue[ue_entity_col].nunique() if not window_ue.empty and ue_entity_col in window_ue.columns else 0
            },
            'completeness': {
                'cell_completeness': len(window_cell) / self.window_specs['expected_records']['cells_per_window'],
                'ue_completeness': len(window_ue) / self.window_specs['expected_records']['ues_per_window'],
                'total_completeness': completeness
            },
            'actual_timestamps': {
                'cell_timestamps': window_cell['timestamp'].unique().tolist() if not window_cell.empty else [],
                'ue_timestamps': window_ue['timestamp'].unique().tolist() if not window_ue.empty else [],
                'cell_count': window_cell['timestamp'].nunique() if not window_cell.empty else 0,
                'ue_count': window_ue['timestamp'].nunique() if not window_ue.empty else 0
            }
        }
        
        return metadata
    

    
    def _save_window_summary(self, windows: List[Dict], output_path: Path):
        """Save summary statistics for all windows."""
        if not windows:
            return
        
        #completeness_values = [w['metadata']['completeness']['total_completeness'] for w in windows]
        
        summary = self.get_window_statistics(windows)
        
        with open(output_path / 'windows_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
    
    def get_window_statistics(self, windows: List[Dict]) -> Dict:
        """Calculate statistics across all windows."""
        
        if not windows:
            return {'error': 'No windows provided'}
        
        completeness_values = [w['metadata']['completeness']['total_completeness'] for w in windows]
        
        stats = {
            'window_count': len(windows),
            'time_span': {
                'start': min(w['start_time'] for w in windows),
                'end': max(w['end_time'] for w in windows),
                'total_hours': (max(w['end_time'] for w in windows) - min(w['start_time'] for w in windows)).total_seconds() / 3600
            },
            'completeness': {
                'mean': float(np.mean(completeness_values)),
                'std': float(np.std(completeness_values)),
                'min': float(np.min(completeness_values)),
                'max': float(np.max(completeness_values)),
                'median': float(np.median(completeness_values))
                #'above_threshold': sum(1 for c in completeness_values if c >= self.window_specs.get('min_completeness', 0.95))
            },
            'record_counts': {
                'mean': float(np.mean([w['metadata']['record_counts']['total'] for w in windows])),
                'std': float(np.std([w['metadata']['record_counts']['total'] for w in windows])),
                'total': sum(w['metadata']['record_counts']['total'] for w in windows)
            }
        }
        
        return stats

    def save_all_windows(self, windows: List[Dict], output_path: Path, batch_size: int = 50):
        """Save all windows to disk efficiently, processing in batches to manage memory."""
    
        if not windows:
            self.logger.warning("No windows to save")
            return
    
        self.logger.info(f"Saving {len(windows)} windows to {output_path}")
        output_path.mkdir(parents=True, exist_ok=True)
    
        # Create window index for quick lookup
        window_index = []
    
        # Process in batches to avoid memory issues
        for batch_start in range(0, len(windows), batch_size):
            batch_end = min(batch_start + batch_size, len(windows))
            batch = windows[batch_start:batch_end]
        
            for window in batch:
                window_dir = output_path / window['window_id']
                window_dir.mkdir(exist_ok=True)
            
                # Save data files
                if not window['cell_data'].empty:
                    window['cell_data'].to_parquet(window_dir / 'cell_data.parquet', compression='snappy')
                if not window['ue_data'].empty:
                    window['ue_data'].to_parquet(window_dir / 'ue_data.parquet', compression='snappy')
            
                # Save metadata
                metadata = {
                    'window_id': window['window_id'],
                    'start_time': window['start_time'].isoformat(),
                    'end_time': window['end_time'].isoformat(),
                    'metadata': window['metadata']
                }
                with open(window_dir / 'metadata.json', 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
            
                # Add to index
                window_index.append({
                    'window_id': window['window_id'],
                    'path': str(window_dir),
                    'start_time': window['start_time'].isoformat(),
                    'completeness': window['metadata']['completeness']['total_completeness']
                })
            
                # Clear window data from memory after saving
                window['cell_data'] = None
                window['ue_data'] = None
        
            self.logger.info(f"Saved batch {batch_start//batch_size + 1}/{(len(windows)-1)//batch_size + 1}")
    
        # Save window index
        with open(output_path / 'window_index.json', 'w') as f:
            json.dump({'windows': window_index, 'total_count': len(windows)}, f, indent=2)
    
        self._save_window_summary(windows, output_path)
        self.logger.info(f"All windows saved successfully")    