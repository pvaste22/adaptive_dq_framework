"""
Base class for all quality dimension calculators.
Provides interface and common functionality for dimension scoring.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import sys
import yaml


# Import from common modules
from common.constants import (
    COLUMN_NAMES,
    PATHS
)
from common.utils import load_artifact
from common.logger import get_phase2_logger


class BaseDimension(ABC):
    """
    Abstract base class for quality dimension calculators.
    Each dimension inherits from this class and implements calculate_score method.
    """
    
    def __init__(self, name: str, config: Optional[Path] = None, baselines: Optional[Dict] = None):
        """
        Initialize base dimension calculator.
        
        Args:
            name: Dimension name (e.g., 'completeness', 'accuracy')
            config: Optional configuration dictionary
            baselines: Pre-loaded baselines dictionary (optional)
        """
        self.name = name
        self.config = config or {}
        self.baselines = baselines or {}
        
        # Setup logger for this dimension
        self.logger = get_phase2_logger(f'dimension_{name}')
        self.logger.info(f"Initializing {name} dimension calculator")
        
        # Load column names from constants
        self.timestamp_col = COLUMN_NAMES['timestamp']
        self.cell_entity_col = COLUMN_NAMES['cell_entity']
        self.ue_entity_col = COLUMN_NAMES['ue_entity']
        
        # Load dimension-specific configuration
        #self.dimension_thresholds = self._load_dimension_thresholds(name, config)
        
        # Initialize score history for tracking
        self.score_history = []
        
        self.logger.debug(f"Configuration loaded for {name} dimension")

  

    def get_dq_baseline(self):
        # cache
        if hasattr(self, '_dq_baseline_cache') and self._dq_baseline_cache is not None:
            return self._dq_baseline_cache

        b = {}
        try:
            if hasattr(self, 'load_artifact_baseline'):
                b = self.load_artifact_baseline('dq_baseline') or {}
        except Exception:
            b = {}

        if not b:
            # fallback for old runs: inside metadata_config
            try:
                meta = self.load_artifact_baseline('metadata_config') or {}
                b = meta.get('dq_baseline', {}) or {}
            except Exception:
                b = {}

        self._dq_baseline_cache = b
        return b
    

    def load_artifact_baseline(self, artifact_name: str) -> Optional[Dict]:
        """
        Load baseline artifact using common utils.

        Args:
            artifact_name: Name of artifact (e.g., 'temporal_templates')

        Returns:
            Loaded artifact or None if not found
        """
        try:
            # Find latest artifact file
            artifact_path = Path(PATHS['artifacts']) / artifact_name

            if not artifact_path.exists():
                self.logger.warning(f"Artifact directory not found: {artifact_path}")
                return None

            # Get most recent file
            pkl_files = list(artifact_path.glob('*.pkl'))
            json_files = list(artifact_path.glob('*.json'))
            all_files = pkl_files + json_files

            if not all_files:
                self.logger.warning(f"No artifact files found in {artifact_path}")
                return None

            latest_file = max(all_files, key=lambda p: p.stat().st_mtime)
            self.logger.debug(f"Loading artifact from {latest_file}")

            return load_artifact(latest_file)

        except Exception as e:
            self.logger.error(f"Error loading artifact {artifact_name}: {e}")
            return None


    def _apr_mpr(self, check_series_list, check_tuples_list=None):
        """
        Mixed-input APR/MPR calculator.
        Args:
            check_series_list: List[pd.Series] - row-wise boolean checks (for APR + MPR)
            check_tuples_list: List[Tuple[int, int]] - (pass, total) aggregate checks (for MPR only)
    
        Returns:
            apr, mpr, coverage, fail_dict
        """
        mat = []
        for s in check_series_list or []:
            if s is None or len(s) == 0:
                continue
            a = pd.Series(s).astype('float')  # True=1.0, False=0.0, NaN=NA
            mat.append(a)
        if mat:
            lengths = [len(s) for s in mat]
            if len(set(lengths)) > 1:
                self.logger.warning(f"Series length mismatch: {lengths}. Truncating to minimum.")
                min_len = min(lengths)
                mat = [s.iloc[:min_len].reset_index(drop=True) for s in mat]
        
            M_rows = pd.concat(mat, axis=1, ignore_index=True)
            row_passed = (M_rows == 1.0)
            row_all_passed = (M_rows.isna() | (M_rows == 1.0)).all(axis=1) # Strict AND per row
            apr = float(row_all_passed.mean()) if len(row_all_passed) > 0 else 0.0
        
            # MPR contribution from row-wise
            row_applicable = M_rows.notna().sum().sum()
            row_pass_count = row_passed.sum().sum()
        else:
            apr = 0.0
            row_applicable = 0
            row_pass_count = 0
    
        # aggregate check (for MPR only) 
        agg_applicable = 0
        agg_pass_count = 0
    
        for pass_count, total_count in (check_tuples_list or []):
            if total_count is not None and total_count > 0:
                agg_applicable += total_count
                agg_pass_count += pass_count if pass_count is not None else 0
    
        # combined MPR
        total_applicable = row_applicable + agg_applicable
        total_passed = row_pass_count + agg_pass_count
    
        mpr = float(total_passed / total_applicable) if total_applicable > 0 else 0.0
    
        # Coverage: fraction of checks that were applicable
        total_checks = len(check_series_list or []) + len(check_tuples_list or [])
        applicable_checks = len([s for s in (check_series_list or []) if len(s) > 0])
        applicable_checks += len([t for t in (check_tuples_list or []) if t[1] is not None and t[1] > 0])
        coverage = float(applicable_checks / total_checks) if total_checks > 0 else 0.0
    
        # Fail counts
        fail_dict = {
            'row_wise_fails': int(row_applicable - row_pass_count),
            'aggregate_fails': int(agg_applicable - agg_pass_count),
            'total_fails': int(total_applicable - total_passed)
        }
    
        return apr, mpr, coverage, fail_dict


    def calculate_check_score(self, value: float, pass_threshold: float, 
                          soft_threshold: float, higher_is_better: bool = True) -> float:
        """
        Calculate PASS/SOFT/FAIL score based on thresholds.

        Args:
            value: Actual value to check
            pass_threshold: Threshold for PASS
            soft_threshold: Threshold for SOFT
            higher_is_better: If True, higher values are better

        Returns:
            float: Score between 0-1 
        """
        if value is None or np.isnan(value):
            return None

        if higher_is_better:
            if value >= pass_threshold:
                return 1.0
            elif value >= soft_threshold:
                return 0.5
            else:
                return 0.0
        else:
            if value <= pass_threshold:
                return 1.0
            elif value <= soft_threshold:
                return 0.5
            else:
                return 0.0

    @abstractmethod
    def calculate_score(self, window_data: Dict, baselines: Optional[Dict] = None) -> Dict:
        """
        Calculate quality score for this dimension.
        Must be implemented by each dimension subclass.
        
        Args:
            window_data: Dictionary containing:
                - 'cell_data': DataFrame with cell records
                - 'ue_data': DataFrame with UE records  
                - 'metadata': Dict with window information
            baselines: Optional override for baselines
        
        Returns:
            Dictionary with:
                - 'score': Float between 0-1 (1 is perfect)
                - 'coverage': Float between 0-1 (data coverage)
                - 'details': Dict with measurement details
        """
        pass
    
    def validate_window_data(self, window_data: Dict) -> Tuple[bool, str]:
        """
        Validate that window data has required structure.
        
        Args:
            window_data: Window data to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check if window_data is a dictionary
            if not isinstance(window_data, dict):
                error_msg = "Window data must be a dictionary"
                self.logger.error(error_msg)
                return False, error_msg
            
            # Check for required keys
            required_keys = ['cell_data', 'ue_data', 'metadata']
            missing_keys = [k for k in required_keys if k not in window_data]
            if missing_keys:
                error_msg = f"Missing required keys: {missing_keys}"
                self.logger.error(error_msg)
                return False, error_msg
            
            # Check data types
            if not isinstance(window_data['cell_data'], pd.DataFrame):
                error_msg = "cell_data must be a DataFrame"
                self.logger.error(error_msg)
                return False, error_msg
            
            if not isinstance(window_data['ue_data'], pd.DataFrame):
                error_msg = "ue_data must be a DataFrame"
                self.logger.error(error_msg)
                return False, error_msg
            
            if not isinstance(window_data['metadata'], dict):
                error_msg = "metadata must be a dictionary"
                self.logger.error(error_msg)
                return False, error_msg
            
            self.logger.debug(f"Window validation successful")
            return True, ""
            
        except Exception as e:
            error_msg = f"Validation error: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return False, error_msg
    
    def load_window_from_disk(self, window_path: Path) -> Dict:
        """
        Load a window from disk storage.
        
        Args:
            window_path: Path to window directory
            
        Returns:
            Window data dictionary with cell_data, ue_data, and metadata
        """
        window_data = {}
        
        try:
            self.logger.debug(f"Loading window from: {window_path}")
            
            # Load cell data
            cell_file = window_path / 'cell_data.parquet'
            if cell_file.exists():
                window_data['cell_data'] = pd.read_parquet(cell_file)
                self.logger.debug(f"Loaded {len(window_data['cell_data'])} cell records")
            else:
                self.logger.warning(f"Cell data file not found: {cell_file}")
                window_data['cell_data'] = pd.DataFrame()
            
            # Load UE data
            ue_file = window_path / 'ue_data.parquet'
            if ue_file.exists():
                window_data['ue_data'] = pd.read_parquet(ue_file)
                self.logger.debug(f"Loaded {len(window_data['ue_data'])} UE records")
            else:
                self.logger.warning(f"UE data file not found: {ue_file}")
                window_data['ue_data'] = pd.DataFrame()
            
            # Load metadata
            metadata_file = window_path / 'metadata.json'
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    window_data['metadata'] = json.load(f)
                self.logger.debug(f"Loaded metadata")
            else:
                self.logger.warning(f"Metadata file not found: {metadata_file}")
                window_data['metadata'] = {}
            
            return window_data
            
        except Exception as e:
            self.logger.error(f"Error loading window: {str(e)}", exc_info=True)
            return {
                'cell_data': pd.DataFrame(),
                'ue_data': pd.DataFrame(),
                'metadata': {'error': str(e)}
            }
    
    def format_result(self, score: float, details: Dict) -> Dict:
        """
        Format the dimension result in standard structure.
        
        Args:
            score: Overall dimension score (0-1)
            details: Detailed measurement results
            
        Returns:
            Formatted result dictionary
        """
        result = {
            'dimension': self.name,
            'score': float(score),
            'timestamp': datetime.now().isoformat(),
            'details': details
        }
        
        # Track score in history
        self.score_history.append({
            'score': float(score),
            'timestamp': result['timestamp']
        })
        
        self.logger.info(f"{self.name} dimension score: {score:.4f}")
        
        return result
    
    def get_score_statistics(self) -> Dict:
        """
        Get statistics from score history.
        
        Returns:
            Dictionary with score statistics
        """
        if not self.score_history:
            self.logger.debug("No scores in history yet")
            return {'message': 'No scores in history'}
        
        scores = [s['score'] for s in self.score_history]
        
        stats = {
            'count': len(scores),
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'min': float(np.min(scores)),
            'max': float(np.max(scores)),
            'last_score': scores[-1]
        }
        
        self.logger.debug(f"Score statistics: {stats}")
        return stats