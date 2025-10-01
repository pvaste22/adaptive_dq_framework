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

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# Import from common modules
from common.constants import (
    COLUMN_NAMES,
    PATHS, QUALITY_DIMENSION_THRESHOLDS
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
        self.dimension_thresholds = self._load_dimension_thresholds(name, config)
        
        # Initialize score history for tracking
        self.score_history = []
        
        self.logger.debug(f"Configuration loaded for {name} dimension")


    def _load_dimension_thresholds(self,  dimension_name: str, config_path: Optional[Path] = None) -> Dict:
        """
        Load dimension-specific thresholds from config.yaml
    
        Args:
            dimension_name: Name of the dimension
            config_path: Not used anymore, kept for compatibility
        
        Returns:
            Dictionary of thresholds for this dimension
        """
        try:
            thresholds = QUALITY_DIMENSION_THRESHOLDS.get(dimension_name, {})
        
            if not thresholds:
                self.logger.warning(f"No thresholds found for {dimension_name}, using empty dict")
            else:
                self.logger.debug(f"Loaded {len(thresholds)} threshold groups for {dimension_name}")

            return thresholds
        
        except Exception as e:
            self.logger.error(f"Error loading thresholds: {e}")
            return {}    

    def get_baseline(self, baseline_name: str) -> Optional[Dict]:
        """
        Get a specific baseline from loaded baselines.

        Args:
            baseline_name: Name of baseline to retrieve

        Returns:
            Baseline data or None if not found
        """
        if baseline_name not in self.baselines:
            self.logger.warning(f"Baseline '{baseline_name}' not found")
            return None
        return self.baselines.get(baseline_name)
    

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