
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
import logging

from .constants import VERSIONING, PATHS, WINDOW_SPECS
from .exceptions import ArtifactError, DataValidationError

logger = logging.getLogger(__name__)

def save_artifact(data: Any, 
                 artifact_type: str,
                 version: str = 'v1',
                 metadata: Optional[Dict] = None) -> str:
    """
    Save artifact with proper versioning and metadata.
    
    Args:
        data: Data to save
        artifact_type: Type of artifact (e.g., 'statistical_baselines')
        version: Version string
        metadata: Optional metadata to include
        
    Returns:
        Path to saved artifact
    """
    if not artifact_type:
        raise ArtifactError("Artifact type must be specified")
        
    timestamp = datetime.now().strftime(VERSIONING['timestamp_format'])
    
    # Determine file extension
    if isinstance(data, dict) and _is_json_serializable(data):
        ext = 'json'
    elif isinstance(data, pd.DataFrame):
        ext = 'parquet'
    elif isinstance(data, np.ndarray):
        ext = 'npy'
    elif isinstance(data, (dict, list)):
        ext = 'pkl'
    else:
        ext = 'pkl'

    
    # Create filename
    filename = VERSIONING['artifact_name_pattern'].format(
        type=artifact_type,
        version=version,
        timestamp=timestamp,
        ext=ext
    )
    
    # Create full path
    artifact_path = Path(PATHS['artifacts']) / artifact_type / filename
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    
 
    try:
        # Save based on type
        if ext == 'parquet':
            data.to_parquet(artifact_path, compression='snappy')
        elif ext == 'npy':
            np.save(artifact_path, data)
        elif ext == 'json':
            with open(artifact_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
        else:
            with open(artifact_path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Save metadata if provided
        if metadata:
            metadata_path = artifact_path.with_suffix('.metadata.json')
            metadata['saved_at'] = timestamp
            metadata['artifact_type'] = artifact_type
            metadata['version'] = version
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Saved artifact: {artifact_path}")
        return str(artifact_path)
        
    except Exception as e:
        raise ArtifactError(f"Failed to save artifact {artifact_type}: {e}")

def load_artifact(artifact_path: Union[str, Path]) -> Any:
    """
    Load artifact from path.
    
    Args:
        artifact_path: Path to artifact file
        
    Returns:
        Loaded artifact data
    """
    path = Path(artifact_path)
    
    if not path.exists():
        raise ArtifactError(f"Artifact not found: {path}")
    
    try:
        if path.suffix == '.parquet':
            return pd.read_parquet(path)
        elif path.suffix == '.npy':
            return np.load(path, allow_pickle=False)
        elif path.suffix == '.json':
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif path.suffix == '.pkl':
            with open(path, 'rb') as f:
                return pickle.load(f)
        elif path.suffix == '.csv':
            return pd.read_csv(path)
        else:
            # Try to load as pickle by default
            with open(path, 'rb') as f:
                return pickle.load(f)
    except Exception as e:
        raise ArtifactError(f"Failed to load artifact from {path}: {e}")


def extract_band_from_cell_name(cell_name: str) -> str:
    """
    Extract band information from VIAVI cell name.
    
    Args:
        cell_name: Cell name in format 'S1/B13/C1'
        
    Returns:
        Band name (e.g., 'B13')
    """
    if not cell_name or not isinstance(cell_name, str):
        logger.warning(f"Invalid cell name: {cell_name}")
        return 'UNKNOWN'
    
    try:
        # Validate format first
        if cell_name.count('/') != 2:
            logger.warning(f"Cell name format incorrect: {cell_name}")
            return 'UNKNOWN'
        
        parts = cell_name.split('/')
        if len(parts) >= 2 and parts[1]:
            return parts[1]  # Second part is band
    except (AttributeError, IndexError) as e:
        logger.warning(f"Failed to extract band from {cell_name}: {e}")
    
    return 'UNKNOWN'

def validate_window_completeness(cell_count: int, ue_count: int, 
                                expected_records: Optional[Dict] = None) -> Dict[str, float]:
    """
    Validate window completeness against expected counts.
    
    Args:
        cell_count: Number of cell records in window
        ue_count: Number of UE records in window
        expected_records: Optional override for expected records
        
    Returns:
        Completeness metrics
        
    Raises:
        DataValidationError: If counts are negative
    """
    if cell_count < 0 or ue_count < 0:
        raise DataValidationError("Record counts cannot be negative")
    
    # Use provided expected records or default from constants
    expected = expected_records or WINDOW_SPECS.get('expected_records', {})
    
    cells_expected = expected.get('cells_per_window', 260)
    ues_expected = expected.get('ues_per_window', 240)
    total_expected = expected.get('total_per_window', 500)
    
    # Avoid division by zero
    cell_completeness = cell_count / cells_expected if cells_expected > 0 else 0
    ue_completeness = ue_count / ues_expected if ues_expected > 0 else 0
    total_completeness = (cell_count + ue_count) / total_expected if total_expected > 0 else 0
    
    return {
        'cell_completeness': cell_completeness,
        'ue_completeness': ue_completeness,
        'total_completeness': total_completeness,
        'cell_records': cell_count,
        'ue_records': ue_count,
        'total_records': cell_count + ue_count
        #'meets_threshold': total_completeness >= WINDOW_SPECS.get('min_completeness', 0.95)
    }


def create_timestamp_range(start_time: datetime, 
                          end_time: datetime,
                          interval_seconds: int = 60) -> List[datetime]:
    """
    Create list of timestamps with specified interval.
    
    Args:
        start_time: Start timestamp
        end_time: End timestamp  
        interval_seconds: Interval between timestamps
        
    Returns:
        List of timestamps
        
    Raises:
        DataValidationError: If start_time >= end_time or interval <= 0
    """
    if start_time >= end_time:
        raise DataValidationError("Start time must be before end time")
    if interval_seconds <= 0:
        raise DataValidationError("Interval must be positive")
    
    timestamps = []
    current = start_time
    
    while current <= end_time:
        timestamps.append(current)
        current += pd.Timedelta(seconds=interval_seconds)
    
    return timestamps


def calculate_jsd(p: np.ndarray, q: np.ndarray, base: float = 2.0) -> float:
    """
    Calculate Jensen-Shannon Divergence between two probability distributions.
    
    Args:
        p: First probability distribution
        q: Second probability distribution
        base: Logarithm base (2 for bits, e for nats)
        
    Returns:
        JSD value
        
    Raises:
        DataValidationError: If distributions have different lengths or invalid values
    """
    # Convert to numpy arrays if needed
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    
    # Validate inputs
    if p.shape != q.shape:
        raise DataValidationError("Distributions must have the same shape")
    if len(p) == 0:
        raise DataValidationError("Distributions cannot be empty")
    if np.any(p < 0) or np.any(q < 0):
        raise DataValidationError("Distributions cannot have negative values")
    
    # Normalize to ensure they sum to 1
    p_sum = p.sum()
    q_sum = q.sum()
    
    if p_sum == 0 or q_sum == 0:
        raise DataValidationError("Distributions cannot be all zeros")
    
    p_norm = p / p_sum
    q_norm = q / q_sum
    
    # Calculate middle distribution
    m = (p_norm + q_norm) / 2
    
    # Use scipy if available for better numerical stability
    try:
        from scipy.stats import entropy
        # JSD = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
        jsd = 0.5 * entropy(p_norm, m, base=base) + 0.5 * entropy(q_norm, m, base=base)
    except ImportError:
        # Fallback to manual calculation
        epsilon = 1e-10
        log_base = np.log2 if base == 2 else np.log
        
        # Calculate KL divergences
        kl_pm = np.sum(p_norm * log_base((p_norm + epsilon) / (m + epsilon)))
        kl_qm = np.sum(q_norm * log_base((q_norm + epsilon) / (m + epsilon)))
        
        # Convert to selected base if needed
        if base != 2 and base != np.e:
            kl_pm = kl_pm / np.log(base)
            kl_qm = kl_qm / np.log(base)
        
        jsd = 0.5 * kl_pm + 0.5 * kl_qm
    
    return float(jsd)


def get_current_timestamp() -> str:
    """Get current timestamp in standard format."""
    return datetime.now().strftime(VERSIONING['timestamp_format'])

def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if needed.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    dir_path = Path(path)
    try:
        dir_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.warning(f"Could not create directory {dir_path}: {e}")
    return dir_path

# Helper function for JSON serialization check
def _is_json_serializable(data: Any) -> bool:
    """Check if data is JSON serializable."""
    try:
        json.dumps(data, default=str)
        return True
    except (TypeError, ValueError):
        return False