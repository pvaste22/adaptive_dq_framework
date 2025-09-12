
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import pandas as pd
import numpy as np

from .constants import VERSIONING, PATHS

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
    timestamp = datetime.now().strftime(VERSIONING['timestamp_format'])
    
    # Determine file extension
    if isinstance(data, (dict, list)):
        ext = 'pkl'
    elif isinstance(data, pd.DataFrame):
        ext = 'parquet'
    elif isinstance(data, np.ndarray):
        ext = 'npy'
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
    
    # Save based on type
    if ext == 'parquet':
        data.to_parquet(artifact_path)
    elif ext == 'npy':
        np.save(artifact_path, data)
    else:
        with open(artifact_path, 'wb') as f:
            pickle.dump(data, f)
    
    # Save metadata if provided
    if metadata:
        metadata_path = artifact_path.with_suffix('.metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    return str(artifact_path)

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
        raise FileNotFoundError(f"Artifact not found: {path}")
    
    if path.suffix == '.parquet':
        return pd.read_parquet(path)
    elif path.suffix == '.npy':
        return np.load(path)
    elif path.suffix == '.pkl':
        with open(path, 'rb') as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unsupported artifact type: {path.suffix}")

def extract_band_from_cell_name(cell_name: str) -> str:
    """
    Extract band information from VIAVI cell name.
    
    Args:
        cell_name: Cell name in format 'S1/B13/C1'
        
    Returns:
        Band name (e.g., 'B13')
    """
    try:
        parts = cell_name.split('/')
        if len(parts) >= 2:
            return parts[1]  # Second part is band
    except (AttributeError, IndexError):
        pass
    
    return 'UNKNOWN'

def validate_window_completeness(cell_count: int, ue_count: int) -> Dict[str, float]:
    """
    Validate window completeness against expected counts.
    
    Args:
        cell_count: Number of cell records in window
        ue_count: Number of UE records in window
        
    Returns:
        Completeness metrics
    """
    from .constants import WINDOW_SPECS
    
    expected = WINDOW_SPECS['expected_records']
    
    return {
        'cell_completeness': cell_count / expected['cells_per_window'],
        'ue_completeness': ue_count / expected['ues_per_window'],
        'total_completeness': (cell_count + ue_count) / expected['total_per_window'],
        'cell_records': cell_count,
        'ue_records': ue_count,
        'total_records': cell_count + ue_count
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
    """
    timestamps = []
    current = start_time
    
    while current <= end_time:
        timestamps.append(current)
        current += pd.Timedelta(seconds=interval_seconds)
    
    return timestamps

def calculate_jsd(p: np.ndarray, q: np.ndarray) -> float:
    """
    Calculate Jensen-Shannon Divergence between two probability distributions.
    
    Args:
        p: First probability distribution
        q: Second probability distribution
        
    Returns:
        JSD value
    """
    # Ensure arrays are normalized
    p = p / p.sum()
    q = q / q.sum()
    
    # Calculate middle distribution
    m = (p + q) / 2
    
    # Calculate KL divergences (with small epsilon to avoid log(0))
    epsilon = 1e-10
    kl_pm = np.sum(p * np.log((p + epsilon) / (m + epsilon)))
    kl_qm = np.sum(q * np.log((q + epsilon) / (m + epsilon)))
    
    # Jensen-Shannon divergence
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
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path