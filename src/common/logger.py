"""
Module: Common
Phase: All
Author: Pranjal V
Created: 06/09/2025
Purpose: Centralized logging setup for all phases
"""
import logging
import logging.handlers
from pathlib import Path
from typing import Optional
from datetime import datetime

# Import from constants to use consistent paths
from .constants import PATHS

def setup_logger(name: str, 
                log_file: Optional[str] = None,
                level: int = logging.INFO,
                max_bytes: int = 10485760,  # 10MB
                backup_count: int = 5) -> logging.Logger:
    """
    Setup logger with file and console handlers.
    
    Args:
        name: Logger name (usually __name__)
        log_file: Optional log file path
        level: Logging level
        max_bytes: Max file size before rotation
        backup_count: Number of backup files to keep
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    # Console handler with standard format
    console_handler = logging.StreamHandler()
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    console_handler.setLevel(level)
    logger.addHandler(console_handler)
    
    # File handler with detailed format
    if log_file:
        log_path = Path(log_file)
        # Ensure directory exists
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            file_handler = logging.handlers.RotatingFileHandler(
                log_file, 
                maxBytes=max_bytes, 
                backupCount=backup_count,
                encoding='utf-8'
            )
            
            file_format = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_format)
            file_handler.setLevel(logging.DEBUG)  # More detailed for files
            logger.addHandler(file_handler)
        except (OSError, IOError) as e:
            logger.warning(f"Could not create file handler for {log_file}: {e}")
    
    return logger

def get_phase_logger(phase: str, component: str) -> logging.Logger:
    """
    Get logger for specific phase and component.
    
    Args:
        phase: Phase name (phase1, phase2, etc.)
        component: Component name (data_loader, drift_detector, etc.)
        
    Returns:
        Configured logger with appropriate file
    """
    logger_name = f"{phase}.{component}"
    
    # Use PATHS from constants for consistent directory structure
    log_dir = PATHS['logs'] / phase
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"{component}_{datetime.now().strftime('%Y%m%d')}.log"
    
    return setup_logger(
        name=logger_name,
        log_file=str(log_file),
        level=logging.INFO
    )

# Generic phase logger function to reduce duplication
def _create_phase_logger(phase_num: int):
    """Factory function to create phase-specific logger functions."""
    def phase_logger(component: str) -> logging.Logger:
        return get_phase_logger(f'phase{phase_num}', component)
    return phase_logger

# Create phase-specific logger functions
get_phase1_logger = _create_phase_logger(1)
get_phase2_logger = _create_phase_logger(2)
get_phase3_logger = _create_phase_logger(3)
get_phase4_logger = _create_phase_logger(4)

# Add docstrings for clarity
get_phase1_logger.__doc__ = "Get Phase 1 logger for data preparation components."
get_phase2_logger.__doc__ = "Get Phase 2 logger for quality scoring components."
get_phase3_logger.__doc__ = "Get Phase 3 logger for drift detection components."
get_phase4_logger.__doc__ = "Get Phase 4 logger for deployment components."

# Optional: Add log cleanup function
def cleanup_old_logs(days_to_keep: int = 30):
    """Remove log files older than specified days."""
    import time
    current_time = time.time()
    
    for phase_dir in PATHS['logs'].iterdir():
        if phase_dir.is_dir():
            for log_file in phase_dir.glob('*.log*'):
                if log_file.is_file():
                    file_age_days = (current_time - log_file.stat().st_mtime) / 86400
                    if file_age_days > days_to_keep:
                        try:
                            log_file.unlink()
                        except OSError:
                            pass  # Skip if file is in use