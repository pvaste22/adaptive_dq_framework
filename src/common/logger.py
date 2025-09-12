
import logging
import logging.handlers
from pathlib import Path
from typing import Optional
from datetime import datetime

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
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
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
    log_file = f"data/logs/{phase}/{component}_{datetime.now().strftime('%Y%m%d')}.log"
    
    return setup_logger(
        name=logger_name,
        log_file=log_file,
        level=logging.INFO
    )

# Convenience functions for each phase
def get_phase1_logger(component: str) -> logging.Logger:
    """Get Phase 1 logger for data preparation components."""
    return get_phase_logger('phase1', component)

def get_phase2_logger(component: str) -> logging.Logger:
    """Get Phase 2 logger for quality scoring components."""
    return get_phase_logger('phase2', component)

def get_phase3_logger(component: str) -> logging.Logger:
    """Get Phase 3 logger for drift detection components."""
    return get_phase_logger('phase3', component)

def get_phase4_logger(component: str) -> logging.Logger:
    """Get Phase 4 logger for deployment components."""
    return get_phase_logger('phase4', component)