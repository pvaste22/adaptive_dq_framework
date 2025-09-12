"""
Module: Data Processing
Phase: 1
Author: Pranjal V
Created: 06/09/2025
Purpose: Centralized configuration constants
"""
import json
import yaml
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Configuration paths
CONFIG_DIR = PROJECT_ROOT / 'config'
DATA_DIR = PROJECT_ROOT / 'data'

MAIN_CONFIG_PATH = CONFIG_DIR / 'config.yaml'
BAND_CONFIG_PATH = CONFIG_DIR / 'band_configs.json'
DRIFT_CONFIG_PATH = CONFIG_DIR / 'drift_thresholds.yaml'

# Load configurations
def _load_main_config():
    with open(MAIN_CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)

def _load_band_configs():
    with open(BAND_CONFIG_PATH, 'r') as f:
        config = json.load(f)
    return config['band_specifications']

def _load_drift_config():
    with open(DRIFT_CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)

MAIN_CONFIG = _load_main_config()
BAND_SPECS = _load_band_configs()
DRIFT_CONFIG = _load_drift_config()

# Dataset specifications
VIAVI_CONFIG = MAIN_CONFIG['data_sources']['viavi_dataset']
EXPECTED_ENTITIES = {
    'cells': VIAVI_CONFIG['expected_entities']['cells'],
    'ues': VIAVI_CONFIG['expected_entities']['ues']
}
MEAS_INTERVAL_SEC = VIAVI_CONFIG['measurement_interval_seconds']

# Column names
COLUMN_NAMES = VIAVI_CONFIG['column_names']

# Metric groups
METRIC_GROUPS = VIAVI_CONFIG['metric_groups']

# Data quirks
DATA_QUIRKS = VIAVI_CONFIG['data_quirks']

# Band limits
BAND_LIMITS = {
    band: {'prb_count': spec['prb_count'], 'name': spec['name']} 
    for band, spec in BAND_SPECS.items()
}

# Window specifications
WINDOW_CONFIG = MAIN_CONFIG['processing']['windowing']
WINDOW_SPECS = {
    'size_minutes': WINDOW_CONFIG['size_minutes'],
    'overlap_percent': WINDOW_CONFIG['overlap_percent'],
    'expected_records': {
        'cells_per_window': EXPECTED_ENTITIES['cells'] * WINDOW_CONFIG['size_minutes'],
        'ues_per_window': EXPECTED_ENTITIES['ues'] * WINDOW_CONFIG['size_minutes'],
        'total_per_window': (EXPECTED_ENTITIES['cells'] + EXPECTED_ENTITIES['ues']) * WINDOW_CONFIG['size_minutes']
    }
}

# Quality thresholds
QUALITY_THRESHOLDS = {
    'completeness_min_ratio': DRIFT_CONFIG['quality_thresholds']['completeness']['minimum_ratio'],
    'z_score_outlier_threshold': DRIFT_CONFIG['quality_thresholds']['accuracy']['z_score_threshold'],
    'spectral_efficiency_max': DRIFT_CONFIG['quality_thresholds']['accuracy']['spectral_efficiency_max'],
    'energy_validation_tolerance': DRIFT_CONFIG['quality_thresholds']['accuracy']['energy_validation_tolerance']
}

# Drift parameters
DRIFT_PARAMS = {
    'significance_threshold': DRIFT_CONFIG['drift_detection']['significance_threshold'],
    'min_history_size': DRIFT_CONFIG['drift_detection']['min_history_size'],
    'distribution_bins': DRIFT_CONFIG['drift_detection']['distribution_bins'],
    'pattern_tolerance': DRIFT_CONFIG['pattern_tolerances']
}

# Expected patterns
PATTERN_TOLERANCES = DRIFT_CONFIG['pattern_tolerances']
EXPECTED_PATTERNS = {
    'mimo_zero_rate': 0.86,
    'cqi_zero_rate': 0.60,
    'ul_dl_symmetry': 0.87,
    'mimo_tolerance': PATTERN_TOLERANCES['mimo_zero_rate'],
    'cqi_tolerance': PATTERN_TOLERANCES['cqi_zero_rate'],
    'symmetry_tolerance': PATTERN_TOLERANCES['ul_dl_symmetry']
}

# Paths
STORAGE_CONFIG = MAIN_CONFIG['storage']
PATHS = {
    'raw_data': DATA_DIR / 'raw',
    'processed_data': DATA_DIR / 'processed',
    'artifacts': DATA_DIR / 'artifacts',
    'logs': DATA_DIR / 'logs',
    'config': CONFIG_DIR
}

# Versioning
VERSIONING = {
    'initial_version': 'v0',
    'timestamp_format': '%Y%m%d_%H%M%S',
    'artifact_name_pattern': '{type}_v{version}_{timestamp}.{ext}',
    'max_versions_keep': STORAGE_CONFIG['artifacts']['max_versions_keep'],
    'compression_enabled': STORAGE_CONFIG['artifacts']['compression']
}

# Data types for pandas (code-specific, not config)
CELL_DTYPES = {
    'timestamp': 'int64',
    'Viavi.Cell.Name': 'category',
    'DRB.UEThpDl': 'float32',
    'DRB.UEThpUl': 'float32',
    'RRU.PrbUsedDl': 'float32',
    'RRU.PrbUsedUl': 'float32',
    'RRU.PrbAvailDl': 'uint16',
    'RRU.PrbAvailUl': 'uint16',
    'RRU.PrbTotDl': 'float32',
    'RRU.PrbTotUl': 'float32',
    'RRU.MaxLayerDlMimo': 'uint8',
    'CARR.AverageLayersDl': 'float32',
    'RRC.ConnMean': 'float32',
    'RRC.ConnMax': 'uint8',
    'QosFlow.TotPdcpPduVolumeDl': 'float64',
    'QosFlow.TotPdcpPduVolumeUl': 'float64',
    'PEE.AvgPower': 'float32',
    'PEE.Energy': 'float64'
}

UE_DTYPES = {
    'timestamp': 'int64',
    'Viavi.UE.Name': 'category',
    'DRB.UECqiDl': 'uint8',
    'DRB.UECqiUl': 'uint8',
    'DRB.UEThpDl': 'float32',
    'DRB.UEThpUl': 'float32',
    'RRU.PrbUsedDl': 'float32',
    'RRU.PrbUsedUl': 'float32',
    'TB.TotNbrDl': 'float32',
    'TB.TotNbrUl': 'float32'
}

# Regex patterns (code-specific)
PATTERNS = {
    'cell_name_parser': r'(S\d+)/([^/]+)/(C\d+)',
    'band_extractor': r'S\d+/([^/]+)/C\d+',
    'site_extractor': r'(S\d+)/[^/]+/C\d+',
    'cell_id_extractor': r'S\d+/[^/]+/(C\d+)'
}