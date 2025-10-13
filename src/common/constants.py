"""
Module: Data Processing
Phase: 1 
Author: Pranjal V
Created: 06/09/2025
Purpose: Centralized configuration constants - Single source of truth
"""
import json
import yaml
from pathlib import Path
import os
import sys

# Project root - more robust path handling
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Add src to Python path for imports
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

# Configuration paths
CONFIG_DIR = PROJECT_ROOT / 'config'
DATA_DIR = PROJECT_ROOT / 'data'

# Ensure critical directories exist
REQUIRED_DIRS = {
    'config': CONFIG_DIR,
    'raw_data': DATA_DIR / 'raw',
    'processed': DATA_DIR / 'processed',
    'artifacts': DATA_DIR / 'artifacts',
    'logs': DATA_DIR / 'logs',
    'reports': DATA_DIR / 'reports',
    'windows': DATA_DIR / 'processed' / 'windowed_data',
    'historical_windows': DATA_DIR / 'processed' / 'historical_windows',
    'sample_windows': DATA_DIR / 'processed' / 'sample_windows', 
    'training': DATA_DIR / 'processed' / 'training' / 'v0'
}

for name, dir_path in REQUIRED_DIRS.items():
    dir_path.mkdir(parents=True, exist_ok=True)

# Configuration file paths
MAIN_CONFIG_PATH = CONFIG_DIR / 'config.yaml'
BAND_CONFIG_PATH = CONFIG_DIR / 'band_configs.json'
DRIFT_CONFIG_PATH = CONFIG_DIR / 'drift_thresholds.yaml'
LOGGING_CONFIG_PATH = CONFIG_DIR / 'logging_config.yaml'

# Load configurations with error handling
def _load_config_safe(path, loader='yaml'):
    """Safely load configuration file with error handling."""
    if not path.exists():
        print(f"Warning: Config file not found: {path}")
        return {}
    
    try:
        with open(path, 'r') as f:
            if loader == 'yaml':
                return yaml.safe_load(f)
            elif loader == 'json':
                return json.load(f)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return {}

# Load all configurations
MAIN_CONFIG = _load_config_safe(MAIN_CONFIG_PATH, 'yaml')
BAND_CONFIG_FULL = _load_config_safe(BAND_CONFIG_PATH, 'json')
BAND_SPECS = BAND_CONFIG_FULL.get('band_specifications', {}) if BAND_CONFIG_FULL else {}
DRIFT_CONFIG = _load_config_safe(DRIFT_CONFIG_PATH, 'yaml')

# Dataset expectations from config
DATASET_EXPECTATIONS = MAIN_CONFIG.get('dataset_expectations', {})
PATTERN_TOLERANCES = DRIFT_CONFIG.get('pattern_tolerances',{})
# Expected patterns with tolerances
EXPECTED_PATTERNS = {
    'mimo_zero_rate': DATASET_EXPECTATIONS.get('patterns', {}).get('mimo_zero_rate', {}).get('expected', 0.86),
    'cqi_no_measurement_rate': DATASET_EXPECTATIONS.get('patterns', {}).get('cqi_no_measurement_rate', {}).get('expected', 0.60),
    'ul_dl_symmetry': DATASET_EXPECTATIONS.get('patterns', {}).get('ul_dl_symmetry', {}).get('expected', 0.87),
    # Keep existing tolerances from DRIFT_CONFIG
    'mimo_tolerance': PATTERN_TOLERANCES.get('mimo_zero_rate', 0.10),
    'cqi_tolerance': PATTERN_TOLERANCES.get('cqi_zero_rate', 0.10),
    'symmetry_tolerance': PATTERN_TOLERANCES.get('ul_dl_symmetry', 0.15)
}

# Expected patterns
#PATTERN_TOLERANCES = DRIFT_CONFIG.get('pattern_tolerances', {})
"""EXPECTED_PATTERNS = {
    'mimo_zero_rate': 0.86,
    'cqi_zero_rate': 0.60,
    'ul_dl_symmetry': 0.87,
    'mimo_tolerance': PATTERN_TOLERANCES.get('mimo_zero_rate', 0.10),
    'cqi_tolerance': PATTERN_TOLERANCES.get('cqi_zero_rate', 0.10),
    'symmetry_tolerance': PATTERN_TOLERANCES.get('ul_dl_symmetry', 0.15)
}"""

# Expected correlations
"""EXPECTED_CORRELATIONS = []
for corr_name, corr_data in DATASET_EXPECTATIONS.get('correlations', {}).items():
    if 'metrics' in corr_data and len(corr_data['metrics']) == 2:
        EXPECTED_CORRELATIONS.append((
            corr_data['metrics'][0],
            corr_data['metrics'][1],
            corr_data.get('expected', 0.0),
            corr_data.get('tolerance', 0.30)
        ))"""

UNRELIABLE_METRICS = MAIN_CONFIG.get('unreliable_metrics',['TB.TotNbrDl','TB.TotNbrUl'])
# Temporal coefficients
TEMPORAL_COEFFICIENTS = DATASET_EXPECTATIONS.get('temporal', {})

# Physical limits
PHYSICAL_LIMITS = DATASET_EXPECTATIONS.get('physical_limits', {})

# Dataset specifications from main config
VIAVI_CONFIG = MAIN_CONFIG.get('data_sources', {}).get('viavi_dataset', {})
EXPECTED_ENTITIES = {
    'cells': VIAVI_CONFIG.get('expected_entities', {}).get('cells', 52),
    'ues': VIAVI_CONFIG.get('expected_entities', {}).get('ues', 48)
}
MEAS_INTERVAL_SEC = VIAVI_CONFIG.get('measurement_interval_seconds', 60)

# File paths
DATA_FILES = {
    'cell_reports': DATA_DIR / 'raw' / 'CellReports_v0.csv',
    'ue_reports': DATA_DIR / 'raw' / 'UEReports_v0.csv'
}

# Column names
COLUMN_NAMES = VIAVI_CONFIG.get('column_names', {
    'timestamp': 'timestamp',
    'cell_entity': 'Viavi.Cell.Name',
    'ue_entity': 'Viavi.UE.Name',
    'avg_power': 'PEE.AvgPower'
})

# Metric groups
METRIC_GROUPS = VIAVI_CONFIG.get('metric_groups', {})

# Data quirks
DATA_QUIRKS = VIAVI_CONFIG.get('data_quirks', {
    'tb_counters_unreliable': True,
    'cqi_zero_is_no_measurement': True,
    'prb_tot_is_percentage': True,
    'energy_is_cumulative': True,
    'qos_flow_has_1s_semantics': True
})

# Band limits
BAND_LIMITS = {
    band: {'prb_count': spec.get('prb_count'), 'name': spec.get('name')} 
    for band, spec in BAND_SPECS.items()
}

# Window specifications
WINDOW_CONFIG = MAIN_CONFIG.get('processing', {}).get('windowing', {})
WINDOW_SPECS = {
    'size_minutes': WINDOW_CONFIG.get('size_minutes', 5),
    'overlap_percent': WINDOW_CONFIG.get('overlap_percent', 80),
    'min_completeness': WINDOW_CONFIG.get('minimum_completeness', 0.95),
    'expected_records': {
        'cells_per_window': EXPECTED_ENTITIES['cells'] * WINDOW_CONFIG.get('size_minutes', 5)* (60 / MEAS_INTERVAL_SEC),
        'ues_per_window': EXPECTED_ENTITIES['ues'] * WINDOW_CONFIG.get('size_minutes', 5)* (60 / MEAS_INTERVAL_SEC),
        'total_per_window': (EXPECTED_ENTITIES['cells'] + EXPECTED_ENTITIES['ues']) * WINDOW_CONFIG.get('size_minutes', 5)* (60 / MEAS_INTERVAL_SEC)
    }
}

# Unit conversion settings
CONVERSION_CONFIG = MAIN_CONFIG.get('processing', {}).get('unit_conversion', {
    'prb_percentage_to_absolute': True,
    'energy_cumulative_to_interval': True,
    'qos_flow_validation': True,
    'validation_tolerance': 0.20
})

# Quality thresholds
QUALITY_THRESHOLDS = DRIFT_CONFIG.get('quality_thresholds', {})
QUALITY_THRESHOLDS_FLAT = {
    'completeness_min_ratio': QUALITY_THRESHOLDS.get('completeness', {}).get('minimum_ratio', 0.95),
    'z_score_outlier_threshold': QUALITY_THRESHOLDS.get('accuracy', {}).get('z_score_threshold', 3.0),
    'spectral_efficiency_max': QUALITY_THRESHOLDS.get('accuracy', {}).get('spectral_efficiency_max', 30.0),
    'energy_validation_tolerance': QUALITY_THRESHOLDS.get('accuracy', {}).get('energy_validation_tolerance', 0.20)
}

# Drift parameters
DRIFT_PARAMS = {
    'significance_threshold': DRIFT_CONFIG.get('drift_detection', {}).get('significance_threshold', 0.08),
    'min_history_size': DRIFT_CONFIG.get('drift_detection', {}).get('min_history_size', 100),
    'distribution_bins': DRIFT_CONFIG.get('drift_detection', {}).get('distribution_bins', 50),
    'pattern_tolerance': DRIFT_CONFIG.get('pattern_tolerances', {})
}

# Paths
PATHS = REQUIRED_DIRS

# Versioning
STORAGE_CONFIG = MAIN_CONFIG.get('storage', {})
VERSIONING = {
    'initial_version': 'v0',
    'timestamp_format': '%Y%m%d_%H%M%S',
    'artifact_name_pattern': '{type}_v{version}_{timestamp}.{ext}',
    'max_versions_keep': STORAGE_CONFIG.get('artifacts', {}).get('max_versions_keep', 10),
    'compression_enabled': STORAGE_CONFIG.get('artifacts', {}).get('compression', True)
}

# Data types for pandas
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
    'RRU.MaxLayerDlMimo': 'float64',
    'CARR.AverageLayersDl': 'float32',
    'RRC.ConnMean': 'float32',
    'RRC.ConnMax': 'float64',
    'QosFlow.TotPdcpPduVolumeDl': 'float64',
    'QosFlow.TotPdcpPduVolumeUl': 'float64',
    'PEE.AvgPower': 'float32',
    'PEE.Energy': 'float64'
}

UE_DTYPES = {
    'timestamp': 'int64',
    'Viavi.UE.Name': 'category',
    'DRB.UECqiDl': 'float64',
    'DRB.UECqiUl': 'float64',
    'DRB.UEThpDl': 'float32',
    'DRB.UEThpUl': 'float32',
    'RRU.PrbUsedDl': 'float32',
    'RRU.PrbUsedUl': 'float32',
    'TB.TotNbrDl': 'float32',
    'TB.TotNbrUl': 'float32'
}

# Regex patterns
PATTERNS = {
    'cell_name_parser': r'(S\d+)/([^/]+)/(C\d+)',
    'band_extractor': r'S\d+/([^/]+)/C\d+',
    'site_extractor': r'(S\d+)/[^/]+/C\d+',
    'cell_id_extractor': r'S\d+/[^/]+/(C\d+)'
}

SCORING_LEVELS = {
    'PASS': 1.0,
    'SOFT': 0.5,
    'FAIL': 0.0
}

NEAR_ZERO_THRESHOLDS = {
    'throughput_gbps': 0.002,  # < 1 Mbps considered ~0
    'power_watts': 0.1,        # < 0.1 W considered ~0
    'connections': 0           # Exactly 0 (no approximation)
}

RECONCILIATION_CONFIG = {
    'min_samples': 30,
    'description': 'Minimum samples needed for UE-cell throughput ratio bands'
}

KPM_TO_CANON = {
    # Cell metrics (add what's in your CSVs)
    #"timestamp": "timestamp",
    "Viavi.Cell.Name": "Viavi.Cell.Name",
    "DRB.UEThpDl": "DRB.UEThpDl",
    "DRB.UEThpUl": "DRB.UEThpUl",
    "RRU.PrbUsedDl": "RRU.PrbUsedDl",
    "RRU.PrbUsedUl": "RRU.PrbUsedUl",
    "RRU.PrbAvailDl": "RRU.PrbAvailDl",
    "RRU.PrbAvailUl": "RRU.PrbAvailUl",
    "RRU.PrbTotUl":   "RRU.PrbTotUl",
    "RRU.PrbTotDl":   "RRU.PrbTotDl",
    "RRU.MaxLayerDlMimo": "RRU.MaxLayerDlMimo",
    "CARR.AverageLayersDl": "CARR.AverageLayersDl",
    "RRC.ConnMean": "RRC.ConnMean",
    "RRC.ConnMax": "RRC.ConnMax",
    "QosFlow.TotPdcpPduVolumeUl": "QosFlow.TotPdcpPduVolumeUl",
    "QosFlow.TotPdcpPduVolumeDl": "QosFlow.TotPdcpPduVolumeDl",
    "PEE.AvgPower":   "PEE.AvgPower",
    "PEE.Energy":     "PEE.Energy",

    # UE metrics
    #"timestamp": "timestamp",
    "Viavi.UE.Name": "Viavi.UE.Name",
    "DRB.UECqiUl": "DRB.UECqiUl",
    "DRB.UEThpUl": "DRB.UEThpUl",
    "RRU.PrbUsedUl": "RRU.PrbUsedUl",
    "TB.TotNbrUl": "TB.TotNbrUl",
    "DRB.UECqiDl": "DRB.UECqiDl",
    "DRB.UEThpDl": "DRB.UEThpDl",
    "RRU.PrbUsedDl": "RRU.PrbUsedDl",
    "TB.TotNbrDl": "TB.TotNbrDl",
    
}

CANON_TO_KPM = {v: k for k, v in KPM_TO_CANON.items()}