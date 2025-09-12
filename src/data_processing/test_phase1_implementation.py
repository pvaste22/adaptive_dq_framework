# tests/test_phase1_integration.py

"""
Phase 1 Integration Test
Purpose: End-to-end test of data loading, conversion, and windowing
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.append('../src')

from common.constants import *
from common.logger import get_phase1_logger
from data_processing.data_loader import DataLoader
from data_processing.unit_converter import UnitConverter
from data_processing.window_generator import WindowGenerator

def test_phase1_pipeline():
    """Complete Phase 1 pipeline test."""
    logger = get_phase1_logger('phase1_test')
    
    print("="*60)
    print("PHASE 1 INTEGRATION TEST")
    print("="*60)
    
    # Test 1: Data Loading
    print("\n[TEST 1] Data Loading")
    config = {
        'cell_file': '../data/raw/CellReports.csv',
        'ue_file': '../data/raw/UEReports.csv',
        'validate_on_load': True
    }
    
    loader = DataLoader(config)
    cell_data, ue_data = loader.load_data()
    
    assert not cell_data.empty, "Cell data is empty!"
    assert not ue_data.empty, "UE data is empty!"
    print(f"✓ Loaded {len(cell_data)} cell records, {len(ue_data)} UE records")
    
    # Test 2: Unit Conversions
    print("\n[TEST 2] Unit Conversions")
    converter_config = {
        'prb_percentage_to_absolute': True,
        'energy_cumulative_to_interval': True,
        'qos_flow_validation': True,
        'measurement_interval_seconds': 60
    }
    
    converter = UnitConverter(converter_config)
    cell_data_conv, ue_data_conv = converter.standardize_units_comprehensive(
        cell_data, ue_data
    )
    
    # Verify conversions applied
    assert 'prb_conversion_applied' in cell_data_conv.columns, "PRB conversion not applied!"
    assert 'energy_conversion_applied' in cell_data_conv.columns, "Energy conversion not applied!"
    
    # Check PrbTot conversion
    if 'RRU.PrbTotDl_abs' in cell_data_conv.columns:
        assert cell_data_conv['RRU.PrbTotDl_abs'].max() > 100, "PrbTot not converted to absolute!"
        print(f"✓ PrbTot converted: max={cell_data_conv['RRU.PrbTotDl_abs'].max():.1f}")
    
    # Check energy conversion
    if 'PEE.Energy_interval' in cell_data_conv.columns:
        valid_intervals = cell_data_conv['PEE.Energy_interval'].dropna()
        print(f"✓ Energy intervals calculated: {len(valid_intervals)} valid values")
    
    # Test 3: Window Generation
    print("\n[TEST 3] Window Generation")
    window_config = {
        'window_size_minutes': 5,
        'overlap_percent': 40,
        'min_completeness': 0.95,
        'save_metadata': True
    }
    
    generator = WindowGenerator(window_config)
    windows = generator.generate_windows(cell_data_conv, ue_data_conv)
    
    assert len(windows) > 0, "No windows generated!"
    print(f"✓ Generated {len(windows)} windows")
    
    # Check window properties
    first_window = windows[0]
    assert 'window_id' in first_window
    assert 'cell_data' in first_window
    assert 'ue_data' in first_window
    assert 'metadata' in first_window
    
    completeness = first_window['metadata']['completeness']['total_completeness']
    print(f"✓ First window completeness: {completeness:.3f}")
    
    # Test 4: Artifact Saving
    print("\n[TEST 4] Artifact Saving")
    from common.utils import save_artifact, load_artifact
    
    # Save test artifact
    test_data = pd.DataFrame({'test': [1, 2, 3]})
    artifact_path = save_artifact(test_data, 'test_artifact', 'phase1')
    assert Path(artifact_path).exists(), "Artifact not saved!"
    print(f"✓ Artifact saved: {artifact_path}")
    
    # Load test artifact
    loaded_data = load_artifact(artifact_path)
    assert len(loaded_data) == 3, "Artifact loading failed!"
    print("✓ Artifact loaded successfully")
    
    # Test 5: Quality Flags
    print("\n[TEST 5] Quality Flags")
    
    # Check cell quality flags
    if 'Data_quality_flags' in cell_data_conv.columns:
        flag_counts = cell_data_conv['Data_quality_flags'].value_counts()
        print(f"Cell quality flags: {dict(flag_counts.head())}")
    
    # Check UE quality flags
    if 'Data_quality_flags' in ue_data_conv.columns:
        flag_counts = ue_data_conv['Data_quality_flags'].value_counts()
        print(f"UE quality flags: {dict(flag_counts.head())}")
    
    print("\n" + "="*60)
    print("ALL PHASE 1 TESTS PASSED!")
    print("="*60)
    
    return True

if __name__ == "__main__":
    test_phase1_pipeline()