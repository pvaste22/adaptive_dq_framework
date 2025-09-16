#!/usr/bin/env python3
"""
Test script for Phase 1 components
Tests each module individually before full pipeline execution
"""

import sys
from pathlib import Path
import pandas as pd
import logging

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

from common.constants import DATA_FILES, PATHS
from data_processing.data_loader import DataLoader
from data_processing.unit_converter import UnitConverter
from data_processing.window_generator import WindowGenerator

def test_data_files_exist():
    """Test 1: Verify data files exist"""
    print("\n=== Test 1: Data Files ===")
    
    for name, path in DATA_FILES.items():
        if path.exists():
            size_mb = path.stat().st_size / 1024 / 1024
            print(f"✓ {name}: {path} ({size_mb:.2f} MB)")
        else:
            print(f"✗ {name}: {path} NOT FOUND")
            return False
    return True

def test_module_initialization():
    """Test 2: Initialize each module"""
    print("\n=== Test 2: Module Initialization ===")
    
    try:
        loader = DataLoader()
        print("✓ DataLoader initialized")
        
        converter = UnitConverter()
        print("✓ UnitConverter initialized")
        
        generator = WindowGenerator()
        print("✓ WindowGenerator initialized")
        
        return True
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        return False

def test_data_loading_sample():
    """Test 3: Load sample of data"""
    print("\n=== Test 3: Sample Data Loading ===")
    
    try:
        loader = DataLoader()
        
        # Load just first 1000 rows for testing
        cell1 = pd.read_csv(DATA_FILES['cell_reports'], low_memory=False)
        print(f"dtypes: {cell1.dtypes}")
        ue1= pd.read_csv(DATA_FILES['ue_reports'], low_memory=False)
        print(f"dtypes: {ue1.dtypes}")
        cell_sample = pd.read_csv(DATA_FILES['cell_reports'], nrows=1000)
        ue_sample = pd.read_csv(DATA_FILES['ue_reports'], nrows=1000)
        
        print(f"✓ Cell sample: {cell_sample.shape}", "dtypes: {cell_sample.dtypes}")
        print(f"✓ UE sample: {ue_sample.shape}", "dtypes: {ue_sample.dtypes}")
        
        # Test corrections
        cell_sample['timestamp'] = pd.to_datetime(cell_sample['timestamp'], unit='s')
        ue_sample['timestamp'] = pd.to_datetime(ue_sample['timestamp'], unit='s')
        
        corrected_cell = loader._correct_cell_data(cell_sample)
        corrected_ue = loader._correct_ue_data(ue_sample)
        
        # Check if Band was extracted
        if 'Band' in corrected_cell.columns:
            print(f"✓ Band extraction: {corrected_cell['Band'].value_counts().to_dict()}")
        
        # Check CQI correction
        if 'DRB.UECqiDl' in corrected_ue.columns:
            nan_count = corrected_ue['DRB.UECqiDl'].isna().sum()
            print(f"✓ CQI correction: {nan_count} NaN values created")
        
        return True
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        return False

def test_unit_conversions():
    """Test 4: Verify unit conversions create expected columns"""
    print("\n=== Test 4: Unit Conversions ===")
    
    try:
        converter = UnitConverter()
        
        # Create sample data
        sample_cell = pd.DataFrame({
            'RRU.PrbTotDl': [50.0, 75.0],
            'RRU.PrbAvailDl': [100, 100],
            'RRU.PrbTotUl': [40.0, 60.0],
            'RRU.PrbAvailUl': [100, 100],
            'PEE.Energy': [100, 200],
            'PEE.AvgPower': [50, 50],
            'Viavi.Cell.Name': ['S1/B2/C1', 'S1/B2/C1'],
            'timestamp': pd.date_range('2023-01-01', periods=2, freq='60s')
        })
        
        sample_ue = pd.DataFrame()
        
        converted_cell, converted_ue = converter.standardize_units_comprehensive(
            sample_cell, sample_ue
        )
        
        # Check new columns
        expected_columns = ['RRU.PrbTotDl_abs', 'RRU.PrbTotUl_abs', 
                          'PEE.Energy_interval', 'unit_conversion_version']
        
        for col in expected_columns:
            if col in converted_cell.columns:
                print(f"✓ Created column: {col}")
            else:
                print(f"✗ Missing column: {col}")
        
        return all(col in converted_cell.columns for col in expected_columns[:3])
        
    except Exception as e:
        print(f"✗ Conversion test failed: {e}")
        return False

def test_window_generation():
    """Test 5: Generate single window"""
    print("\n=== Test 5: Window Generation ===")
    
    try:
        # Use actual data loading process
        loader = DataLoader()
        generator = WindowGenerator()
        
        # Load enough data for at least one window (5 minutes = 5 timestamps)
        # 52 cells × 5 timestamps = 260 minimum cell records needed
        # 48 UEs × 5 timestamps = 240 minimum UE records needed
        cell_sample = pd.read_csv(DATA_FILES['cell_reports'], nrows=300)
        ue_sample = pd.read_csv(DATA_FILES['ue_reports'], nrows=300)
        
        # Process data properly
        cell_sample['timestamp'] = pd.to_datetime(cell_sample['timestamp'], unit='s')
        ue_sample['timestamp'] = pd.to_datetime(ue_sample['timestamp'], unit='s')
        
        corrected_cell = loader._correct_cell_data(cell_sample)
        corrected_ue = loader._correct_ue_data(ue_sample)
        
        print(f"  Cell data shape: {corrected_cell.shape}")
        print(f"  UE data shape: {corrected_ue.shape}")
        print(f"  Timestamps in cell data: {corrected_cell['timestamp'].nunique()}")
        print(f"  Timestamps in UE data: {corrected_ue['timestamp'].nunique()}")
        
        # Generate windows
        windows = generator.generate_windows(corrected_cell, corrected_ue)
        
        if windows:
            print(f"✓ Generated {len(windows)} window(s)")
            print(f"  Window ID: {windows[0]['window_id']}")
            print(f"  Completeness: {windows[0]['metadata']['completeness']['total_completeness']:.2%}")
            return True
        else:
            print("✗ No windows generated")
            # Debug why
            print(f"  Check expected records per window:")
            print(f"    Expected cells: {generator.window_specs['expected_records']['cells_per_window']}")
            print(f"    Expected UEs: {generator.window_specs['expected_records']['ues_per_window']}")
            return False
            
    except Exception as e:
        print(f"✗ Window generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
def main():
    """Run all tests"""
    print("="*60)
    print("PHASE 1 COMPONENT TESTING")
    print("="*60)
    
    tests = [
        ("Data Files Exist", test_data_files_exist),
        ("Module Initialization", test_module_initialization),
        ("Sample Data Loading", test_data_loading_sample),
        ("Unit Conversions", test_unit_conversions),
        ("Window Generation", test_window_generation)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n✗ Test '{name}' crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status}: {name}")
    
    all_passed = all(p for _, p in results)
    
    if all_passed:
        print("\n✓ All tests passed! Ready to run Phase 1 pipeline.")
        return 0
    else:
        print("\n✗ Some tests failed. Fix issues before running pipeline.")
        return 1

if __name__ == "__main__":
    sys.exit(main())