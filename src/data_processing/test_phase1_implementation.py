#!/usr/bin/env python3
"""
Complete Phase 1 Testing Script
Tests all components step by step
"""

import sys
from pathlib import Path
import pandas as pd

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

def test_step1_configuration():
    """Test configuration loading"""
    print("Testing configuration...")
    try:
        from common.constants import EXPECTED_ENTITIES, VIAVI_CONFIG, PATHS
        print(f"  ✓ Expected cells: {EXPECTED_ENTITIES['cells']}")
        print(f"  ✓ Expected UEs: {EXPECTED_ENTITIES['ues']}")
        print(f"  ✓ Cell file: {VIAVI_CONFIG['cell_reports_file']}")
        return True
    except Exception as e:
        print(f"  ❌ Configuration failed: {e}")
        return False

def test_step2_data_files():
    """Test data file existence"""
    print("Testing data files...")
    try:
        from common.constants import VIAVI_CONFIG
        
        cell_file = Path(VIAVI_CONFIG['cell_reports_file'])
        ue_file = Path(VIAVI_CONFIG['ue_reports_file'])
        
        if cell_file.exists():
            print(f"  ✓ Cell file found: {cell_file}")
        else:
            print(f"  ❌ Cell file missing: {cell_file}")
            return False
            
        if ue_file.exists():
            print(f"  ✓ UE file found: {ue_file}")
        else:
            print(f"  ❌ UE file missing: {ue_file}")
            return False
            
        return True
    except Exception as e:
        print(f"  ❌ Data file test failed: {e}")
        return False

def test_step3_data_loading():
    """Test data loading"""
    print("Testing data loading...")
    try:
        from data_processing.data_loader import DataLoader
        
        config = {'measurement_interval_seconds': 60}
        loader = DataLoader(config)
        cell_data, ue_data = loader.load_data()
        
        print(f"  ✓ Loaded {len(cell_data)} cell records")
        print(f"  ✓ Loaded {len(ue_data)} UE records")
        print(f"  ✓ Cell columns: {len(cell_data.columns)}")
        print(f"  ✓ UE columns: {len(ue_data.columns)}")
        
        return cell_data, ue_data
    except Exception as e:
        print(f"  ❌ Data loading failed: {e}")
        return None, None

def test_step4_unit_conversion(cell_data, ue_data):
    """Test unit conversions"""
    print("Testing unit conversions...")
    try:
        from data_processing.unit_converter import UnitConverter
        
        config = {
            'measurement_interval_seconds': 60,
            'prb_percentage_to_absolute': True,
            'energy_cumulative_to_interval': True,
            'qos_flow_validation': True
        }
        
        converter = UnitConverter(config)
        cell_converted, ue_converted = converter.standardize_units_comprehensive(cell_data, ue_data)
        
        print(f"  ✓ Cell data converted: {len(cell_converted)} records")
        print(f"  ✓ UE data converted: {len(ue_converted)} records")
        
        # Check for conversion columns
        if 'RRU.PrbTotDl_abs' in cell_converted.columns:
            print("  ✓ PrbTot conversion applied")
        if 'PEE.Energy_interval' in cell_converted.columns:
            print("  ✓ Energy conversion applied")
            
        return cell_converted, ue_converted
    except Exception as e:
        print(f"  ❌ Unit conversion failed: {e}")
        return None, None

def test_step5_window_generation(cell_data, ue_data):
    """Test window generation"""
    print("Testing window generation...")
    try:
        from data_processing.window_generator import WindowGenerator
        
        config = {'min_window_completeness': 0.8}
        generator = WindowGenerator(config)
        windows = generator.generate_windows(cell_data, ue_data)
        
        print(f"  ✓ Generated {len(windows)} windows")
        
        if windows:
            sample_window = windows[0]
            print(f"  ✓ Sample window ID: {sample_window['window_id']}")
            print(f"  ✓ Cell records in first window: {len(sample_window['cell_data'])}")
            print(f"  ✓ UE records in first window: {len(sample_window['ue_data'])}")
            
        return windows
    except Exception as e:
        print(f"  ❌ Window generation failed: {e}")
        return None

def test_step6_save_windows(windows):
    """Test saving windows"""
    print("Testing window saving...")
    try:
        from data_processing.window_generator import WindowGenerator
        from common.constants import PATHS
        
        output_dir = PATHS['processed_data'] / 'test_phase1' / 'windows'
        
        config = {}
        generator = WindowGenerator(config)
        
        # Save first 5 windows only for testing
        test_windows = windows[:5] if len(windows) > 5 else windows
        generator.save_windows(test_windows, output_dir)
        
        print(f"  ✓ Saved {len(test_windows)} test windows to {output_dir}")
        
        # Verify files exist
        first_window_dir = output_dir / test_windows[0]['window_id']
        if first_window_dir.exists():
            print("  ✓ Window directory created")
            if (first_window_dir / 'cell_data.parquet').exists():
                print("  ✓ Cell data parquet saved")
            if (first_window_dir / 'metadata.json').exists():
                print("  ✓ Metadata JSON saved")
        
        return True
    except Exception as e:
        print(f"  ❌ Window saving failed: {e}")
        return False

def main():
    """Run complete Phase 1 test"""
    print("="*60)
    print("PHASE 1 COMPLETE TESTING")
    print("="*60)
    
    # Test each step
    tests = [
        ("Configuration", test_step1_configuration),
        ("Data Files", test_step2_data_files)
    ]
    
    # Run basic tests first
    for test_name, test_func in tests:
        print(f"\n[{test_name}]")
        if not test_func():
            print(f"\n❌ {test_name} test failed. Fix before continuing.")
            return False
    
    # Run data processing tests
    print(f"\n[Data Loading]")
    cell_data, ue_data = test_step3_data_loading()
    if cell_data is None:
        print("\n❌ Data loading failed. Cannot continue.")
        return False
    
    print(f"\n[Unit Conversion]")
    cell_converted, ue_converted = test_step4_unit_conversion(cell_data, ue_data)
    if cell_converted is None:
        print("\n❌ Unit conversion failed. Cannot continue.")
        return False
    
    print(f"\n[Window Generation]")
    windows = test_step5_window_generation(cell_converted, ue_converted)
    if not windows:
        print("\n❌ Window generation failed. Cannot continue.")
        return False
    
    print(f"\n[Window Saving]")
    if not test_step6_save_windows(windows):
        print("\n❌ Window saving failed.")
        return False
    
    # Summary
    print("\n" + "="*60)
    print("PHASE 1 TEST RESULTS")
    print("="*60)
    print("✓ All tests passed!")
    print(f"✓ Data processed: {len(cell_data)} cell + {len(ue_data)} UE records")
    print(f"✓ Windows generated: {len(windows)}")
    print("✓ Ready for full Phase 1 execution")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)