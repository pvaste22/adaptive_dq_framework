# test_config_system_fixed.py
"""
Test script to verify the configuration system works properly.
FIXED VERSION - no import * syntax errors
"""

import sys
import os
from pathlib import Path

def test_config_system():
    """Test that all configuration files load properly"""
    
    print("Testing Configuration System...")
    print("-" * 40)
    
    # Check if config files exist
    project_root = Path.cwd()
    config_files = {
        'config.yaml': project_root / 'config' / 'config.yaml',
        'band_configs.json': project_root / 'config' / 'band_configs.json', 
        'drift_thresholds.yaml': project_root / 'config' / 'drift_thresholds.yaml'
    }
    
    print("1. Checking config files exist...")
    missing_files = []
    for name, path in config_files.items():
        if path.exists():
            print(f"   ✅ {name}")
        else:
            print(f"   ❌ {name} - Missing!")
            missing_files.append(name)
    
    if missing_files:
        print(f"\n❌ Missing config files: {missing_files}")
        print("Please create these files first.")
        return False
    
    # Test importing constants (FIXED - no import * inside function)
    print("\n2. Testing constants import...")
    try:
        sys.path.append(str(project_root / 'src'))
        import common.constants as const
        print("   ✅ Constants imported successfully!")
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        print("   Install dependencies: pip install pyyaml")
        return False
    except Exception as e:
        print(f"   ❌ Configuration error: {e}")
        return False
    
    # Test key values (FIXED - using const. prefix)
    print("\n3. Testing key configuration values...")
    try:
        print(f"   Expected cells: {const.EXPECTED_ENTITIES['cells']}")
        print(f"   Expected UEs: {const.EXPECTED_ENTITIES['ues']}")
        print(f"   Measurement interval: {const.MEAS_INTERVAL_SEC}s")
        print(f"   Window size: {const.WINDOW_SPECS['size_minutes']} minutes")
        print(f"   Total records per window: {const.WINDOW_SPECS['expected_records']['total_per_window']}")
        
        # Test band limits
        print(f"   Band limits: {const.BAND_LIMITS}")
        
        # Test drift parameters
        print(f"   Drift threshold: {const.DRIFT_PARAMS['significance_threshold']}")
        
        print("   ✅ All values loaded correctly!")
        
    except Exception as e:
        print(f"   ❌ Error accessing values: {e}")
        return False
    
    # Basic validation (FIXED - using const. prefix)
    print("\n4. Basic validation...")
    errors = []
    
    if const.EXPECTED_ENTITIES['cells'] != 52:
        errors.append(f"Expected 52 cells, got {const.EXPECTED_ENTITIES['cells']}")
    
    if const.EXPECTED_ENTITIES['ues'] != 48:
        errors.append(f"Expected 48 UEs, got {const.EXPECTED_ENTITIES['ues']}")
        
    if const.WINDOW_SPECS['expected_records']['total_per_window'] != 500:
        errors.append(f"Expected 500 total records per window, got {const.WINDOW_SPECS['expected_records']['total_per_window']}")
    
    if errors:
        print("   ❌ Validation errors:")
        for error in errors:
            print(f"      - {error}")
        return False
    else:
        print("   ✅ All validations passed!")
    
    return True

def main():
    print("🔧 Configuration System Test")
    print("=" * 50)
    
    success = test_config_system()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ Configuration system working properly!")
        print("🎯 Ready for next step: Create data inspector")
    else:
        print("❌ Configuration system has issues")
        print("Fix the errors above before proceeding")

if __name__ == "__main__":
    main()