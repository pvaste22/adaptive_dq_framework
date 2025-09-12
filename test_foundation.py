
#!/usr/bin/env python3
"""
Test script for foundation files
"""
from pathlib import Path
import os, glob
def test_imports():
    """Test that all foundation modules can be imported."""
    try:
        from src.common.constants import BAND_LIMITS, EXPECTED_ENTITIES
        from src.common.logger import setup_logger, get_phase1_logger
        from src.common.utils import extract_band_from_cell_name, save_artifact
        from src.common.exceptions import DataQualityFrameworkError
        
        print("✅ All imports successful")
        print(f"✅ Band limits loaded: {list(BAND_LIMITS.keys())}")
        print(f"✅ Expected entities: {EXPECTED_ENTITIES}")
        
        # Test logger
        logger = get_phase1_logger('foundation_test')
        logger.info("Foundation test logging successful")
        print("✅ Logger working")
        
        # Test utility function
        band = extract_band_from_cell_name('S1/B13/C1')
        print(f"✅ Utility function working: extracted band '{band}'")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_configurations():
    """Test configuration files can be loaded."""
    try:
        import json
        import yaml
        
        # Test band configs
        print("CWD:", os.getcwd())
        with open('config/band_configs.json', 'r') as f:
            band_config = json.load(f)
        print(f"✅ Band config loaded: {len(band_config['band_specifications'])} bands")
        
        # Test drift thresholds
        with open('config/drift_thresholds.yaml', 'r') as f:
            drift_config = yaml.safe_load(f)
        print(f"✅ Drift config loaded: significance threshold = {drift_config['drift_detection']['significance_threshold']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing foundation files...")
    
    imports_ok = test_imports()
    configs_ok = test_configurations()
    
    if imports_ok and configs_ok:
        print("\n🎉 Foundation setup successful!")
        print("Ready to proceed with Phase 1 implementation")
    else:
        print("\n⚠️  Foundation setup has issues - fix before proceeding")
