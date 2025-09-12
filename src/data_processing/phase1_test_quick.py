#!/usr/bin/env python3
"""Quick Phase 1 smoke test - run from project root"""

import sys
sys.path.append('src')

print("Testing Phase 1 imports...")
try:
    from common import constants, logger, utils, exceptions
    from data_processing import data_loader, unit_converter, window_generator
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

print("\nChecking configurations...")
try:
    print(f"✓ Cells expected: {constants.EXPECTED_ENTITIES['cells']}")
    print(f"✓ UEs expected: {constants.EXPECTED_ENTITIES['ues']}")
    print(f"✓ Window size: {constants.WINDOW_SPECS['size_minutes']} minutes")
    print(f"✓ Bands configured: {list(constants.BAND_SPECS.keys())}")
except Exception as e:
    print(f"✗ Config error: {e}")
    sys.exit(1)

print("\n✅ Phase 1 smoke test passed!")