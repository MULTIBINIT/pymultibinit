#!/usr/bin/env python3
"""
Simple test for pymultibinit wrapper - library loading only.

Run with: python tests/test_import.py
"""
import sys
import os

# Add pymultibinit to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

print("Testing pymultibinit imports...")

try:
    from pymultibinit.wrapper import MultibinitWrapper
    print("✓ MultibinitWrapper imported")
except Exception as e:
    print(f"✗ MultibinitWrapper import failed: {e}")
    sys.exit(1)

try:
    from pymultibinit.potential import MultibinitPotential
    print("✓ MultibinitPotential imported")
except Exception as e:
    print(f"✗ MultibinitPotential import failed: {e}")
    sys.exit(1)

try:
    from pymultibinit.calculator import MultibinitCalculator
    print("✓ MultibinitCalculator imported")
except Exception as e:
    print(f"✗ MultibinitCalculator import failed: {e}")
    sys.exit(1)

try:
    from pymultibinit import MultibinitPotential, MultibinitCalculator
    print("✓ Main package imports work")
except Exception as e:
    print(f"✗ Main package import failed: {e}")
    sys.exit(1)

# Try to load library
try:
    wrapper = MultibinitWrapper()
    print(f"✓ Library loaded successfully")
    print(f"  Library path: {wrapper.lib._name}")
except Exception as e:
    print(f"✗ Library loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✓ ALL IMPORT TESTS PASSED")
