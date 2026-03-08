#!/usr/bin/env python
"""
Example: Using LIBABINIT_PATH environment variable

This example shows how to set and use the LIBABINIT_PATH environment
variable to specify the location of the ABINIT shared library.

Method 1: Set in shell before running Python (recommended)
    export LIBABINIT_PATH=/path/to/libabinit.dylib
    python example_libabinit_path.py

Method 2: Set in Python script (for testing/debugging)
    os.environ['LIBABINIT_PATH'] = '/path/to/libabinit.dylib'

Run:
    python examples/example_libabinit_path.py
"""
import os
from pathlib import Path

# For this example, we'll programmatically find the library
# In production, you would set LIBABINIT_PATH in your shell config
base_dir = Path(__file__).parent.parent.parent
lib_path = base_dir / "abinit" / "build_so" / "src" / "98_main" / "libabinit.dylib"

if not lib_path.exists():
    # Try Linux extension
    lib_path = lib_path.with_suffix('.so')

if lib_path.exists():
    print(f"Found library at: {lib_path}")
    os.environ['LIBABINIT_PATH'] = str(lib_path)
else:
    print("Library not found, will use auto-detection")

# Now import pymultibinit - it will use LIBABINIT_PATH if set
from pymultibinit import MultibinitPotential
from pymultibinit.utils import find_library

# Show which library was found
print(f"\nLibrary location: {find_library()}")
print(f"LIBABINIT_PATH: {os.environ.get('LIBABINIT_PATH', 'Not set')}")

# Verify we can import and the library loads correctly
print("\n=== Verifying Library ===")
try:
    from pymultibinit.wrapper_cffi import MultibinitWrapperCFFI
    wrapper = MultibinitWrapperCFFI()
    print("✓ Library loaded successfully")
    print("  CFFI wrapper created")
except Exception as e:
    print(f"✗ Failed to load library: {e}")

print("\n=== Summary ===")
print("LIBABINIT_PATH provides explicit control over library location:")
print("  ✓ No guessing or auto-detection needed")
print("  ✓ Works across different environments")
print("  ✓ Easy to configure in shell startup files")
print("\nRecommended usage:")
print("  # Add to ~/.bashrc or ~/.zshrc")
print("  export LIBABINIT_PATH=/path/to/libabinit.dylib")
