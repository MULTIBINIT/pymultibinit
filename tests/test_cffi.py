#!/usr/bin/env python3
"""
Test CFFI wrapper implementation.

Run with: python tests/test_cffi.py

Tests the CFFI wrapper against the ctypes wrapper to ensure
identical behavior.
"""
import sys
import os
import numpy as np

# Add pymultibinit to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

print("Testing CFFI wrapper...")

# Check if cffi is available
try:
    import cffi
    print("✓ CFFI package is available")
except ImportError:
    print("✗ CFFI not installed. Install with: pip install cffi")
    sys.exit(1)

# Test importing wrapper
try:
    from pymultibinit.wrapper_cffi import MultibinitWrapperCFFI
    print("✓ MultibinitWrapperCFFI imported")
except Exception as e:
    print(f"✗ Failed to import MultibinitWrapperCFFI: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test library loading
try:
    wrapper_cffi = MultibinitWrapperCFFI()
    print(f"✓ CFFI library loaded successfully")
    print(f"  Library: {wrapper_cffi.lib}")
except Exception as e:
    print(f"✗ Failed to load library with CFFI: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Compare with ctypes wrapper
print("\nComparing CFFI vs ctypes wrapper...")
try:
    from pymultibinit.wrapper import MultibinitWrapper
    wrapper_ctypes = MultibinitWrapper()
    print("✓ Both wrappers loaded")
except Exception as e:
    print(f"✗ Failed to load ctypes wrapper: {e}")
    sys.exit(1)

# Test API compatibility - check that both have same methods
methods = ['init_from_abi_file', 'init_from_params', 'evaluate', 'free']
for method in methods:
    has_cffi = hasattr(wrapper_cffi, method)
    has_ctypes = hasattr(wrapper_ctypes, method)
    if has_cffi and has_ctypes:
        print(f"  ✓ Both have method: {method}")
    else:
        print(f"  ✗ Method mismatch: {method} (cffi={has_cffi}, ctypes={has_ctypes})")

# Test with potential API
print("\nTesting MultibinitPotential with CFFI backend...")
try:
    from pymultibinit import MultibinitPotential
    
    test_data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'tests', 'data')
    ddb_file = os.path.join(test_data_dir, 'tmulti_l_6_DDB')
    sys_file = os.path.join(test_data_dir, 'tmulti_l_8_1.xml')
    
    # Test ctypes backend
    print("\n  Testing ctypes backend...")
    pot_ctypes = MultibinitPotential.from_params(
        ddb_file=ddb_file,
        sys_file=sys_file,
        ncell=(2, 2, 2),
        ngqpt=(4, 4, 4),
        dipdip=1,
        use_atomic_units=True,
        backend="ctypes"
    )
    print("  ✓ ctypes backend initialized")
    
    # Test CFFI backend
    print("\n  Testing CFFI backend...")
    pot_cffi = MultibinitPotential.from_params(
        ddb_file=ddb_file,
        sys_file=sys_file,
        ncell=(2, 2, 2),
        ngqpt=(4, 4, 4),
        dipdip=1,
        use_atomic_units=True,
        backend="cffi"
    )
    print("  ✓ CFFI backend initialized")
    
    # Test structure
    positions = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 2.0, 0.0],
        [2.0, 0.0, 2.0],
        [0.0, 2.0, 2.0],
        [1.0, 1.0, 1.0],
    ])
    lattice = np.array([
        [7.0, 0.0, 0.0],
        [0.0, 7.0, 0.0],
        [0.0, 0.0, 7.0],
    ])
    
    # Evaluate with ctypes
    energy_ctypes, forces_ctypes, stress_ctypes = pot_ctypes.evaluate(positions, lattice)
    print(f"\n  ctypes results:")
    print(f"    Energy: {energy_ctypes:.6e} Hartree")
    print(f"    Forces[0]: {forces_ctypes[0]}")
    
    # Evaluate with CFFI
    energy_cffi, forces_cffi, stress_cffi = pot_cffi.evaluate(positions, lattice)
    print(f"\n  CFFI results:")
    print(f"    Energy: {energy_cffi:.6e} Hartree")
    print(f"    Forces[0]: {forces_cffi[0]}")
    
    # Compare results
    energy_match = np.allclose(energy_ctypes, energy_cffi, rtol=1e-10)
    forces_match = np.allclose(forces_ctypes, forces_cffi, rtol=1e-10)
    stress_match = np.allclose(stress_ctypes, stress_cffi, rtol=1e-10)
    
    print(f"\n  Comparison:")
    print(f"    Energy match: {energy_match} (diff: {abs(energy_ctypes - energy_cffi):.2e})")
    print(f"    Forces match: {forces_match} (max diff: {np.max(np.abs(forces_ctypes - forces_cffi)):.2e})")
    print(f"    Stress match: {stress_match} (max diff: {np.max(np.abs(stress_ctypes - stress_cffi)):.2e})")
    
    if energy_match and forces_match and stress_match:
        print("\n  ✓ Results match perfectly!")
    else:
        print("\n  ✗ Results differ between backends")
        sys.exit(1)
    
    # Cleanup
    pot_ctypes.free()
    pot_cffi.free()
    print("\n✓ Cleanup successful")
    
except Exception as e:
    print(f"\n✗ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("✓ ALL CFFI TESTS PASSED")
print("="*70)
