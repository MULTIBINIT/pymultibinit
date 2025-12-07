#!/usr/bin/env python3
"""
Functional test for pymultibinit Python API - validates actual computation.

This test initializes the potential and evaluates a test structure.
Run with: python tests/test_functional.py

Note: This initializes MPI internally, so run in a single process.
"""
import sys
import os
import numpy as np

# Add pymultibinit to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pymultibinit import MultibinitPotential


def main():
    """Run functional test."""
    print("\n" + "="*70)
    print("PYMULTIBINIT FUNCTIONAL TEST")
    print("="*70)
    
    # Get test data paths
    test_data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'tests', 'data')
    abi_file = os.path.join(test_data_dir, 'tmulti_l_8_1.abi')
    ddb_file = os.path.join(test_data_dir, 'tmulti_l_6_DDB')
    sys_file = os.path.join(test_data_dir, 'tmulti_l_8_1.xml')
    
    print(f"\nTest data directory: {test_data_dir}")
    print(f"ABI file exists: {os.path.exists(abi_file)}")
    print(f"DDB file exists: {os.path.exists(ddb_file)}")
    print(f"XML file exists: {os.path.exists(sys_file)}")
    
    # Test 1: Initialization from .abi file
    print("\n" + "-"*70)
    print("TEST 1: Initialize from .abi file")
    print("-"*70)
    try:
        pot = MultibinitPotential.from_abi(abi_file, use_atomic_units=True)
        print("✓ Initialized from .abi file")
        
        # Create test structure (atomic units - Bohr)
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
        
        print(f"  Test structure: {positions.shape[0]} atoms")
        
        # Evaluate
        energy, forces, stress = pot.evaluate(positions, lattice)
        
        print(f"  Energy: {energy:.6e} Hartree")
        print(f"  Forces[0]: {forces[0]}")
        print(f"  Stress: {stress}")
        
        # Sanity checks
        assert np.isfinite(energy), "Energy is not finite"
        assert np.all(np.isfinite(forces)), "Forces contain non-finite values"
        assert np.all(np.isfinite(stress)), "Stress contains non-finite values"
        assert forces.shape == (5, 3), f"Forces shape should be (5,3), got {forces.shape}"
        assert stress.shape == (6,), f"Stress shape should be (6,), got {stress.shape}"
        
        pot.free()
        print("✓ TEST 1 PASSED")
        
    except Exception as e:
        print(f"✗ TEST 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Test 2: Initialization from parameters
    print("\n" + "-"*70)
    print("TEST 2: Initialize from parameters (no .abi file)")
    print("-"*70)
    try:
        pot = MultibinitPotential.from_params(
            ddb_file=ddb_file,
            sys_file=sys_file,
            ncell=(2, 2, 2),
            ngqpt=(4, 4, 4),
            dipdip=1,
            use_atomic_units=True
        )
        print("✓ Initialized from parameters")
        
        # Same test structure
        energy, forces, stress = pot.evaluate(positions, lattice)
        
        print(f"  Energy: {energy:.6e} Hartree")
        print(f"  Forces[0]: {forces[0]}")
        print(f"  Stress: {stress}")
        
        # Sanity checks
        assert np.isfinite(energy), "Energy is not finite"
        assert np.all(np.isfinite(forces)), "Forces contain non-finite values"
        assert np.all(np.isfinite(stress)), "Stress contains non-finite values"
        
        pot.free()
        print("✓ TEST 2 PASSED")
        
    except Exception as e:
        print(f"✗ TEST 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Test 3: Unit conversions (Angstrom/eV)
    print("\n" + "-"*70)
    print("TEST 3: Unit conversions (Angstrom/eV)")
    print("-"*70)
    try:
        pot = MultibinitPotential.from_abi(abi_file, use_atomic_units=False)
        print("✓ Initialized with eV/Angstrom units")
        
        # Convert test structure to Angstrom
        BOHR_TO_ANG = 0.529177210903
        positions_ang = positions * BOHR_TO_ANG
        lattice_ang = lattice * BOHR_TO_ANG
        
        # Evaluate in eV/Angstrom
        energy_ev, forces_ev_ang, stress_ev_ang3 = pot.evaluate(positions_ang, lattice_ang)
        
        print(f"  Energy: {energy_ev:.6e} eV")
        print(f"  Forces[0]: {forces_ev_ang[0]} eV/Angstrom")
        
        # Check conversion is correct (energy_ev should be ~27.2 * energy_ha from test 1)
        assert np.isfinite(energy_ev), "Energy is not finite"
        assert energy_ev > 0, "Energy should be positive"
        
        pot.free()
        print("✓ TEST 3 PASSED")
        
    except Exception as e:
        print(f"✗ TEST 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n" + "="*70)
    print("✓ ALL FUNCTIONAL TESTS PASSED")
    print("="*70 + "\n")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
