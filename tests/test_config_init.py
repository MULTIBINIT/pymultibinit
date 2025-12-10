#!/usr/bin/env python3
"""
Test config file initialization and physics validation.

Tests:
1. MultibinitPotential.from_config_file() - Basic initialization
2. MultibinitCalculator.from_config_file() - ASE calculator initialization
3. Zero forces at equilibrium - Physics validation for cubic BaHfO3
4. Non-zero restoring forces with displacement - Force direction test

This is a standalone test file that can run without MPI conflicts.

Run with:
    cd pymultibinit_dev
    PYTHONPATH=pymultibinit/src:$PYTHONPATH python pymultibinit/tests/test_config_init.py

Expected results:
- Test 1&2: Config file initialization succeeds
- Test 3: Forces < 1e-5 eV/Ang at equilibrium (validates potential correctness)
- Test 4: Forces are non-zero and restoring when atom is displaced
"""
import sys
import os
import numpy as np
import tempfile

try:
    from ase import Atoms
except ImportError:
    Atoms = None

from pymultibinit import MultibinitPotential, MultibinitCalculator


def test_potential_from_config():
    """Test MultibinitPotential.from_config_file()."""
    print("\nTesting MultibinitPotential.from_config_file()...")
    
    test_data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'tests', 'data')
    ddb_file = os.path.join(test_data_dir, 'tmulti_l_6_DDB')
    sys_file = os.path.join(test_data_dir, 'tmulti_l_8_1.xml')
    
    # Create temporary config file
    config_content = f"""# Test configuration
ddb_file: {ddb_file}
sys_file: {sys_file}
ncell: 2 2 2
ngqpt: 4 4 4
dipdip: 1
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.conf', delete=False) as f:
        f.write(config_content)
        config_file = f.name
    
    try:
        # Initialize from config
        pot = MultibinitPotential.from_config_file(config_file)
        print("  ✓ Potential initialized from config file")
        
        # Verify
        assert pot._initialized, "Potential should be initialized"
        assert pot.wrapper is not None, "Wrapper should exist"
        
        # Create test structure using ASE with scaled positions (BaHfO3 cubic perovskite)
        # Need to create 2x2x2 supercell (40 atoms) to match ncell config
        a_bohr = 7.8411196  # From test data DDB file
        a_ang = a_bohr * 0.529177210903  # Convert to Angstrom (~4.149 Å)
        unit_cell = Atoms('BaHfO3',
                         scaled_positions=[
                             [0.0, 0.0, 0.0],      # Ba
                             [0.5, 0.5, 0.5],      # Hf
                             [0.5, 0.0, 0.5],      # O
                             [0.0, 0.5, 0.5],      # O
                             [0.5, 0.5, 0.0],      # O
                         ],
                         cell=[a_ang, a_ang, a_ang],
                         pbc=True)
        atoms = unit_cell * (2, 2, 2)  # Create 2x2x2 supercell
        
        # Get positions and cell from ASE atoms
        positions = atoms.get_positions()
        lattice = atoms.get_cell().array
        
        energy, forces, stress = pot.evaluate(positions, lattice)
        
        print(f"  ✓ Energy: {energy:.6e} eV")
        print(f"  ✓ Forces shape: {forces.shape}")
        
        assert np.isfinite(energy), "Energy should be finite"
        assert forces.shape == (40, 3), "Forces shape should be (40,3) for 2x2x2 supercell"
        
        pot.free()
        print("  ✓ TEST PASSED\n")
        return True
        
    finally:
        if os.path.exists(config_file):
            os.remove(config_file)


def test_zero_forces_equilibrium():
    """Test that forces are zero at equilibrium (cubic structure)."""
    print("\nTesting zero forces at equilibrium (BaHfO3 cubic)...")
    
    test_data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'tests', 'data')
    ddb_file = os.path.join(test_data_dir, 'tmulti_l_6_DDB')
    sys_file = os.path.join(test_data_dir, 'tmulti_l_8_1.xml')
    
    # Create temporary config file
    config_content = f"""# Test configuration
ddb_file: {ddb_file}
sys_file: {sys_file}
ncell: 2 2 2
ngqpt: 4 4 4
dipdip: 1
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.conf', delete=False) as f:
        f.write(config_content)
        config_file = f.name
    
    try:
        # Initialize from config
        pot = MultibinitPotential.from_config_file(config_file)
        print("  ✓ Potential initialized from config file")
        
        # Build equilibrium structure (from DDB file)
        a_bohr = 7.8411196  # From test data DDB file
        a_ang = a_bohr * 0.529177210903  # Convert to Angstrom (~4.149 Å)
        unit_cell = Atoms('BaHfO3',
                         scaled_positions=[
                             [0.0, 0.0, 0.0],      # Ba
                             [0.5, 0.5, 0.5],      # Hf
                             [0.5, 0.0, 0.5],      # O
                             [0.0, 0.5, 0.5],      # O
                             [0.5, 0.5, 0.0],      # O
                         ],
                         cell=[a_ang, a_ang, a_ang],
                         pbc=True)
        atoms = unit_cell * (2, 2, 2)  # Create 2x2x2 supercell
        
        # Get positions and cell from ASE atoms
        positions = atoms.get_positions()
        lattice = atoms.get_cell().array
        
        energy, forces, stress = pot.evaluate(positions, lattice)
        
        print(f"  ✓ Energy: {energy:.6e} eV")
        
        # Check forces are close to zero (equilibrium structure)
        max_force = np.abs(forces).max()
        rms_force = np.sqrt(np.mean(forces**2))
        
        print(f"  ✓ Max force: {max_force:.6e} eV/Angstrom")
        print(f"  ✓ RMS force: {rms_force:.6e} eV/Angstrom")
        
        # Forces should be very close to zero at equilibrium
        force_tolerance = 1e-5  # eV/Angstrom
        assert max_force < force_tolerance, f"Max force {max_force} should be < {force_tolerance} at equilibrium"
        assert rms_force < force_tolerance, f"RMS force {rms_force} should be < {force_tolerance} at equilibrium"
        
        print(f"  ✓ Forces are zero (within tolerance {force_tolerance} eV/Ang)")
        
        pot.free()
        print("  ✓ TEST PASSED\n")
        return True
        
    finally:
        if os.path.exists(config_file):
            os.remove(config_file)


def test_nonzero_forces_displaced():
    """Test that forces are non-zero with small atomic displacement."""
    print("\nTesting non-zero forces with displacement...")
    
    test_data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'tests', 'data')
    ddb_file = os.path.join(test_data_dir, 'tmulti_l_6_DDB')
    sys_file = os.path.join(test_data_dir, 'tmulti_l_8_1.xml')
    
    # Create temporary config file
    config_content = f"""# Test configuration
ddb_file: {ddb_file}
sys_file: {sys_file}
ncell: 2 2 2
ngqpt: 4 4 4
dipdip: 1
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.conf', delete=False) as f:
        f.write(config_content)
        config_file = f.name
    
    try:
        # Initialize from config
        pot = MultibinitPotential.from_config_file(config_file)
        print("  ✓ Potential initialized from config file")
        
        # Build equilibrium structure
        a_bohr = 7.8411196  # From test data DDB file
        a_ang = a_bohr * 0.529177210903  # Convert to Angstrom (~4.149 Å)
        unit_cell = Atoms('BaHfO3',
                         scaled_positions=[
                             [0.0, 0.0, 0.0],      # Ba
                             [0.5, 0.5, 0.5],      # Hf
                             [0.5, 0.0, 0.5],      # O
                             [0.0, 0.5, 0.5],      # O
                             [0.5, 0.5, 0.0],      # O
                         ],
                         cell=[a_ang, a_ang, a_ang],
                         pbc=True)
        atoms = unit_cell * (2, 2, 2)  # Create 2x2x2 supercell
        
        # Apply small displacement to first Ba atom (atom 0)
        displacement = 0.01  # Angstrom (small displacement)
        positions = atoms.get_positions()
        positions[0, 0] += displacement  # Displace first Ba in x direction
        lattice = atoms.get_cell().array
        
        print(f"  ✓ Applied displacement: {displacement} Å to first Ba atom (x direction)")
        
        energy, forces, stress = pot.evaluate(positions, lattice)
        
        print(f"  ✓ Energy: {energy:.6e} eV")
        
        # Check forces are non-zero
        max_force = np.abs(forces).max()
        rms_force = np.sqrt(np.mean(forces**2))
        force_on_displaced_atom = forces[0]  # Force on displaced atom
        
        print(f"  ✓ Max force: {max_force:.6e} eV/Angstrom")
        print(f"  ✓ RMS force: {rms_force:.6e} eV/Angstrom")
        print(f"  ✓ Force on displaced Ba: [{force_on_displaced_atom[0]:.6e}, "
              f"{force_on_displaced_atom[1]:.6e}, {force_on_displaced_atom[2]:.6e}] eV/Ang")
        
        # Force should be significant and pointing opposite to displacement
        force_threshold = 1e-6  # eV/Angstrom (should be much larger than this)
        assert max_force > force_threshold, f"Max force {max_force} should be > {force_threshold} with displacement"
        
        # Force on displaced atom should be primarily in x direction (opposite to displacement)
        # and negative (restoring force)
        assert abs(force_on_displaced_atom[0]) > force_threshold, \
            f"Force in x-direction {force_on_displaced_atom[0]} should be significant"
        assert force_on_displaced_atom[0] < 0, \
            f"Force should be restoring (negative, opposite to +x displacement), got {force_on_displaced_atom[0]}"
        
        print(f"  ✓ Forces are non-zero and restoring (displacement: +x, force: -x)")
        print(f"  ✓ Force magnitude ratio: {abs(force_on_displaced_atom[0]) / force_threshold:.1f}x threshold")
        
        pot.free()
        print("  ✓ TEST PASSED\n")
        return True
        
    finally:
        if os.path.exists(config_file):
            os.remove(config_file)


def test_calculator_from_config():
    """Test MultibinitCalculator.from_config_file()."""
    print("\nTesting MultibinitCalculator.from_config_file()...")
    
    try:
        from ase import Atoms
        
    except ImportError:
        print("  ⚠ Skipping (ASE not installed)\n")
        return True
    
    test_data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'tests', 'data')
    ddb_file = os.path.join(test_data_dir, 'tmulti_l_6_DDB')
    sys_file = os.path.join(test_data_dir, 'tmulti_l_8_1.xml')
    
    # Create temporary config file
    config_content = f"""# Test configuration
ddb_file: {ddb_file}
sys_file: {sys_file}
ncell: 2 2 2
ngqpt: 4 4 4
dipdip: 1
use_atomic_units: false
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.conf', delete=False) as f:
        f.write(config_content)
        config_file = f.name
    
    try:
        # Initialize calculator
        calc = MultibinitCalculator.from_config_file(config_file)
        print("  ✓ Calculator initialized from config file")
        
        # Verify
        assert calc.potential is not None, "Potential should exist"
        assert calc.potential._initialized, "Potential should be initialized"
        
        # Create test structure using ASE with scaled positions (BaHfO3 cubic perovskite)
        # Need to create 2x2x2 supercell (40 atoms) to match ncell config
        a_bohr = 7.8411196  # From test data DDB file
        a_ang = a_bohr * 0.529177210903  # Convert to Angstrom (~4.149 Å)
        unit_cell = Atoms('BaHfO3',
                         scaled_positions=[
                             [0.0, 0.0, 0.0],      # Ba
                             [0.5, 0.5, 0.5],      # Hf
                             [0.5, 0.0, 0.5],      # O
                             [0.0, 0.5, 0.5],      # O
                             [0.5, 0.5, 0.0],      # O
                         ],
                         cell=[a_ang, a_ang, a_ang],
                         pbc=True)
        atoms = unit_cell * (2, 2, 2)  # Create 2x2x2 supercell
        atoms.calc = calc
        
        # Get properties
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        
        print(f"  ✓ Energy: {energy:.6e} eV")
        print(f"  ✓ Forces shape: {forces.shape}")
        
        assert np.isfinite(energy), "Energy should be finite"
        assert forces.shape == (40, 3), "Forces shape should be (40,3) for 2x2x2 supercell"
        
        print("  ✓ TEST PASSED\n")
        return True
        
    finally:
        if os.path.exists(config_file):
            os.remove(config_file)


if __name__ == '__main__':
    print("\n" + "="*60)
    print("Testing Config File Initialization and Physics")
    print("="*60)
    
    tests = [
        test_potential_from_config,
        test_calculator_from_config,
        test_zero_forces_equilibrium,
        test_nonzero_forces_displaced,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            failed += 1
            print(f"\n✗ FAILED: {test_func.__name__}")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("="*60)
    print(f"SUMMARY: {passed} passed, {failed} failed")
    print("="*60 + "\n")
    
    sys.exit(0 if failed == 0 else 1)
