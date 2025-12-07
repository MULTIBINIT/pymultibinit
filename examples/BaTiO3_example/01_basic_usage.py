#!/usr/bin/env python
"""
Basic usage example for pymultibinit with BaHfO3 perovskite.

This script demonstrates:
1. Loading a potential from a config file
2. Building the equilibrium supercell structure 
3. Evaluating energy and forces for the structure
4. Exporting the supercell structure to file

How to run:
    python 01_basic_usage.py

Requirements:
    - pymultibinit installed
    - ase installed
    - BaTiO3_DDB and BaTiO3.xml files in this directory
    - libabinit.so/dylib accessible

Note: Despite the filename prefix, the actual data is for BaHfO3.
"""

import numpy as np
from pymultibinit import MultibinitPotential

from ase import Atoms

def main():
    print("=" * 60)
    print("BaHfO3 Perovskite Basic Usage Example")
    print("=" * 60)
    
    
    # Method 1: Load from config file
    print("\n1. Loading potential from config file...")
    pot = MultibinitPotential.from_config_file("BaTiO3_config.conf")
    print("   ✓ Potential loaded successfully")
    
    # Method 2: Alternative - load from parameters directly
    # pot = MultibinitPotential.from_params(
    #     ddb_file="BaTiO3_DDB",
    #     sys_file="BaTiO3.xml",
    #     ncell=(2, 2, 2),
    #     ngqpt=(4, 4, 4),
    #     dipdip=1
    # )
    
    # Method 3: Alternative - load from .abi file
    # pot = MultibinitPotential.from_abi("BaTiO3.abi")
    
    # Build the reference 2x2x2 supercell structure
    # From the DDB file (BaHfO3 perovskite):
    # - Lattice: 7.8411196 Bohr (cubic)
    # - 5 atoms per unit cell: Ba, Hf, O, O, O
    print("\n2. Building reference 2x2x2 supercell structure...")
    
    # Convert from Bohr to Angstrom
    a_bohr = 7.8411196
    a_angstrom = a_bohr * 0.529177210903
    
    # Create unit cell with correct fractional coordinates
    unit_cell = Atoms('BaHfO3',
                     scaled_positions=[
                         [0, 0, 0],        # Ba
                         [0.5, 0.5, 0.5],  # Hf
                         [0.5, 0, 0.5],    # O
                         [0, 0.5, 0.5],    # O
                         [0.5, 0.5, 0]     # O
                     ],
                     cell=[a_angstrom, a_angstrom, a_angstrom],
                     pbc=True)
    
    # Create 2x2x2 supercell
    supercell = unit_cell * (2, 2, 2)
    positions = supercell.get_positions()
    lattice = supercell.get_cell()
    
    print(f"   ✓ Structure has {len(supercell)} atoms")
    print(f"   ✓ Lattice: {lattice[0,0]:.2f} x {lattice[1,1]:.2f} x {lattice[2,2]:.2f} Å³")
    print(f"   ✓ Chemical formula: {supercell.get_chemical_formula()}")
    print(f"   (This is the equilibrium structure from the DDB file)")
    
    # Evaluate energy and forces
    print("\n3. Evaluating energy and forces...")
    energy, forces, stress = pot.evaluate(positions, lattice)
    
    print(f"   Energy: {energy:.6f} eV")
    print(f"   Energy per atom: {energy/len(supercell):.6f} eV")
    print(f"   Max force: {np.max(np.abs(forces)):.6e} eV/Å")
    print(f"   RMS force: {np.sqrt(np.mean(forces**2)):.6e} eV/Å")
    
    if stress is not None:
        print(f"   Max stress: {np.max(np.abs(stress)):.6e} GPa")
    
    # Verify forces are essentially zero (equilibrium structure)
    if np.max(np.abs(forces)) < 1e-6:
        print("   ✓ Forces are zero - structure is at equilibrium")
    
    # Export supercell structure
    print("\n4. Exporting supercell structure...")
    supercell.write('BaHfO3_supercell.cif')
    print(f"   ✓ Exported supercell to BaHfO3_supercell.cif")
    
    # Clean up
    print("\n5. Cleaning up...")
    pot.free()
    print("   ✓ Resources freed")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)
    return 0

if __name__ == "__main__":
    exit(main())
