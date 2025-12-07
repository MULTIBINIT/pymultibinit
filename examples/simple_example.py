#!/usr/bin/env python
"""
Simple working example for pymultibinit.

Demonstrates:
1. Loading potential from DDB and XML files
2. Building a reference structure  
3. Evaluating forces (should be zero for equilibrium)

Uses test data (BaHfO3) included with pymultibinit.
"""
import numpy as np
from pathlib import Path
from pymultibinit import MultibinitPotential

def main():
    print("=" * 70)
    print("PyMultibinit Simple Example")
    print("=" * 70)
    
    # Find test data
    test_data = Path(__file__).parent.parent.parent / "tests" / "data"
    
    if not test_data.exists():
        print(f"\n❌ Test data not found: {test_data}")
        return
    
    ddb_file = test_data / "tmulti_l_6_DDB"
    xml_file = test_data / "tmulti_l_8_1.xml"
    
    # Load potential for 2x2x2 supercell
    print("\n1. Loading potential (2×2×2 supercell)...")
    pot = MultibinitPotential.from_params(
        ddb_file=str(ddb_file),
        sys_file=str(xml_file),
        ncell=(2, 2, 2),
        use_atomic_units=False  # Use Angstrom/eV
    )
    print("   ✓ Potential loaded")
    
    # Build reference structure
    # Unit cell: a = 7.8411196 Bohr = 4.1493418 Å (cubic)
    # 5 atoms: Ba, Hf, O, O, O
    print("\n2. Building 2×2×2 reference structure (40 atoms)...")
    
    a = 4.1493418  # Angstrom
    unit_positions = np.array([
        [0.0, 0.0, 0.0],      # Ba
        [0.5*a, 0.5*a, 0.5*a],  # Hf
        [0.5*a, 0.0, 0.5*a],    # O
        [0.0, 0.5*a, 0.5*a],    # O
        [0.5*a, 0.5*a, 0.0]     # O
    ])
    
    # Build 2x2x2 supercell
    positions = []
    for ix in range(2):
        for iy in range(2):
            for iz in range(2):
                offset = np.array([ix*a, iy*a, iz*a])
                for pos in unit_positions:
                    positions.append(pos + offset)
    
    positions = np.array(positions)
    lattice = np.diag([2*a, 2*a, 2*a])
    
    print(f"   ✓ Created {len(positions)} atoms")
    print(f"   ✓ Lattice: {lattice[0,0]:.3f} × {lattice[1,1]:.3f} × {lattice[2,2]:.3f} Å")
    
    # Evaluate
    print("\n3. Evaluating energy and forces...")
    energy, forces, stress = pot.evaluate(positions, lattice)
    
    max_force = np.max(np.abs(forces))
    print(f"   Energy: {energy:.4f} eV")
    print(f"   Max force: {max_force:.2e} eV/Å")
    
    # Check
    print("\n4. Verification...")
    if max_force < 1e-10:
        print(f"   ✅ PASS: Forces are zero (equilibrium structure)")
    else:
        print(f"   ⚠ Warning: Non-zero forces ({max_force:.2e} eV/Å)")
    
    pot.free()
    print("\n" + "=" * 70)
    print("✓ Example completed")
    print("=" * 70)

if __name__ == "__main__":
    main()
