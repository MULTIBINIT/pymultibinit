#!/usr/bin/env python3
"""
Integration test for MultibinitPotential class.
"""
import os
import sys
import numpy as np
from ase import Atoms

try:
    from pymultibinit.potential import MultibinitPotential
    print("✅ Import successful")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

def test_potential_integration():
    print("\n" + "="*70)
    print("TEST: MultibinitPotential Integration")
    print("="*70)
    
    # Get data files
    # Assuming running from project root
    root_dir = os.path.abspath(os.getcwd())
    test_data = os.path.join(root_dir, "tests", "data")
    if not os.path.exists(test_data):
        # Try relative to this file
        test_data = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../tests/data")
    
    ddb_file = os.path.join(test_data, "tmulti_l_6_DDB")
    xml_file = os.path.join(test_data, "tmulti_l_8_1.xml")
    
    print(f"Data files:")
    print(f"  DDB: {ddb_file}")
    print(f"  XML: {xml_file}")
    
    print(f"\nCreating potential (should use CFFI by default)...")
    
    try:
        pot = MultibinitPotential.from_params(
            ddb_file=ddb_file,
            sys_file=xml_file,
            ncell=(2, 2, 2),
            ngqpt=(4, 4, 4),
            use_atomic_units=False, # Use Angstrom/eV
            backend="cffi" # Explicitly request cffi to be safe
        )
        
        print(f"✅ SUCCESS: Potential created!")
        print(f"   Backend: {pot.backend}")
        
        # Note: Manually creating BaTiO3 2x2x2 supercell
        a_param = 4.149
        unit_cell = Atoms('BaTiO3',
                         scaled_positions=[
                             [0, 0, 0],        # Ba
                             [0.5, 0.5, 0.5],  # Ti
                             [0.5, 0.5, 0],    # O
                             [0.5, 0, 0.5],    # O
                             [0, 0.5, 0.5]     # O
                         ],
                         cell=[a_param, a_param, a_param],
                         pbc=True)
        structure = unit_cell * (2, 2, 2)
        
        # Set reference
        pot.set_reference_structure(structure.get_positions(), structure.get_cell())
        print(f"   Number of atoms: {pot.natoms}")
        
        # Calculate energy
        print(f"\nCalculating energy at reference positions...")
        energy, forces, stress = pot.evaluate(structure.get_positions(), structure.get_cell())
        print(f"   Energy: {energy:.6f} eV")
        
        # Calculate forces
        print(f"\nForces:")
        print(f"   Forces shape: {forces.shape}")
        print(f"   Max force: {np.max(np.abs(forces)):.6e} eV/A")
        
        # Test cleanup
        print(f"\nCleaning up...")
        pot.wrapper.free()
        print(f"✅ Cleanup successful")
        
        return True
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_potential_integration()
    if success:
        print("\n🎉 Integration test passed!")
        sys.exit(0)
    else:
        print("\n⚠️ Integration test failed")
        sys.exit(1)
