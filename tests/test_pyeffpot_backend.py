"""
Test MultibinitPotential with pyeffpot backend.

How to run:
    pytest pymultibinit/tests/test_pyeffpot_backend.py -v

What it tests:
- from_pyeffpot() factory method
- Backend switching
- Unit conversions (Angstrom/eV <-> Bohr/Hartree)
- Energy/forces/stress evaluation
"""

import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from pymultibinit import MultibinitPotential
from pymultibinit.potential import BOHR_TO_ANGSTROM, HARTREE_TO_EV


class TestPyeffpotBackend:
    """Test MultibinitPotential with pyeffpot backend."""
    
    @pytest.fixture
    def bto_ddb_path(self):
        """Path to BaHfO3 DDB test file."""
        return Path(__file__).parent.parent.parent / 'abinit/tests/v9/Input/BTO.DDB'
    
    def test_from_pyeffpot_creation(self, bto_ddb_path):
        """Test creating potential with pyeffpot backend."""
        pot = MultibinitPotential.from_pyeffpot(
            ddb_file=str(bto_ddb_path),
            ncell=(2, 2, 2)
        )
        
        assert pot.backend == 'pyeffpot'
        assert pot.supercell == (2, 2, 2)
        assert pot.expected_natoms == 40
    
    def test_reference_structure_set(self, bto_ddb_path):
        """Test that reference structure is set automatically."""
        pot = MultibinitPotential.from_pyeffpot(
            ddb_file=str(bto_ddb_path),
            ncell=(2, 2, 2)
        )
        
        ref_pos, ref_lat, _ = pot.get_supercell_structure()
        assert ref_pos is not None
        assert ref_lat is not None
        assert ref_pos.shape == (40, 3)
        assert ref_lat.shape == (3, 3)
    
    def test_evaluate_at_equilibrium(self, bto_ddb_path):
        """Test evaluation at equilibrium positions."""
        pot = MultibinitPotential.from_pyeffpot(
            ddb_file=str(bto_ddb_path),
            ncell=(2, 2, 2)
        )
        
        ref_pos, ref_lat, _ = pot.get_supercell_structure()
        energy, forces, stress = pot.evaluate(ref_pos, ref_lat)
        
        assert isinstance(energy, float)
        assert forces.shape == (40, 3)
        assert stress.shape == (6,)
        
        assert energy < 0
        assert np.max(np.abs(forces)) < 1e-10
    
    def test_evaluate_with_displacement(self, bto_ddb_path):
        """Test evaluation with displaced atoms."""
        pot = MultibinitPotential.from_pyeffpot(
            ddb_file=str(bto_ddb_path),
            ncell=(2, 2, 2)
        )
        
        ref_pos, ref_lat, _ = pot.get_supercell_structure()
        
        displaced_pos = ref_pos.copy()
        displaced_pos[0] += np.array([0.1, 0.0, 0.0])
        
        energy, forces, stress = pot.evaluate(displaced_pos, ref_lat)
        
        assert energy > pot.evaluate(ref_pos, ref_lat)[0]
        assert not np.allclose(forces, 0.0)
    
    def test_unit_conversions(self, bto_ddb_path):
        """Test that unit conversions are correct."""
        pot = MultibinitPotential.from_pyeffpot(
            ddb_file=str(bto_ddb_path),
            ncell=(2, 2, 2)
        )
        
        ref_pos, ref_lat, _ = pot.get_supercell_structure()
        
        ref_pos_bohr = pot._pyeffpot_potential._reference_positions
        ref_lat_bohr = pot._pyeffpot_potential._reference_lattice
        
        assert np.allclose(ref_pos, ref_pos_bohr * BOHR_TO_ANGSTROM)
        assert np.allclose(ref_lat, ref_lat_bohr * BOHR_TO_ANGSTROM)
    
    def test_different_supercell_sizes(self, bto_ddb_path):
        """Test different supercell sizes."""
        for ncell in [(1, 1, 1), (2, 2, 2), (3, 2, 1)]:
            pot = MultibinitPotential.from_pyeffpot(
                ddb_file=str(bto_ddb_path),
                ncell=ncell
            )
            
            expected_natoms = 5 * ncell[0] * ncell[1] * ncell[2]
            assert pot.expected_natoms == expected_natoms
    
    @pytest.mark.skip(reason="Atom reordering energy invariance needs investigation")
    def test_auto_atom_matching(self, bto_ddb_path):
        """Test automatic atom matching."""
        pot = MultibinitPotential.from_pyeffpot(
            ddb_file=str(bto_ddb_path),
            ncell=(2, 2, 2),
            auto_match_atoms=True
        )
        
        ref_pos, ref_lat, _ = pot.get_supercell_structure()
        
        np.random.seed(42)
        perm = np.random.permutation(len(ref_pos))
        shuffled_pos = ref_pos[perm]
        
        energy_shuffled, forces_shuffled, _ = pot.evaluate(shuffled_pos, ref_lat)
        energy_ref, forces_ref, _ = pot.evaluate(ref_pos, ref_lat)
        
        assert np.allclose(energy_shuffled, energy_ref, rtol=1e-10)
    
    def test_context_manager(self, bto_ddb_path):
        """Test context manager support."""
        with MultibinitPotential.from_pyeffpot(
            ddb_file=str(bto_ddb_path),
            ncell=(2, 2, 2)
        ) as pot:
            assert pot._initialized
            ref_pos, ref_lat, _ = pot.get_supercell_structure()
            energy, forces, stress = pot.evaluate(ref_pos, ref_lat)
            assert isinstance(energy, float)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
