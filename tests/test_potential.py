"""
Unit tests for EffectivePotential evaluation.

How to run:
    pytest pymultibinit/tests/test_potential.py -v

What it tests:
- EffectivePotential class creation
- Reference energy calculation
- Displacement calculation
- Strain calculation
- Harmonic energy/forces
- Full evaluation workflow
"""

import pytest
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from pymultibinit.pyeffpot import (
    read_ddb,
    build_supercell,
    EffectivePotential,
)
from pymultibinit.pyeffpot.datastructures import CrystalInfo, IFCData, UnitcellData


class TestEffectivePotential:
    """Test EffectivePotential evaluation."""
    
    @pytest.fixture
    def bto_ddb_path(self):
        """Path to BaTiO3 DDB test file."""
        return Path(__file__).parent.parent.parent / 'abinit/tests/v9/Input/BTO.DDB'
    
    @pytest.fixture
    def simple_supercell(self):
        """Create a simple supercell for testing."""
        # Simple 2-atom unitcell
        crystal = CrystalInfo(
            natom=2,
            ntypat=2,
            rprimd=np.eye(3) * 7.0,
            xred=np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
            xcart=np.array([[0.0, 0.0, 0.0], [3.5, 3.5, 3.5]]),
            typat=np.array([1, 2]),
            amu=np.array([100.0, 50.0]),
            znucl=np.array([50, 25])
        )
        
        # Simple IFCs
        ifcs = IFCData(
            nrpt=1,
            cell=np.zeros((3, 1), dtype=int),
            atmfrc=np.zeros((3, 2, 3, 2, 1)),
            short_atmfrc=np.zeros((3, 2, 3, 2, 1))
        )
        
        # Add diagonal force constants
        ifcs.atmfrc[:, 0, :, 0, 0] = np.eye(3) * 0.1
        ifcs.atmfrc[:, 1, :, 1, 0] = np.eye(3) * 0.1
        ifcs.short_atmfrc = ifcs.atmfrc.copy()
        
        unitcell = UnitcellData(
            crystal=crystal,
            energy=-100.0,
            ifcs=ifcs,
            epsilon_inf=np.eye(3),
            zeff=np.zeros((3, 3, 2))
        )
        
        return build_supercell(unitcell, (2, 2, 2))
    
    def test_potential_creation(self, simple_supercell):
        """Test EffectivePotential creation."""
        potential = EffectivePotential(simple_supercell)
        
        assert potential.supercell is simple_supercell
        assert potential._reference_positions.shape == (16, 3)
        assert potential._reference_lattice.shape == (3, 3)
    
    def test_from_files(self, bto_ddb_path):
        """Test creating potential from DDB file."""
        potential = EffectivePotential.from_files(
            str(bto_ddb_path),
            ncell=(2, 2, 2)
        )
        
        assert isinstance(potential, EffectivePotential)
        assert potential.supercell.natom_sc == 40
    
    def test_reference_energy(self, simple_supercell):
        """Test reference energy calculation."""
        potential = EffectivePotential(simple_supercell)
        
        e_ref = potential._compute_reference_energy()
        
        # Should be ncells * E0 = 8 * (-100) = -800
        expected = 8 * (-100.0)
        assert abs(e_ref - expected) < 1e-10
    
    def test_displacement_calculation(self, simple_supercell):
        """Test displacement calculation."""
        potential = EffectivePotential(simple_supercell)
        
        # Zero displacement
        xcart = potential._reference_positions.copy()
        u = potential._compute_displacements(xcart)
        assert np.allclose(u, 0.0)
        
        # Small displacement
        xcart_displaced = xcart + np.array([0.1, 0.0, 0.0])
        u = potential._compute_displacements(xcart_displaced)
        assert np.allclose(u[0], [0.1, 0.0, 0.0])
    
    def test_strain_calculation(self, simple_supercell):
        """Test strain calculation."""
        potential = EffectivePotential(simple_supercell)
        
        # No strain
        rprimd = potential._reference_lattice.copy()
        strain = potential._compute_strain(rprimd)
        assert np.allclose(strain, 0.0, atol=1e-10)
        
        # Uniform expansion
        rprimd_expanded = rprimd * 1.01  # 1% expansion
        strain = potential._compute_strain(rprimd_expanded)
        # Diagonal should be approximately 0.01 for small strain
        assert strain[0, 0] > 0
        assert strain[1, 1] > 0
        assert strain[2, 2] > 0
    
    def test_harmonic_evaluation(self, simple_supercell):
        """Test harmonic IFC evaluation."""
        potential = EffectivePotential(simple_supercell)
        
        # Small displacement
        displacements = np.zeros((16, 3))
        displacements[0, 0] = 0.01  # Small x-displacement on first atom
        
        e_harm, f_harm, s_harm = potential._evaluate_harmonic(displacements)
        
        # Energy should be positive for displacement from equilibrium
        assert e_harm >= 0
        
        # Force should be restoring (negative gradient)
        assert f_harm.shape == (16, 3)
        
        # Stress should be 3x3
        assert s_harm.shape == (3, 3)
    
    def test_full_evaluation(self, simple_supercell):
        """Test full energy/forces/stress evaluation."""
        potential = EffectivePotential(simple_supercell)
        
        # Evaluate at equilibrium
        energy, forces, stress = potential.evaluate()
        
        # At equilibrium, forces should be small (but not exactly zero due to ASR)
        assert isinstance(energy, float)
        assert forces.shape == (16, 3)
        assert stress.shape == (3, 3)
        
        # Energy should be close to reference energy at equilibrium
        e_ref = potential._compute_reference_energy()
        assert abs(energy - e_ref) < 0.1  # Small harmonic contribution
    
    def test_evaluation_with_displacement(self, simple_supercell):
        """Test evaluation with displaced atoms."""
        potential = EffectivePotential(simple_supercell)
        
        # Displace atoms
        xcart = potential._reference_positions.copy()
        xcart[0] += np.array([0.1, 0.0, 0.0])
        
        energy, forces, stress = potential.evaluate(xcart)
        
        # Energy should be higher than reference
        e_ref = potential._compute_reference_energy()
        assert energy > e_ref
        
        # Force should be non-zero
        assert not np.allclose(forces, 0.0)
    
    def test_energy_only_evaluation(self, simple_supercell):
        """Test energy-only evaluation."""
        potential = EffectivePotential(simple_supercell)
        
        energy = potential.evaluate_energy_only()
        assert isinstance(energy, float)
    
    def test_forces_only_evaluation(self, simple_supercell):
        """Test forces-only evaluation."""
        potential = EffectivePotential(simple_supercell)
        
        forces = potential.evaluate_forces_only()
        assert forces.shape == (16, 3)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
