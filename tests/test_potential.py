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
    set_anharmonic_coeffs,
    EffectivePotential,
)
from pymultibinit.pyeffpot.datastructures import CrystalInfo, IFCData, UnitcellData
from pymultibinit.pyeffpot.xml_parser import PolynomialCoefficient, PolynomialTerm


class TestEffectivePotential:
    """Test EffectivePotential evaluation."""
    
    @pytest.fixture
    def bto_ddb_path(self):
        """Path to BaHfO3 DDB test file."""
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

    def test_batio3_ddb_elastic_response_is_nonzero(self):
        ddb_path = Path(__file__).parent.parent / "examples/BaHfO3_example/BaHfO3_DDB"
        unitcell = read_ddb(str(ddb_path))
        supercell = build_supercell(unitcell, (2, 2, 2))
        potential = EffectivePotential(supercell)
        strained_lattice = potential._reference_lattice * 1.001
        xcart = supercell.crystal_sc.xred @ strained_lattice.T

        energy0, _, stress0 = potential.evaluate(supercell.crystal_sc.xcart, potential._reference_lattice)
        energy1, forces1, stress1 = potential.evaluate(xcart, strained_lattice)

        assert np.linalg.norm(unitcell.elastic_constants) > 0.0
        assert np.linalg.norm(unitcell.strain_coupling) >= 0.0
        assert abs(energy1 - energy0) > 0.0
        assert np.linalg.norm(stress1) > np.linalg.norm(stress0)
        assert np.all(np.isfinite(forces1))
     
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

    def test_displacement_pbc_wrapping(self, simple_supercell):
        """Regression test: displacements must be PBC-wrapped to the minimum image.

        Bug: ``_compute_displacements`` computed ``xcart - ref_xred @ rprimd.T``
        without applying the minimum-image convention. When an atom drifts
        across a cell boundary (e.g., fractional position 0.99 vs. reference
        fractional position 0.0), the raw Cartesian displacement was
        ``~0.99 * lattice`` instead of ``~-0.01 * lattice``. This produced
        huge spurious forces in MD trajectories where atoms naturally wrap.

        Reference: BFO trajectory in ``mix_mb_deepmd/prefer.nc`` produced
        max |F| ~800 eV/Å at frame 0 because Bi atoms at fractional
        x=0.999999 were matched to ref atoms at x=0.0 and given a 7.7 Å
        displacement in Cartesian.
        """
        potential = EffectivePotential(simple_supercell)
        ref_pos = potential._reference_positions
        lattice = potential._reference_lattice

        xcart = ref_pos.copy()
        xcart[0] = ref_pos[0] + 0.99 * lattice[0]
        u = potential._compute_displacements(xcart, lattice)

        expected_dx = -0.01 * lattice[0]
        assert np.allclose(u[0], expected_dx, atol=1e-10), (
            f"PBC-wrapped displacement for atom 0 should be ~{expected_dx}, "
            f"got {u[0]}"
        )
        assert np.allclose(u[1:], 0.0, atol=1e-10)

        # simple_supercell IFC = 0.1 Ha/Bohr^2, lattice = 7 Å.
        # Wrapped disp ~0.07 Å -> |F| ~0.013 Ha/Bohr. Buggy unwrapped: ~99x larger.
        _, forces, _ = potential.evaluate(xcart, lattice)
        assert np.abs(forces).max() < 1.0, (
            f"Forces should be small for a PBC-wrapped displacement; "
            f"got max |F| = {np.abs(forces).max()} Ha/Bohr."
        )

    def test_displacement_pbc_wrapping_negative_boundary(self, simple_supercell):
        """Regression test companion: wrap across the -x boundary."""
        potential = EffectivePotential(simple_supercell)
        ref_pos = potential._reference_positions
        lattice = potential._reference_lattice

        xcart = ref_pos.copy()
        xcart[0] = ref_pos[0] - 1.01 * lattice[0]
        u = potential._compute_displacements(xcart, lattice)

        assert np.allclose(u[0], -0.01 * lattice[0], atol=1e-10), (
            f"PBC-wrapped displacement for atom 0 should be ~-0.01*lattice[0], "
            f"got {u[0]}"
        )

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

    def test_homogeneous_strain_does_not_create_internal_displacements(self, simple_supercell):
        """Pure lattice deformation should not be interpreted as atom shuffling."""
        potential = EffectivePotential(simple_supercell)
        strained_lattice = potential._reference_lattice * 0.99
        xcart = potential.supercell.crystal_sc.xred @ strained_lattice.T

        displacements = potential._compute_displacements(xcart, strained_lattice)

        np.testing.assert_allclose(displacements, 0.0, atol=1e-12)
    
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

    def test_harmonic_stress_includes_du_delta_force_correction(self, simple_supercell):
        potential = EffectivePotential(simple_supercell)
        displacements = np.zeros((simple_supercell.natom_sc, 3), dtype=float)
        displacements[0, 0] = 0.2

        _, forces, stress = potential._evaluate_harmonic(displacements, np.zeros((3, 3)), potential._reference_lattice)

        expected = potential._finalize_stress(np.zeros(6), forces, displacements, np.zeros((3, 3)), potential._reference_lattice)
        np.testing.assert_allclose(stress, expected)
        assert stress[0, 0] != pytest.approx(0.0)
     
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

    def test_anharmonic_strain_stress_uses_current_volume_and_strain_scaling(self, simple_supercell):
        term = PolynomialTerm(weight=1.0, strains=[{"power": 1, "voigt": 1}])
        coeff = PolynomialCoefficient(number=1, value=3.0, text="eta_xx", terms=[term])
        set_anharmonic_coeffs(simple_supercell, [coeff])
        potential = EffectivePotential(simple_supercell)
        stretch = 1.2
        rprimd = potential._reference_lattice * stretch
        strain = potential._compute_strain(rprimd)
        displacements = np.zeros((simple_supercell.natom_sc, 3), dtype=float)

        energy, forces, stress = potential._evaluate_anharmonic(displacements, strain, rprimd)

        expected_volume = abs(np.linalg.det(rprimd))
        assert energy == pytest.approx(3.0 * 0.2 * simple_supercell.ncells)
        np.testing.assert_allclose(forces, 0.0)
        assert stress[0, 0] == pytest.approx(3.0 * simple_supercell.ncells * 1.2 / expected_volume)
        np.testing.assert_allclose(stress[1:, :], 0.0, atol=1e-14)

    def test_anharmonic_stress_includes_du_delta_force_correction(self, simple_supercell):
        term = PolynomialTerm(
            weight=1.0,
            displacements=[
                {
                    "atom_a": 0,
                    "atom_b": 1,
                    "direction": "x",
                    "power": 2,
                    "cell_a": [0, 0, 0],
                    "cell_b": [0, 0, 0],
                }
            ],
        )
        coeff = PolynomialCoefficient(number=1, value=1.0, text="u2", terms=[term])
        set_anharmonic_coeffs(simple_supercell, [coeff])
        potential = EffectivePotential(simple_supercell)
        displacements = np.zeros((simple_supercell.natom_sc, 3), dtype=float)
        displacements[0, 0] = 0.1
        displacements[1, 0] = 0.4

        _, forces, stress = potential._evaluate_anharmonic(displacements, np.zeros((3, 3)), potential._reference_lattice)

        raw_stress = np.zeros(6)
        expected = potential._finalize_stress(raw_stress, forces, displacements, np.zeros((3, 3)), potential._reference_lattice)
        np.testing.assert_allclose(stress, expected)
        assert stress[0, 0] != pytest.approx(0.0)

    def test_anharmonic_uses_fortran_displacement_orientation(self, simple_supercell):
        term = PolynomialTerm(
            weight=1.0,
            displacements=[
                {
                    "atom_a": 0,
                    "atom_b": 1,
                    "direction": "x",
                    "power": 1,
                    "cell_a": [0, 0, 0],
                    "cell_b": [0, 0, 0],
                }
            ],
        )
        coeff = PolynomialCoefficient(number=1, value=2.0, text="u", terms=[term])
        set_anharmonic_coeffs(simple_supercell, [coeff])
        potential = EffectivePotential(simple_supercell)
        displacements = np.zeros((simple_supercell.natom_sc, 3), dtype=float)
        displacements[0, 0] = 0.1
        displacements[1, 0] = 0.4

        energy, forces, _ = potential._evaluate_anharmonic(displacements, np.zeros((3, 3)), potential._reference_lattice)

        assert energy == pytest.approx(-0.6)
        np.testing.assert_allclose(forces[:, 0], np.tile([-2.0, 2.0], simple_supercell.ncells))
        np.testing.assert_allclose(forces[:, 1:], 0.0)

    def test_strain_coupling_stress_uses_multibinit_finalization(self, simple_supercell):
        potential = EffectivePotential(simple_supercell)
        potential._phonon_strain_matrices = [None] * 6
        matrix = np.eye(simple_supercell.natom_sc * 3)
        potential._phonon_strain_matrices[0] = matrix
        displacements = np.zeros((simple_supercell.natom_sc, 3), dtype=float)
        displacements[0, 0] = 0.2
        strain = np.zeros((3, 3), dtype=float)
        strain[0, 0] = 0.1

        energy, forces, stress = potential._evaluate_strain_coupling(strain, displacements, potential._reference_lattice)

        raw_stress = np.zeros(6)
        raw_stress[0] = 0.5 * np.dot(displacements.ravel(), displacements.ravel())
        expected = potential._finalize_stress(raw_stress, forces, displacements, strain, potential._reference_lattice)
        assert energy == pytest.approx((1.0 / 6.0) * 0.1 * 0.04)
        np.testing.assert_allclose(stress, expected)

    def test_elastic_energy_stress_and_internal_strain_coupling(self, simple_supercell):
        simple_supercell.unitcell.elastic_constants = np.zeros((6, 6), dtype=float)
        simple_supercell.unitcell.elastic_constants[0, 0] = 2.0
        simple_supercell.unitcell.strain_coupling = np.zeros((6, 3, simple_supercell.unitcell.natom), dtype=float)
        simple_supercell.unitcell.strain_coupling[0, 0, 0] = 4.0
        potential = EffectivePotential(simple_supercell)
        strain = np.zeros((3, 3), dtype=float)
        strain[0, 0] = 0.1
        displacements = np.zeros((simple_supercell.natom_sc, 3), dtype=float)
        displacements[0, 0] = 0.2

        energy, forces, stress = potential._evaluate_elastic(strain, displacements, potential._reference_lattice)

        raw_stress = np.zeros(6)
        raw_stress[0] = simple_supercell.ncells * 2.0 * 0.1 + 0.5 * 4.0 * 0.2
        expected_stress = potential._finalize_stress(raw_stress, forces, displacements, strain, potential._reference_lattice)
        assert energy == pytest.approx(0.5 * simple_supercell.ncells * 2.0 * 0.1**2 + 0.5 * 4.0 * 0.1 * 0.2)
        expected_forces = np.zeros_like(forces)
        expected_forces[0::simple_supercell.unitcell.natom, 0] = -0.5 * 4.0 * 0.1
        np.testing.assert_allclose(forces, expected_forces)
        np.testing.assert_allclose(stress, expected_stress)
     
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
