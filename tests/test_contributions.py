"""
Energy / force / stress decomposition API.

The total energy, forces and stress must split into per-term contributions
(reference, local harmonic IFCs, dipole-dipole, anharmonic, elastic,
strain-coupling) whose arrays sum back to the ASE getters exactly.

Run:
    pytest pymultibinit/tests/test_contributions.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ase import Atoms  # noqa: E402

from pymultibinit import MultibinitCalculator, Contributions  # noqa: E402
from pymultibinit.potential import MultibinitPotential  # noqa: E402

_DDB = Path(__file__).resolve().parents[1] / "examples/BaHfO3_example/BaHfO3_DDB"
_XML = Path(__file__).resolve().parents[1] / "examples/BaHfO3_example/BaHfO3.xml"
NCELL = (2, 2, 2)

needs_fixtures = pytest.mark.skipif(
    not _DDB.exists(), reason="BaHfO3 DDB fixture missing"
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _disable_jax(potential):
    """Force the deterministic NumPy anharmonic path.

    The decomposition is mathematically exact, but the JAX anharmonic backend
    is non-deterministic across two independent evaluations, which would make
    the per-term sum disagree with evaluate() at ~1e-7. Disabling JAX isolates
    decomposition correctness from backend non-determinism.
    """
    potential._pyeffpot_potential._use_jax = False
    return potential


@pytest.fixture(scope="module")
def pyeffpot_potential():
    return _disable_jax(MultibinitPotential.from_pyeffpot(
        ddb_file=str(_DDB), xml_file=str(_XML), ncell=NCELL, dipdip=True))


@pytest.fixture(scope="module")
def reference_atoms(pyeffpot_potential):
    """Equilibrium supercell with the calculator attached.

    Evaluating at equilibrium first locks the identity atom mapping on the
    shared potential, so subsequent displaced configurations (same atom
    order) never re-trigger atom matching.
    """
    atoms = pyeffpot_potential.export_supercell_to_ase()
    atoms.calc = MultibinitCalculator(pyeffpot_potential)
    atoms.get_potential_energy()  # cache identity mapping
    return atoms


def _displace_and_strain(atoms, seed=0, amp=0.05, strain=0.01):
    """Return a copy with random displacements + uniaxial strain applied."""
    rng = np.random.default_rng(seed)
    work = atoms.copy()
    work.calc = atoms.calc
    work.set_positions(work.get_positions() + rng.normal(scale=amp, size=work.positions.shape))
    cell = work.get_cell().array.copy()
    cell[0, 0] *= 1.0 + strain
    work.set_cell(cell, scale_atoms=True)
    return work


# --------------------------------------------------------------------------- #
# Sum invariant: contributions must reproduce the ASE getters
# --------------------------------------------------------------------------- #
@needs_fixtures
class TestSumInvariant:
    """Sum over terms == get_potential_energy / get_forces / get_stress."""

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_energy_forces_stress_sum(self, reference_atoms, seed):
        atoms = _displace_and_strain(reference_atoms, seed=seed)
        e = atoms.get_potential_energy()
        f = atoms.get_forces()
        s = atoms.get_stress(voigt=True)

        contrib = atoms.calc.get_contributions(atoms)

        # JAX is disabled in the fixture, so both the per-term sum and the
        # direct evaluate() use the deterministic NumPy path: the decomposition
        # must reproduce the ASE getters at machine precision.
        assert abs(contrib.total_energy() - e) < 1e-10
        assert np.allclose(contrib.total_forces(), f, rtol=0, atol=1e-10)
        assert np.allclose(contrib.total_stress(), s, rtol=0, atol=1e-10)

    def test_contributions_cached_in_results(self, reference_atoms):
        atoms = _displace_and_strain(reference_atoms, seed=3)
        atoms.calc.get_contributions(atoms)
        assert "contributions" in atoms.calc.results
        assert isinstance(atoms.calc.results["contributions"], Contributions)


# --------------------------------------------------------------------------- #
# Physical sanity of individual terms
# --------------------------------------------------------------------------- #
@needs_fixtures
class TestTermBehaviour:
    def test_expected_terms_present(self, reference_atoms):
        atoms = _displace_and_strain(reference_atoms)
        contrib = atoms.calc.get_contributions(atoms)
        # The three user-facing decomposition terms are always present here:
        for key in ("reference", "harmonic_local", "dipdip", "anharmonic"):
            assert key in contrib.terms, f"missing term {key!r}"

    def test_equilibrium_only_reference_energy(self, reference_atoms):
        """At zero displacement / strain the harmonic and anharmonic terms vanish."""
        atoms = reference_atoms.copy()
        atoms.calc = reference_atoms.calc
        contrib = atoms.calc.get_contributions(atoms)
        # Reference carries the constant offset; every other energy is ~0.
        for key in ("harmonic_local", "dipdip", "anharmonic", "elastic"):
            assert key in contrib.energy
            assert abs(contrib.energy[key]) < 1e-8, f"{key} nonzero at equilibrium"
        # Forces are all ~0 at equilibrium.
        for key in contrib.terms:
            assert np.max(np.abs(contrib.forces[key])) < 1e-9

    def test_harmonic_split_matches_combined(self, pyeffpot_potential, reference_atoms):
        """harmonic_local + dipdip == the combined harmonic phi matrix."""
        from pymultibinit.pyeffpot.potential import EffectivePotential

        atoms = _displace_and_strain(reference_atoms)
        ep = pyeffpot_potential._pyeffpot_potential
        assert isinstance(ep, EffectivePotential)

        pos_bohr = atoms.get_positions() / 0.529177210903
        lat_bohr = atoms.get_cell().array / 0.529177210903
        u = ep._compute_displacements(pos_bohr, lat_bohr.T)
        strain = ep._compute_strain(lat_bohr.T)

        e_local, f_local, _ = ep._evaluate_harmonic_with_phi(
            ep._phi_local, u, strain, lat_bohr.T)
        e_dd, f_dd, _ = ep._evaluate_harmonic_with_phi(
            ep._phi_dipdip, u, strain, lat_bohr.T)
        e_comb, f_comb, _ = ep._evaluate_harmonic_with_phi(
            ep._phi_matrix, u, strain, lat_bohr.T)

        assert abs((e_local + e_dd) - e_comb) < 1e-10
        assert np.max(np.abs((f_local + f_dd) - f_comb)) < 1e-10
        # phi_matrix == phi_local + phi_dipdip exactly
        assert np.allclose(ep._phi_matrix, ep._phi_local + ep._phi_dipdip)

    def test_dipdip_absent_when_disabled(self):
        """With dipdip=False there is no dipole-dipole term."""
        pot = MultibinitPotential.from_pyeffpot(
            ddb_file=str(_DDB), xml_file=None, ncell=NCELL, dipdip=False
        )
        atoms = pot.export_supercell_to_ase()
        atoms.calc = MultibinitCalculator(pot)
        atoms.get_potential_energy()  # cache identity mapping
        atoms = _displace_and_strain(atoms)
        contrib = atoms.calc.get_contributions(atoms)
        assert "dipdip" not in contrib.terms
        assert "harmonic_local" in contrib.terms


# --------------------------------------------------------------------------- #
# API surface across the three layers
# --------------------------------------------------------------------------- #
@needs_fixtures
class TestAPILayers:
    def test_calculator_returns_contributions(self, reference_atoms):
        atoms = _displace_and_strain(reference_atoms)
        contrib = atoms.calc.get_contributions(atoms)
        assert isinstance(contrib, Contributions)
        assert set(contrib.energy) == set(contrib.forces) == set(contrib.stress)
        assert set(contrib.terms) == set(contrib.energy)
        # force arrays have the right shape
        n = len(atoms)
        for f in contrib.forces.values():
            assert f.shape == (n, 3)
        for st in contrib.stress.values():
            assert st.shape == (6,)

    def test_potential_evaluate_contributions(self, pyeffpot_potential, reference_atoms):
        atoms = _displace_and_strain(reference_atoms)
        raw = pyeffpot_potential.evaluate_contributions(
            atoms.get_positions(), atoms.get_cell().array)
        assert isinstance(raw, dict)
        for value in raw.values():
            e, f, s = value
            assert isinstance(e, float)
            assert np.asarray(f).shape == (len(atoms), 3)
            assert np.asarray(s).shape == (6,)

    def test_pyeffpot_effective_potential_contributions(self, pyeffpot_potential):
        from pymultibinit.pyeffpot.potential import EffectivePotential

        ep = pyeffpot_potential._pyeffpot_potential
        assert isinstance(ep, EffectivePotential)
        raw = ep.evaluate_contributions()  # defaults to reference positions
        assert "reference" in raw
        # stress is a full 3x3 tensor at this layer
        for _, _, s in raw.values():
            assert np.asarray(s).shape == (3, 3)


# --------------------------------------------------------------------------- #
# Backend that cannot decompose must raise NotImplementedError
# --------------------------------------------------------------------------- #
class _NoDecomposePotential:
    """Stand-in for the CFFI / spawned backends: has evaluate(), no decomposition."""

    def evaluate(self, positions, cell):
        n = len(positions)
        return 0.0, np.zeros((n, 3)), np.zeros(6)

    def free(self):
        pass


def test_get_contributions_raises_for_undecomposable_backend():
    calc = MultibinitCalculator(_NoDecomposePotential())
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]], cell=np.eye(3) * 5)
    atoms.calc = calc
    with pytest.raises(NotImplementedError):
        calc.get_contributions(atoms)


@needs_fixtures
def test_closed_calculator_get_contributions_raises(reference_atoms):
    calc = MultibinitCalculator(reference_atoms.calc.potential)
    calc.close()
    with pytest.raises(RuntimeError):
        calc.get_contributions(reference_atoms)
