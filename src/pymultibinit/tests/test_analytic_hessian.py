"""Tests for analytic second-derivative blocks (pyeffpot.second_derivatives).

Validates HessianBlocks / elastic_affine / coupling_fixed_xcart against
finite differences of EffectivePotential itself on the fitted BaTiO3 model
(rattle_fw_s0.25). Conventions: Bohr/Hartree, engineering Voigt strain
(xx,yy,zz,yz,xz,xy), atom-major flat indices (3*atom + dir).

Skips (with reason) when the reference model files are absent.
"""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pymultibinit.potential import BOHR_TO_ANGSTROM, HARTREE_TO_EV
from pymultibinit.pyeffpot.potential import EffectivePotential
from pymultibinit.pyeffpot.second_derivatives import (
    analytic_blocks,
    coupling_fixed_xcart,
    elastic_affine,
)

FW = Path("/home/hexu/projects/atomchain_dev/atomchain/examples/"
          "08_training_set_strategies_batio3/batio3_rattle_fw_s0.25")
DDB = FW / "ddb" / "model.ddb"
XML = FW / "model" / "fitted.nc"

VOIGT_PAIRS = ((0, 0), (1, 1), (2, 2), (2, 1), (2, 0), (1, 0))


@pytest.fixture(scope="module")
def potential():
    ddb = os.environ.get("PMB_TEST_DDB", str(DDB))
    xml = os.environ.get("PMB_TEST_XML", str(XML))
    if not (Path(ddb).exists() and Path(xml).exists()):
        pytest.skip(f"reference model not found: {ddb}")
    pot = EffectivePotential.from_files(ddb_file=ddb, xml_file=xml, ncell=(2, 2, 2))
    pot._use_jax = False  # exact numpy path for FD comparisons
    return pot


@pytest.fixture(scope="module")
def config(potential):
    rng = np.random.default_rng(42)
    ref = potential.supercell.crystal_sc.xcart
    rprimd0 = potential._reference_lattice
    u = rng.normal(scale=0.05, size=(potential.supercell.natom_sc, 3))
    xcart = ref + u
    eta = np.zeros((3, 3))
    return u, eta, xcart, rprimd0


def test_blocks_shapes_and_symmetry(potential, config):
    u, eta, xcart, rprimd0 = config
    bl = analytic_blocks(potential, u, eta)
    n3 = 3 * potential.supercell.natom_sc
    assert bl.ifc.shape == (n3, n3)
    assert bl.elastic_fixed_u.shape == (6, 6)
    assert bl.coupling.shape == (6, n3)
    assert bl.forces.shape == u.shape
    np.testing.assert_allclose(bl.ifc, bl.ifc.T, atol=1e-10)
    np.testing.assert_allclose(bl.elastic_fixed_u, bl.elastic_fixed_u.T, atol=1e-10)
    with pytest.raises(ValueError):
        analytic_blocks(potential, u[:, :2], eta)


def test_forces_match_evaluator(potential, config):
    """HessianBlocks.forces (F = -dE/du) equals the evaluator forces."""
    u, eta, xcart, rprimd0 = config
    bl = analytic_blocks(potential, u, eta)
    F = potential.evaluate(xcart, rprimd0)[1]
    assert np.abs(F - bl.forces).max() < 1e-9


def test_ifc_matches_force_fd(potential, config):
    """d2E/du du vs central differences of evaluator forces (O(h^2))."""
    u, eta, xcart, rprimd0 = config
    bl = analytic_blocks(potential, u, eta)
    h = 1e-5
    rng = np.random.default_rng(7)
    cols = rng.choice(3 * potential.supercell.natom_sc, size=6, replace=False)
    for c in cols:
        n, mu = divmod(int(c), 3)
        xp = xcart.copy()
        xp[n, mu] += h
        xm = xcart.copy()
        xm[n, mu] -= h
        fd = (potential.evaluate(xp, rprimd0)[1]
              - potential.evaluate(xm, rprimd0)[1]) / (2 * h)
        np.testing.assert_allclose(fd, -bl.ifc[:, c].reshape(-1, 3), atol=1e-6)


def test_coupling_matches_force_fd_fixed_xcart(potential, config):
    """Force response to strain at fixed xcart.

    dF/deta_nu = -coupling[nu] + ifc @ flat(x_ref @ E_nu @ R^T)
    (min-image displacement frame; see coupling_fixed_xcart docstring).
    """
    u, eta, xcart, rprimd0 = config
    bl = analytic_blocks(potential, u, eta)
    expected = coupling_fixed_xcart(potential, u, eta, bl, rprimd0)
    h = 1e-5
    for nu, (a, b) in enumerate(VOIGT_PAIRS):
        eps = np.zeros((3, 3))
        eps[a, b] += 0.5 * h
        eps[b, a] += 0.5 * h
        fd = (potential.evaluate(xcart, rprimd0 + eps @ rprimd0)[1]
              - potential.evaluate(xcart, rprimd0 - eps @ rprimd0)[1]) / (2 * h)
        np.testing.assert_allclose(fd, expected[nu].reshape(-1, 3), atol=1e-5)


def test_elastic_affine_matches_energy_fd(potential, config):
    """Clamped-ion C0 vs second differences of the energy on the affine path.

    Elastic constants are defined at the reference strain (eta = 0); the
    affine path is xcart -> xcart + eps @ xcart, lattice -> lattice +
    eps @ lattice (scale_atoms semantics), on which u(eps) = u(I+eps)^T
    exactly (min-image wrap inactive at these amplitudes).
    """
    u, eta, xcart, rprimd0 = config
    C0, chain = elastic_affine(potential, u, eta)
    h = 3e-3
    scale = float(np.abs(C0).max())

    def energy(nu, om, s1, s2):
        a, b = VOIGT_PAIRS[nu]
        g, d = VOIGT_PAIRS[om]
        e = np.zeros((3, 3))
        e[a, b] += 0.5 * h * s1
        e[b, a] += 0.5 * h * s1
        e[g, d] += 0.5 * h * s2
        e[d, g] += 0.5 * h * s2
        rp = rprimd0 + e @ rprimd0
        xc = xcart + (e @ xcart.T).T
        return potential.evaluate_energy_only(xc, rp)

    for nu in range(6):
        for om in range(nu, 6):
            if nu == om:
                fd = (energy(nu, om, +1, 0) - 2 * energy(nu, om, 0, 0)
                      + energy(nu, om, -1, 0)) / h ** 2
            else:
                fd = (energy(nu, om, +1, +1) - energy(nu, om, +1, -1)
                      - energy(nu, om, -1, +1) + energy(nu, om, -1, -1)) / (4 * h ** 2)
            assert abs(C0[nu, om] - fd) < 1e-5 * scale, (nu, om, C0[nu, om], fd)


def test_elastic_fixed_u_at_reference_matches_energy_fd(potential):
    """At u = 0 the affine chain terms vanish, so C0 == elastic_fixed_u.

    Both the Nc * C elastic channel and the fitted pure-strain terms
    (which carry the ncell-origin sum: prod_disp = ones(ncells)) are
    checked together against the energy second difference.
    """
    xcart = potential.supercell.crystal_sc.xcart
    rprimd0 = potential._reference_lattice
    zero_u = np.zeros_like(xcart)
    bl = analytic_blocks(potential, zero_u, np.zeros((3, 3)))
    C0, _ = elastic_affine(potential, zero_u, np.zeros((3, 3)))
    np.testing.assert_allclose(C0, bl.elastic_fixed_u, atol=1e-12)

    h = 1e-3
    scale = float(np.abs(bl.elastic_fixed_u).max())
    for nu in range(6):
        a, b = VOIGT_PAIRS[nu]
        e = np.zeros((3, 3))
        e[a, b] += 0.5 * h
        e[b, a] += 0.5 * h
        rp_p = rprimd0 + e @ rprimd0
        rp_m = rprimd0 - e @ rprimd0
        xc_p = xcart + (e @ xcart.T).T
        xc_m = xcart - (e @ xcart.T).T
        fd = (potential.evaluate_energy_only(xc_p, rp_p)
              - 2 * potential.evaluate_energy_only(xcart, rprimd0)
              + potential.evaluate_energy_only(xc_m, rp_m)) / h ** 2
        assert abs(C0[nu, nu] - fd) < 1e-5 * scale, (nu, C0[nu, nu], fd)



def test_multibinit_potential_wrapper_units_roundtrip():
    """Story 3: Angstrom/eV wrapper matches direct Bohr/Hartree blocks."""
    pytest.importorskip("ase")
    from pymultibinit import MultibinitCalculator

    calc = MultibinitCalculator.from_pyeffpot(
        ddb_file=os.environ.get("PMB_TEST_DDB", str(DDB)),
        xml_file=os.environ.get("PMB_TEST_XML", str(XML)),
        ncell=(2, 2, 2), match_tolerance=0.35)
    pot = calc.potential
    ref_ang = potential_ref(pot)
    rng = np.random.default_rng(3)
    pos = ref_ang[0] + rng.normal(scale=0.03, size=ref_ang[0].shape)
    lat = ref_ang[1]

    bl_ev = pot.analytic_blocks(pos, lat)
    eff = pot._pyeffpot_potential
    bl_ha = analytic_blocks(eff, u_of(pot, pos, lat), np.zeros((3, 3)))
    ff = HARTREE_TO_EV / BOHR_TO_ANGSTROM
    assert np.abs(bl_ev.forces - bl_ha.forces * ff).max() < 1e-12 * np.abs(bl_ha.forces).max()
    assert np.abs(bl_ev.ifc - bl_ha.ifc * ff / BOHR_TO_ANGSTROM).max() < 1e-12 * np.abs(bl_ha.ifc).max()
    assert np.abs(bl_ev.coupling - bl_ha.coupling * ff).max() < 1e-12 * np.abs(bl_ha.coupling).max()
    assert np.abs(bl_ev.elastic_fixed_u - bl_ha.elastic_fixed_u * HARTREE_TO_EV).max() < 1e-12
    _, F_ev, _ = pot.evaluate(pos, lat)
    assert np.abs(bl_ev.forces - F_ev).max() < 1e-6


def potential_ref(pot):
    eff = pot._pyeffpot_potential
    return (eff.supercell.crystal_sc.xcart * BOHR_TO_ANGSTROM,
            eff._reference_lattice.T * BOHR_TO_ANGSTROM)


def u_of(pot, pos_ang, lat_ang):
    eff = pot._pyeffpot_potential
    return eff._compute_displacements(
        pos_ang / BOHR_TO_ANGSTROM, (lat_ang / BOHR_TO_ANGSTROM).T)


class _RecordingCalculator:
    def __init__(self, calculator=None):
        self.calculator = calculator
        self.atoms = None
        self.blocks = None

    def get_analytic_blocks(self, atoms):
        self.atoms = atoms.copy()
        if self.calculator is None:
            n3 = 3 * len(atoms)
            ifc = np.arange(n3 * n3, dtype=float).reshape(n3, n3)
            self.blocks = SimpleNamespace(ifc=ifc)
        else:
            self.blocks = self.calculator.get_analytic_blocks(atoms)
        return self.blocks


def test_calculate_analytic_phonon_preserves_full_fc_order_and_magmoms():
    ase = pytest.importorskip("ase")
    pytest.importorskip("phonopy")
    from pymultibinit.phonon import calculate_analytic_phonon

    atoms = ase.Atoms(
        "Fe2",
        scaled_positions=((0, 0, 0), (0.5, 0.5, 0.5)),
        cell=np.eye(3) * 3,
        pbc=True,
    )
    atoms.set_initial_magnetic_moments((2.0, -2.0))
    calculator = _RecordingCalculator()
    phonon = calculate_analytic_phonon(
        atoms, calculator, supercell_matrix=np.diag((2, 1, 1))
    )

    n = len(phonon.supercell)
    expected = calculator.blocks.ifc.reshape(n, 3, n, 3).transpose(0, 2, 1, 3)
    assert phonon.force_constants.shape == (n, n, 3, 3)
    np.testing.assert_array_equal(phonon.force_constants, expected)
    np.testing.assert_array_equal(
        calculator.atoms.get_scaled_positions(), phonon.supercell.scaled_positions
    )
    np.testing.assert_array_equal(
        calculator.atoms.get_initial_magnetic_moments(),
        phonon.supercell.magnetic_moments,
    )


def test_calculate_analytic_phonon_validates_matrices_and_calculator():
    ase = pytest.importorskip("ase")
    from pymultibinit.phonon import calculate_analytic_phonon

    atoms = ase.Atoms("H", cell=np.eye(3), pbc=True)
    with pytest.raises(ValueError, match="supercell_matrix"):
        calculate_analytic_phonon(atoms, _RecordingCalculator(), np.ones(3))
    with pytest.raises(ValueError, match="primitive_matrix"):
        calculate_analytic_phonon(
            atoms, _RecordingCalculator(), primitive_matrix=np.ones(3)
        )
    with pytest.raises(TypeError, match="get_analytic_blocks"):
        calculate_analytic_phonon(atoms, object())


def test_calculate_analytic_phonon_batio3_exact_supercell_blocks(potential):
    ase = pytest.importorskip("ase")
    pytest.importorskip("phonopy")
    from pymultibinit import MultibinitCalculator, calculate_analytic_phonon

    calculator = MultibinitCalculator.from_pyeffpot(
        ddb_file=os.environ.get("PMB_TEST_DDB", str(DDB)),
        xml_file=os.environ.get("PMB_TEST_XML", str(XML)),
        ncell=(2, 2, 2),
        match_tolerance=0.35,
    )
    try:
        crystal = (
            calculator.potential._pyeffpot_potential.supercell.unitcell.crystal
        )
        numbers = np.rint(crystal.znucl[crystal.typat.astype(int) - 1]).astype(int)
        atoms = ase.Atoms(
            numbers=numbers,
            positions=crystal.xcart * BOHR_TO_ANGSTROM,
            cell=crystal.rprimd.T * BOHR_TO_ANGSTROM,
            pbc=True,
        )
        atoms.set_masses(crystal.amu[crystal.typat.astype(int) - 1])
        recording_calculator = _RecordingCalculator(calculator)

        phonon = calculate_analytic_phonon(
            atoms,
            recording_calculator,
            supercell_matrix=np.diag((2, 2, 2)),
        )

        n = len(phonon.supercell)
        expected = recording_calculator.blocks.ifc.reshape(
            n, 3, n, 3
        ).transpose(0, 2, 1, 3)
        assert phonon.force_constants.shape == (n, n, 3, 3)
        np.testing.assert_array_equal(phonon.force_constants, expected)
        np.testing.assert_allclose(
            recording_calculator.atoms.get_scaled_positions(),
            phonon.supercell.scaled_positions,
            atol=1e-15,
            rtol=0.0,
        )
    finally:
        calculator.close()
