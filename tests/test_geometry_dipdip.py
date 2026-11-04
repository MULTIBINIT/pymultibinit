from pathlib import Path

import numpy as np
import pytest

from pymultibinit import MultibinitPotential


DDB = Path(__file__).resolve().parents[1] / "examples/BaHfO3_example/BaHfO3_DDB"


@pytest.fixture(scope="module")
def models():
    if not DDB.exists():
        pytest.skip("BaHfO3 DDB fixture missing")
    harmonic = MultibinitPotential.from_pyeffpot(str(DDB), ncell=(1, 1, 1))
    geometry = MultibinitPotential.from_pyeffpot(
        str(DDB), ncell=(1, 1, 1), dipdip_model="geometry"
    )
    positions, cell, _ = geometry.get_supercell_structure()
    return harmonic, geometry, positions, cell


def test_geometry_contribution_is_conservative_and_matches_harmonic_limit(models):
    harmonic, geometry, reference, cell = models
    displacement = np.array([
        [0.08, -0.03, 0.01], [-0.05, 0.04, 0.02],
        [0.02, 0.01, -0.03], [0.00, -0.02, 0.05], [-0.05, 0.00, -0.05],
    ])

    def dd(pos, lattice=cell):
        return geometry.evaluate_contributions(pos, lattice, skip_atom_matching=True)["dipdip"]

    small = reference + 1e-4 * displacement
    e_h, f_h, _ = harmonic.evaluate_contributions(small, cell, skip_atom_matching=True)["dipdip"]
    e_g, f_g, _ = dd(small)
    np.testing.assert_allclose(e_g, e_h, rtol=1e-3, atol=1e-12)
    np.testing.assert_allclose(f_g, f_h, rtol=1e-3, atol=1e-7)

    positions = reference + displacement
    energy, forces, stress = dd(positions)
    step = 1e-5
    delta = np.zeros_like(positions)
    delta[0, 0] = step
    numeric_force = -(dd(positions + delta)[0] - dd(positions - delta)[0]) / (2 * step)
    np.testing.assert_allclose(forces[0, 0], numeric_force, rtol=2e-4, atol=2e-6)

    plus = np.diag([1 + step, 1, 1])
    minus = np.diag([1 - step, 1, 1])
    numeric_stress = (dd(positions @ plus, cell @ plus)[0] - dd(positions @ minus, cell @ minus)[0]) / (2 * step * abs(np.linalg.det(cell)))
    np.testing.assert_allclose(stress[0], numeric_stress, rtol=5e-4, atol=2e-6)

    shifted_energy, shifted_force, _ = dd(positions + np.array([0.04, -0.03, 0.02]))
    np.testing.assert_allclose(shifted_energy, energy, atol=1e-8)
    np.testing.assert_allclose(shifted_force, forces, atol=1e-7)
    harmonic_energy = harmonic.evaluate_contributions(positions, cell, skip_atom_matching=True)["dipdip"][0]
    assert abs(energy - harmonic_energy) > 1e-8


def test_geometry_contributions_sum_and_hessian_is_rejected(models):
    _, geometry, positions, cell = models
    positions = positions.copy()
    positions[0] += [0.03, -0.01, 0.02]
    parts = geometry.evaluate_contributions(positions, cell, skip_atom_matching=True)
    total = geometry.evaluate(positions, cell, skip_atom_matching=True)
    for component in range(3):
        np.testing.assert_allclose(
            sum(value[component] for value in parts.values()), total[component], atol=1e-9
        )
    with pytest.raises(NotImplementedError, match="Hessian|hessian|analytic"):
        geometry.analytic_blocks(positions, cell, skip_atom_matching=True)


def test_geometry_correction_stress_under_shear(models):
    harmonic, geometry, reference, cell = models
    positions = reference.copy()
    positions[0] += [0.08, -0.04, 0.02]
    positions[1] += [-0.03, 0.02, -0.01]

    def correction(pos, lat):
        g = geometry.evaluate_contributions(pos, lat, skip_atom_matching=True)["dipdip"]
        h = harmonic.evaluate_contributions(pos, lat, skip_atom_matching=True)["dipdip"]
        return g[0] - h[0], g[2] - h[2]

    _, stress = correction(positions, cell)
    step = 1e-5
    shear = np.zeros((3, 3))
    shear[0, 1] = shear[1, 0] = step / 2
    plus, minus = np.eye(3) + shear, np.eye(3) - shear
    ep = correction(positions @ plus, cell @ plus)[0]
    em = correction(positions @ minus, cell @ minus)[0]
    numeric = (ep - em) / (2 * step * abs(np.linalg.det(cell)))
    np.testing.assert_allclose(stress[5], numeric, rtol=1e-4, atol=1e-8)


def test_skew_anisotropic_ewald_derivatives_and_splitting_invariance(monkeypatch):
    from pymultibinit.pyeffpot import nonlinear_dipdip as dipole

    lattice = np.array([[8., 1., 0.], [0., 9., .5], [0., 0., 10.]])
    positions = np.array([[1., 1., 1.], [4., 4., 3.], [2., 5., 6.]])
    displacement = np.array([[.1, .02, -.04], [-.05, .03, .02], [-.05, -.05, .02]])
    zeff = np.array([np.eye(3), -np.eye(3), .3 * np.eye(3)])
    epsilon = np.diag([3., 4., 5.])

    def evaluate(pos, cell, disp):
        return dipole.evaluate_geometry_dipoles(pos, cell, disp, zeff, epsilon)

    energy, forces, stress = evaluate(positions, lattice, displacement)
    h = 1e-5
    delta = np.zeros_like(positions)
    delta[0, 0] = h
    force_fd = -(evaluate(positions + delta, lattice, displacement + delta)[0]
                 - evaluate(positions - delta, lattice, displacement - delta)[0]) / (2 * h)
    np.testing.assert_allclose(forces[0, 0], force_fd, rtol=1e-6, atol=1e-10)
    strain = np.zeros((3, 3))
    strain[0, 0] = h
    plus, minus = np.eye(3) + strain, np.eye(3) - strain
    ep = evaluate(positions @ plus.T, plus @ lattice, displacement @ plus.T)[0]
    em = evaluate(positions @ minus.T, minus @ lattice, displacement @ minus.T)[0]
    stress_fd = (ep - em) / (2 * h * abs(np.linalg.det(lattice)))
    np.testing.assert_allclose(stress[0, 0], stress_fd, rtol=1e-6, atol=1e-12)

    monkeypatch.setattr(dipole, "_EWALD_ETA", 0.7)
    alternative = evaluate(positions, lattice, displacement)
    np.testing.assert_allclose(alternative[0], energy, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(alternative[1], forces, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(alternative[2], stress, rtol=1e-9, atol=1e-12)


def test_pair_dipoles_approach_vacuum_interaction():
    from pymultibinit.pyeffpot.nonlinear_dipdip import evaluate_geometry_dipoles

    cell = np.eye(3) * 15
    positions = np.array([[5., 7.5, 7.5], [7., 7.5, 7.5]])
    displacement = np.array([[.1, 0., 0.], [-.1, 0., 0.]])
    charges = np.array([np.eye(3), -np.eye(3)])

    def energy(z):
        return evaluate_geometry_dipoles(positions, cell, displacement, z, np.eye(3))[0]

    both = energy(charges)
    first = energy(np.array([charges[0], np.zeros((3, 3))]))
    second = energy(np.array([np.zeros((3, 3)), charges[1]]))
    # Parallel dipoles on their common axis attract: -2 p^2/r^3.
    np.testing.assert_allclose(both - first - second, -2 * .1**2 / 2**3, rtol=0.01)
