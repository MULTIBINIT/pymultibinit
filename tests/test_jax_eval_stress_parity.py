"""Regression tests for anharmonic strain-derivative stress in jax_eval.

``evaluate_jax`` previously used the column-wide maximum strain power when
accumulating d/d(eta) stress contributions. Any model mixing strain powers
(e.g. eta^1 coupling terms together with eta^2 pure-strain terms) had every
lower-power term's stress scaled by the max power (2x for the mixed 1/2 case)
while energies and forces stayed correct.

These tests pin per-term power behaviour via (a) jax-vs-numpy parity and
(b) finite-difference ground truth of the stress.
"""

from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")  # noqa: F841

from pymultibinit.pyeffpot.jax_eval import (  # noqa: E402
    compile_terms,
    evaluate_jax,
    evaluate_numpy,
)


def _term(disps, strains, weight=1.0):
    return SimpleNamespace(displacements=disps, strains=strains, weight=weight)


def _coeff(value, terms):
    return SimpleNamespace(value=value, terms=terms)


def _mixed_power_coeffs():
    """Coefficients mixing eta^1 and eta^2 strain factors on one slot."""
    d = {
        "atom_a": 0,
        "atom_b": 1,
        "direction": "x",
        "power": 3,
        "cell_a": (0, 0, 0),
        "cell_b": (0, 0, 0),
    }
    d2 = dict(d, power=2)
    return [
        # u^3 * eta_1  (strain power 1)
        _coeff(0.7, [_term([d], [{"voigt": 1, "power": 1}])]),
        # pure eta_1^2 (strain power 2, no displacement factors)
        _coeff(-1.3, [_term([], [{"voigt": 1, "power": 2}])]),
        # u^2 * eta_4 coupling with a second displacement factor
        _coeff(0.4, [_term([d2, dict(d2, atom_b=0)], [{"voigt": 4, "power": 1}])]),
    ]


@pytest.fixture(scope="module")
def compiled():
    return compile_terms(_mixed_power_coeffs(), (1, 1, 1), natom_uc=2)


@pytest.fixture(scope="module")
def displacements(compiled):
    rng = np.random.default_rng(0)
    return rng.normal(scale=0.05, size=(compiled.natom_sc, 3))



_STRAIN_CASES = [
    np.array([0.01, 0.0, 0.0, 0.02, 0.0, 0.0]),
    np.array([0.005, 0.003, 0.0, 0.0, 0.004, -0.002]),
    np.zeros(6),
]


@pytest.mark.parametrize("strain", _STRAIN_CASES, ids=["axial", "mixed", "zero"])
def test_jax_numpy_stress_parity(compiled, displacements, strain):
    e_jax, f_jax, s_jax = evaluate_jax(compiled, displacements, strain)
    e_np, f_np, s_np = evaluate_numpy(compiled, displacements, strain)
    assert e_jax == pytest.approx(e_np, rel=1e-6, abs=1e-9)
    np.testing.assert_allclose(f_jax, f_np, atol=1e-9)
    # With the old column-max-power bug this fails at exactly 2x.
    np.testing.assert_allclose(s_jax, s_np, rtol=1e-5, atol=1e-9)


@pytest.mark.parametrize("strain", _STRAIN_CASES, ids=["axial", "mixed", "zero"])
def test_stress_matches_finite_difference(compiled, displacements, strain):
    _, _, s_np = evaluate_numpy(compiled, displacements, strain)
    _, _, s_jax = evaluate_jax(compiled, displacements, strain)

    def energy(sv):
        return evaluate_numpy(compiled, displacements, sv)[0]

    h = 1e-6
    for v in range(6):
        svp = strain.copy()
        svm = strain.copy()
        svp[v] += h
        svm[v] -= h
        fd = (energy(svp) - energy(svm)) / (2.0 * h)
        assert s_np[v] == pytest.approx(fd, rel=1e-5, abs=1e-8), f"voigt {v + 1} numpy"
        assert s_jax[v] == pytest.approx(fd, rel=1e-4, abs=1e-7), f"voigt {v + 1} jax"
