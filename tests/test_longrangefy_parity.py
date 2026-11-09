"""Tests for the optional downstream longrangefy consistency harness.

Run:  pytest tests/test_longrangefy_parity.py -v

The comparison needs the optional ``longrangefy`` package.  Its matched core
is under active correction; a construction rejection is reported as a skip
with the dependency's own message, never silently converted into a parity
claim.  Mathematical alignment and the upstream-only route are tested without
that package.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pymultibinit.longrangefy_parity import (
    build_downstream_correction,
    compare_matched_correction,
    longrangefy_available,
    measure_phi0_symmetry_violation,
)
from pymultibinit.pyeffpot.datastructures import CrystalInfo, UnitcellData
from pymultibinit.pyeffpot.supercell_builder import _generate_full_bz_qpoints
from pymultibinit.potential import ANGSTROM_TO_BOHR

HAVE_LONGRANGEFY = longrangefy_available()
requires_longrangefy = pytest.mark.skipif(
    not HAVE_LONGRANGEFY, reason="optional longrangefy is not installed"
)


# ----------------------------------------------------------------------
# Deterministic synthetic fixtures
# ----------------------------------------------------------------------


def _spring_blocks(natom: int):
    """Short-range nearest-image spring kernel (real, finite range)."""
    translations = [
        (0, 0, 0), (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
    ]
    blocks = {t: np.zeros((natom, 3, natom, 3)) for t in translations}
    for atom in range(natom):
        blocks[(0, 0, 0)][atom, :, atom, :] = 6.0 * np.eye(3)
    for t in translations:
        blocks[t][0, :, 1, :] -= np.eye(3)
        blocks[t][1, :, 0, :] -= np.eye(3)
    return blocks


def _make_unitcell(
    cell: np.ndarray,
    xred: np.ndarray,
    *,
    reps=(2, 2, 2),
    dipole: bool = True,
    short_range: bool = True,
) -> UnitcellData:
    cell = np.asarray(cell, dtype=float)
    xred = np.asarray(xred, dtype=float)
    natom = len(xred)
    positions_ang = xred @ cell

    qbz = _generate_full_bz_qpoints(np.asarray(reps), 1, None)
    dynmat = np.zeros((len(qbz), natom, 3, natom, 3), dtype=complex)
    if short_range:
        blocks = _spring_blocks(natom)
        for index, q in enumerate(qbz):
            for translation, block in blocks.items():
                dynmat[index] += block * np.exp(
                    2j * np.pi * np.dot(q, translation)
                )

    zeff = None
    if dipole:
        zeff = np.stack([np.eye(3), -np.eye(3)])

    return UnitcellData(
        crystal=CrystalInfo(
            natom=natom,
            ntypat=2,
            rprimd=cell * ANGSTROM_TO_BOHR,
            xred=xred,
            xcart=positions_ang * ANGSTROM_TO_BOHR,
            typat=np.array([1, 2], dtype=int),
            amu=np.array([1.0, 1.0]),
            znucl=np.array([1, 2], dtype=int),
        ),
        energy=0.0,
        epsilon_inf=2.0 * np.eye(3),
        zeff=zeff,
        qpoints=qbz.copy(),
        dynmat=np.stack([dynmat.real, dynmat.imag], axis=-1),
        ngqpt=np.asarray(reps, dtype=int),
        nqshft=1,
        q1shft=np.zeros((1, 3)),
        symrel=None,
        tnons=None,
    )


@pytest.fixture(scope="module")
def cubic_unitcell():
    cell = np.eye(3) * 4.0
    xred = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    return _make_unitcell(cell, xred)


@pytest.fixture(scope="module")
def skew_unitcell():
    # A symmetric skew lattice keeps xcart = xred @ rprimd.T consistent with
    # the row-lattice convention while still exercising off-diagonal images.
    cell = np.array(
        [[4.0, 0.8, 0.3], [0.8, 4.5, 0.2], [0.3, 0.2, 5.0]]
    )
    xred = np.array([[0.05, 0.10, 0.02], [-0.45, -0.48, -0.49]])
    return _make_unitcell(cell, xred)


# ----------------------------------------------------------------------
# Optional-dependency boundary
# ----------------------------------------------------------------------


def test_import_and_absence_are_safe(monkeypatch):
    # Importing the harness itself must never require longrangefy.
    assert isinstance(longrangefy_available(), bool)

    monkeypatch.setitem(sys.modules, "longrangefy", None)
    monkeypatch.setitem(sys.modules, "longrangefy.matched", None)
    assert longrangefy_available() is False

    with pytest.raises(ImportError, match="optional 'longrangefy'"):
        build_downstream_correction(None)  # the optional import fails first


# ----------------------------------------------------------------------
# Alignment helpers and the upstream-only route
# ----------------------------------------------------------------------




def test_phi0_symmetry_measurement_is_explicit():
    reps = (2, 2, 2)
    qbz = _generate_full_bz_qpoints(np.asarray(reps), 1, None)
    grid_h = np.rint(qbz * np.asarray(reps)).astype(int) % np.asarray(reps)
    phi0 = np.zeros((len(qbz), 2, 3, 2, 3), dtype=complex)
    phi0[:, 0, 0, 1, 0] = 0.25  # deliberately non-Hermitian
    measured = measure_phi0_symmetry_violation(phi0, grid_h, reps)
    assert measured["max_abs"] == pytest.approx(0.25)
    assert measured["relative_max"] > 0.0
    assert np.isfinite(measured["scale"])


# ----------------------------------------------------------------------
# Numerical consistency (needs the optional package)
# ----------------------------------------------------------------------


def _compare_or_skip(*args, **kwargs):
    try:
        return compare_matched_correction(*args, **kwargs)
    except ValueError as exc:
        pytest.skip(f"longrangefy matched construction unavailable: {exc}")


@requires_longrangefy
def test_cubic_matching_and_target_consistency(cubic_unitcell):
    report = _compare_or_skip(cubic_unitcell, (2, 2, 2), (2, 2, 4))
    entry = report.energy_force["entries"][0]

    assert report.alignment["atom_order"].startswith("image-major")
    assert report.alignment["ordering_residual_angstrom"] < 1e-8
    assert report.alignment["upstream_phi0"].startswith("downstream-accepted")
    floor = report.alignment["expected_difference_floors"]
    assert abs(
        floor["unit_constants"]["relative_coulomb_difference"] + 8.7831989746e-9
    ) < 1e-15
    assert report.alignment["parity_claim"] is None
    assert report.matrices["unsplit"].max_abs < 1e-10
    assert report.matrices["split"].max_abs < 1e-7
    assert report.matrices["correction"].max_abs < 1e-7
    assert report.components["ewald"].max_abs < 1e-7
    assert report.components["short_range"].max_abs < 1e-7
    assert report.components["counterterm"].max_abs < 1e-7
    assert entry["correction"]["energy_abs_difference_eV"] < 1e-9
    assert entry["correction"]["force_max_abs_difference_eV_A"] < 1e-6
    assert report.weights["upstream_vs_reference_max_abs"] < 1e-12
    assert report.weights["weight_sum_rule"]["max_abs_residual"] < 1e-12
    assert report.offgrid["new_point"] is not None
    assert report.offgrid["new_point"]["correction_max_abs"] < 1e-7
    assert report.offgrid["shared_point"]["translation_invariance_residual"] < 1e-10
    assert report.convergence["downstream_eta_sensitivity"][
        "max_abs_change_eta_x2"
    ] < 1e-8


@requires_longrangefy
def test_asr_difference_is_reported_not_aligned(cubic_unitcell):
    report = _compare_or_skip(cubic_unitcell, (2, 2, 2), (2, 2, 4), asr=True)
    assert report.asr["aligned"] is False
    assert "effect" in report.asr
    assert "max_abs_eV_A2" in report.asr["effect"]["correction"]
    assert report.asr["effect"]["correction"]["max_abs_eV_A2"] >= 0.0
    assert "ASR" in report.asr["note"]


@requires_longrangefy
def test_skew_cell_equal_reps_and_nonuniform_target(skew_unitcell):
    equal = _compare_or_skip(skew_unitcell, (2, 2, 2), (2, 2, 2))
    assert equal.weights["upstream_vs_reference_max_abs"] < 1e-12
    assert equal.matrices["correction"].max_abs < 1e-10

    report = compare_matched_correction(skew_unitcell, (2, 2, 2), (2, 2, 4))
    assert report.weights["upstream_vs_reference_max_abs"] < 1e-12
    assert report.matrices["correction"].max_abs < 1e-7
    assert report.offgrid["shared_point"]["translation_invariance_residual"] < 1e-10


@requires_longrangefy
def test_phi0_symmetry_tolerance_is_never_invented(cubic_unitcell):
    dynmat = np.array(cubic_unitcell.dynmat, copy=True)
    dynmat[-1, 0, 0, 1, 0] += 1e-3  # finite reciprocal-kernel violation
    corrupted = UnitcellData(
        crystal=cubic_unitcell.crystal,
        energy=cubic_unitcell.energy,
        ifcs=cubic_unitcell.ifcs,
        epsilon_inf=cubic_unitcell.epsilon_inf,
        zeff=cubic_unitcell.zeff,
        qpoints=cubic_unitcell.qpoints,
        dynmat=dynmat,
        ngqpt=cubic_unitcell.ngqpt,
        nqshft=cubic_unitcell.nqshft,
        q1shft=cubic_unitcell.q1shft,
        symrel=cubic_unitcell.symrel,
        tnons=cubic_unitcell.tnons,
    )
    with pytest.raises(ValueError, match=r"(?i)(reciprocal-kernel|symmetr|phi0)"):
        compare_matched_correction(corrupted, (2, 2, 2), (2, 2, 4))
