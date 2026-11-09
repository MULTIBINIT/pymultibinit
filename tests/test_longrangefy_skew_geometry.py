"""Skew-cell geometry at the public supercell boundary."""
import numpy as np

from pymultibinit.pyeffpot.datastructures import CrystalInfo, UnitcellData
from pymultibinit.pyeffpot.supercell_builder import build_supercell, _generate_full_bz_qpoints


def test_anisotropic_supercell_tiles_a_nonsymmetric_lattice():
    cell = np.array([[4.0, 0.8, 0.3], [0.1, 4.5, 0.2], [0.6, -0.2, 5.0]])
    frac = np.array([[0.05, 0.10, 0.02], [0.55, 0.60, 0.52]])
    reps = (2, 2, 4)
    q = _generate_full_bz_qpoints(np.array((2, 2, 2)), 1, None)
    crystal = CrystalInfo(natom=2, ntypat=2, rprimd=cell.T, xred=frac,
                          xcart=frac @ cell, typat=np.array([1, 2]),
                          amu=np.array([1.0, 1.0]), znucl=np.array([1, 2]))
    unitcell = UnitcellData(crystal=crystal, energy=0.0, qpoints=q,
                            dynmat=np.zeros((8, 2, 3, 2, 3, 2)),
                            ngqpt=np.array((2, 2, 2)), nqshft=1,
                            q1shft=np.zeros((1, 3)), symrel=None, tnons=None)
    result = build_supercell(unitcell, reps, dipdip=False, asr=False)
    expected_positions = np.concatenate([(frac + offset) @ cell for offset in np.ndindex(*reps)])
    np.testing.assert_allclose(result.crystal_sc.rprimd, (np.diag(reps) @ cell).T, atol=1e-12)
    np.testing.assert_allclose(result.crystal_sc.xcart, expected_positions, atol=1e-12)


def test_anisotropic_skew_dipole_branches_match_standalone():
    import pytest
    pytest.importorskip("longrangefy")
    from pymultibinit.longrangefy_parity import compare_matched_correction

    cell = np.array([[4.0, 0.8, 0.3], [0.1, 4.5, 0.2], [0.6, -0.2, 5.0]])
    frac = np.array([[0.05, 0.10, 0.02], [-0.45, -0.40, -0.48]])
    reps = (2, 2, 2)
    q = _generate_full_bz_qpoints(np.array(reps), 1, None)
    crystal = CrystalInfo(natom=2, ntypat=2, rprimd=cell.T, xred=frac,
                          xcart=frac @ cell, typat=np.array([1, 2]),
                          amu=np.array([1.0, 1.0]), znucl=np.array([1, 2]))
    unitcell = UnitcellData(crystal=crystal, energy=0.0, qpoints=q,
                            dynmat=np.zeros((8, 2, 3, 2, 3, 2)),
                            ngqpt=np.array(reps), nqshft=1,
                            q1shft=np.zeros((1, 3)), symrel=None, tnons=None,
                            zeff=np.stack([np.eye(3), -np.eye(3)]),
                            epsilon_inf=2.0 * np.eye(3))
    report = compare_matched_correction(unitcell, reps, (2, 2, 4))
    assert report.alignment["ordering_residual_angstrom"] < 1e-8
    assert report.matrices["correction"].max_abs < 1e-7
    assert report.components["ewald"].max_abs < 1e-7


def test_nonsymmetric_ddb_reference_and_supercell(tmp_path):
    from pymultibinit.pyeffpot.ddb_parser_complete import read_ddb

    ddb = tmp_path / "skew.DDB"
    ddb.write_text(
        " **** DERIVATIVE DATABASE ****\n"
        "+DDB, Version number  20230401\n\n skew reference\n\n"
        "    usepaw 0\n     natom 1\n      nkpt 1\n    nsppol 1\n"
        "      nsym 1\n    ntypat 1\n    occopt 1\n     nband 1\n"
        "     acell 2.0 3.0 4.0\n       amu 1.0\n"
        "     rprim 1.0 0.2 0.1\n           0.1 1.0 0.2\n           0.3 -0.1 1.0\n"
        " dfpt_sciss 0.0\n    symrel 1 0 0 0 1 0 0 0 1\n"
        "     tnons 0.0 0.0 0.0\n     typat 1\n       wtk 1.0\n"
        "      xred 0.1 0.2 0.3\n     znucl 1.0\n      zion 1.0\n"
        " **** Database of total energy derivatives ****\n\n"
        " Number of data blocks=    1\n\n"
        " 2nd derivatives (non-stat.)  - # elements :           1\n"
        " qpt 0.0 0.0 0.0 1.0\n"
        "   1    1    1    1   1.0 0.0\n\n"
    )
    primitive_rows = np.array([[2.0, 0.4, 0.2], [0.3, 3.0, 0.6], [1.2, -0.4, 4.0]])
    reference = read_ddb(str(ddb))
    np.testing.assert_allclose(reference.rprimd, primitive_rows.T, atol=1e-12)
    np.testing.assert_allclose(reference.xcart, [[0.62, 0.52, 1.34]], atol=1e-12)
    expected_ifc = np.array([
        [28900/106929, -3400/320787, -26350/320787],
        [-3400/320787, 400/962361, 3100/962361],
        [-26350/320787, 3100/962361, 24025/962361],
    ])
    np.testing.assert_allclose(reference.dynmat[0, 0, :, 0, :, 0], expected_ifc, atol=1e-12)
    supercell = build_supercell(reference, (2, 1, 3), dipdip=False, asr=False)
    expected = np.concatenate(
        [(np.array([[0.1, 0.2, 0.3]]) + offset) @ primitive_rows
         for offset in np.ndindex(2, 1, 3)]
    )
    np.testing.assert_allclose(supercell.crystal_sc.xcart, expected, atol=1e-12)
