import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np
from scipy.io import netcdf_file

import pymultibinit.training as training_mod
from pymultibinit.cli import main
from pymultibinit.training import (
    FitFeatureMatrices,
    MonomialKey,
    PairKey,
    XmlBasisFunction,
    PythonFitConfig,
    TrainingFrame,
    basis_to_coefficients,
    build_factor_action_map,
    build_training_dataset,
    canonicalize_monomial_orbit,
    compute_goal_function,
    count_fortran_displacement_coefficients,
    count_fortran_irreducible_pair_combinations,
    displacement_pair_diagnostics,
    evaluate_basis_features,
    fit_multibinit_model_python,
    generate_displacement_basis,
    generate_fortran_pair_list,
    select_greedy_coefficients,
    load_xml_basis,
    normalize_pair_key,
    read_hist_frames,
    solve_weighted_least_squares,
    train_multibinit_model,
    with_fortran_text_labels,
    write_fitted_xml,
)
from pymultibinit.pyeffpot.xml_parser import read_coefficient_xml


def _write_fake_multibinit(path: Path, body: str) -> Path:
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import os\n"
        "import pathlib\n"
        "import sys\n"
        + body,
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def _write_tiny_hist(path: Path) -> Path:
    with netcdf_file(str(path), "w") as nc:
        nc.createDimension("time", 2)
        nc.createDimension("natom", 2)
        nc.createDimension("ntypat", 2)
        nc.createDimension("three", 3)
        nc.createDimension("six", 6)

        typat = nc.createVariable("typat", "i", ("natom",))
        znucl = nc.createVariable("znucl", "d", ("ntypat",))
        rprimd = nc.createVariable("rprimd", "d", ("time", "three", "three"))
        xred = nc.createVariable("xred", "d", ("time", "natom", "three"))
        xcart = nc.createVariable("xcart", "d", ("time", "natom", "three"))
        etotal = nc.createVariable("etotal", "d", ("time",))
        fcart = nc.createVariable("fcart", "d", ("time", "natom", "three"))
        strten = nc.createVariable("strten", "d", ("time", "six"))

        typat[:] = np.array([1, 2], dtype=np.int32)
        znucl[:] = np.array([56.0, 8.0])
        rprimd[:] = np.array([np.eye(3) * 7.5, np.eye(3) * 7.6])
        xred[:] = np.array(
            [
                [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
                [[0.0, 0.0, 0.0], [0.51, 0.5, 0.5]],
            ]
        )
        xcart[:] = np.array(
            [
                [[0.0, 0.0, 0.0], [3.75, 3.75, 3.75]],
                [[0.0, 0.0, 0.0], [3.876, 3.8, 3.8]],
            ]
        )
        etotal[:] = np.array([-10.0, -9.9])
        fcart[:] = np.array(
            [
                [[0.0, 0.1, 0.2], [0.3, 0.4, 0.5]],
                [[0.6, 0.7, 0.8], [0.9, 1.0, 1.1]],
            ]
        )
        strten[:] = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1.1, 2.1, 3.1, 4.1, 5.1, 6.1]])
    return path


def test_python_fit_config_defaults_and_validation():
    config = PythonFitConfig(ncell=(2, 2, 2))

    assert config.ncell == (2, 2, 2)
    assert config.fit_on == (True, True, True)
    assert config.fit_factors == (1.0, 1.0, 1.0)
    assert config.regularization == 0.0
    assert config.selection == "all"
    assert config.ncoeff is None
    assert config.cutoff is None
    assert config.power_range == (3, 4)

    with pytest.raises(ValueError, match="ncell values must be positive"):
        PythonFitConfig(ncell=(2, 0, 2))
    with pytest.raises(ValueError, match="selection must be one of"):
        PythonFitConfig(ncell=(1, 1, 1), selection="random")
    with pytest.raises(ValueError, match="regularization must be non-negative"):
        PythonFitConfig(ncell=(1, 1, 1), regularization=-1.0)
    with pytest.raises(ValueError, match="power_range must be ordered"):
        PythonFitConfig(ncell=(1, 1, 1), power_range=(4, 3))


def test_read_hist_frames_loads_abinit_units_and_shapes(tmp_path):
    hist = _write_tiny_hist(tmp_path / "training_HIST.nc")

    frames = read_hist_frames(hist)

    assert len(frames) == 2
    assert frames[0].rprimd.shape == (3, 3)
    assert frames[0].xred.shape == (2, 3)
    assert frames[0].xcart.shape == (2, 3)
    assert frames[0].forces.shape == (2, 3)
    assert frames[0].stress.shape == (6,)
    assert frames[0].energy == -10.0
    assert frames[0].units == {
        "rprimd": "Bohr",
        "xcart": "Bohr",
        "xred": "fractional",
        "energy": "Hartree",
        "forces": "Hartree/Bohr",
        "stress": "Hartree/Bohr^3",
    }
    np.testing.assert_allclose(frames[1].xred[1], [0.51, 0.5, 0.5])


def test_read_hist_frames_reports_missing_required_variable(tmp_path):
    bad_hist = tmp_path / "bad_HIST.nc"
    with netcdf_file(str(bad_hist), "w") as nc:
        nc.createDimension("time", 1)
        nc.createDimension("natom", 1)
        nc.createDimension("three", 3)
        rprimd = nc.createVariable("rprimd", "d", ("time", "three", "three"))
        rprimd[:] = np.eye(3)[None, :, :]

    with pytest.raises(ValueError, match="HIST file is missing required variable 'xred'"):
        read_hist_frames(bad_hist)


def test_build_training_dataset_computes_displacement_strain_and_weights():
    reference = TrainingFrame(
        rprimd=np.eye(3),
        xred=np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        xcart=np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        energy=1.0,
        forces=np.zeros((2, 3)),
        stress=np.zeros(6),
    )
    frame = TrainingFrame(
        rprimd=np.diag([1.1, 1.0, 1.0]),
        xred=np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        xcart=np.array([[0.0, 0.1, 0.0], [0.6, 0.0, 0.0]]),
        energy=2.0,
        forces=np.ones((2, 3)),
        stress=np.arange(6, dtype=float),
    )

    dataset = build_training_dataset(reference, [frame])

    np.testing.assert_allclose(dataset.displacement[0], [[0.0, 0.1, 0.0], [0.05, 0.0, 0.0]])
    np.testing.assert_allclose(dataset.strain[0], [0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert dataset.du_delta.shape == (1, 6, 2, 3)
    assert dataset.du_delta[0, 1, 0, 1] == pytest.approx(0.1)
    assert dataset.du_delta[0, 0, 1, 0] == pytest.approx(0.05 / 1.1)
    np.testing.assert_allclose(dataset.ucvol, [1.1])
    np.testing.assert_allclose(dataset.sqomega, [1.1 ** (4.0 / 3.0) / 2 ** (1.0 / 3.0)])
    np.testing.assert_allclose(dataset.energy_diff, [2.0])
    np.testing.assert_allclose(dataset.force_diff[0], np.ones((2, 3)))
    np.testing.assert_allclose(dataset.stress_diff[0], np.arange(6, dtype=float))


def test_build_training_dataset_removes_pure_homogeneous_strain_from_displacements():
    reference = TrainingFrame(
        rprimd=np.eye(3),
        xred=np.array([[0.25, 0.5, 0.75]]),
        xcart=np.array([[0.25, 0.5, 0.75]]),
        energy=0.0,
        forces=np.zeros((1, 3)),
        stress=np.zeros(6),
    )
    strained_lattice = np.diag([1.2, 1.1, 0.9])
    frame = TrainingFrame(
        rprimd=strained_lattice,
        xred=reference.xred,
        xcart=reference.xred @ strained_lattice.T,
        energy=0.0,
        forces=np.zeros((1, 3)),
        stress=np.zeros(6),
    )

    dataset = build_training_dataset(reference, [frame])

    np.testing.assert_allclose(dataset.displacement[0], 0.0, atol=1e-14)
    np.testing.assert_allclose(dataset.du_delta[0], 0.0, atol=1e-14)


def test_build_training_dataset_aligns_frame_atom_order_to_reference():
    reference = TrainingFrame(
        rprimd=np.eye(3),
        xred=np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.75, 0.0, 0.0]]),
        xcart=np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.75, 0.0, 0.0]]),
        energy=0.0,
        forces=np.zeros((3, 3)),
        stress=np.zeros(6),
    )
    frame = TrainingFrame(
        rprimd=np.eye(3),
        xred=np.array([[0.0, 0.0, 0.0], [0.75, 0.0, 0.0], [0.25, 0.0, 0.0]]),
        xcart=np.array([[0.0, 0.0, 0.0], [0.75, 0.0, 0.0], [0.25, 0.0, 0.0]]),
        energy=0.0,
        forces=np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        stress=np.zeros(6),
    )

    dataset = build_training_dataset(reference, [frame])

    np.testing.assert_allclose(dataset.displacement[0], 0.0)
    np.testing.assert_allclose(dataset.force_diff[0, :, 0], [0.0, 2.0, 3.0])


def test_build_training_dataset_uses_pymultibinit_lattice_orientation():
    reference_lattice = np.array([[1.0, 0.2, 0.0], [0.1, 1.3, 0.0], [0.0, 0.3, 1.5]])
    strained_lattice = np.array([[1.1, 0.25, 0.0], [0.15, 1.25, 0.1], [0.0, 0.35, 1.45]])
    xred = np.array([[0.2, 0.3, 0.4]])
    internal_disp = np.array([[0.01, -0.02, 0.03]])
    reference = TrainingFrame(
        rprimd=reference_lattice,
        xred=xred,
        xcart=xred @ reference_lattice.T,
        energy=0.0,
        forces=np.zeros((1, 3)),
        stress=np.zeros(6),
    )
    frame = TrainingFrame(
        rprimd=strained_lattice,
        xred=xred,
        xcart=xred @ strained_lattice.T + internal_disp,
        energy=0.0,
        forces=np.zeros((1, 3)),
        stress=np.zeros(6),
    )

    dataset = build_training_dataset(reference, [frame])

    np.testing.assert_allclose(dataset.displacement[0], internal_disp)


def test_build_training_dataset_subtracts_fixed_model_predictions():
    class FixedModel:
        def evaluate(self, xcart, rprimd):
            assert xcart.shape == (1, 3)
            assert rprimd.shape == (3, 3)
            return 1.5, np.array([[0.25, 0.5, 0.75]]), np.array(
                [[1.0, 6.0, 5.0], [6.0, 2.0, 4.0], [5.0, 4.0, 3.0]]
            )

    reference = TrainingFrame(
        rprimd=np.eye(3),
        xred=np.zeros((1, 3)),
        xcart=np.zeros((1, 3)),
        energy=0.0,
        forces=np.zeros((1, 3)),
        stress=np.zeros(6),
    )
    frame = TrainingFrame(
        rprimd=np.eye(3),
        xred=np.zeros((1, 3)),
        xcart=np.array([[0.1, 0.0, 0.0]]),
        energy=3.0,
        forces=np.array([[1.0, 1.5, 2.0]]),
        stress=np.array([2.0, 3.0, 4.0, 4.5, 5.5, 6.5]),
    )

    dataset = build_training_dataset(reference, [frame], fixed_model=FixedModel())

    np.testing.assert_allclose(dataset.energy_diff, [1.5])
    np.testing.assert_allclose(dataset.force_diff[0], [[0.75, 1.0, 1.25]])
    np.testing.assert_allclose(dataset.stress_diff[0], [1.0, 1.0, 1.0, 0.5, 0.5, 0.5])


def test_build_training_dataset_reports_shape_mismatch():
    reference = TrainingFrame(
        rprimd=np.eye(3),
        xred=np.zeros((2, 3)),
        xcart=np.zeros((2, 3)),
        energy=0.0,
        forces=np.zeros((2, 3)),
        stress=np.zeros(6),
    )
    frame = TrainingFrame(
        rprimd=np.eye(3),
        xred=np.zeros((1, 3)),
        xcart=np.zeros((1, 3)),
        energy=0.0,
        forces=np.zeros((1, 3)),
        stress=np.zeros(6),
    )

    with pytest.raises(ValueError, match="Frame 0 atom count mismatch"):
        build_training_dataset(reference, [frame])


def test_solve_weighted_least_squares_recovers_synthetic_force_coefficients():
    features = FitFeatureMatrices(
        energy=np.zeros((1, 2)),
        forces=np.array([[[[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]]]),
        stress=np.zeros((1, 6, 2)),
    )
    dataset = _dataset_for_solver(force_diff=np.array([[[2.0, 3.0, 0.0]]]))

    result = solve_weighted_least_squares(features, dataset, PythonFitConfig(ncell=(1, 1, 1), fit_on=(True, False, False)))

    np.testing.assert_allclose(result.coefficients, [2.0, 3.0], atol=1e-12)
    assert result.diagnostics.info == 0


def test_solve_weighted_least_squares_ridge_regularizes_singular_system():
    features = FitFeatureMatrices(
        energy=np.zeros((1, 2)),
        forces=np.array([[[[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]]]]),
        stress=np.zeros((1, 6, 2)),
    )
    dataset = _dataset_for_solver(force_diff=np.array([[[2.0, 0.0, 0.0]]]))

    result = solve_weighted_least_squares(
        features,
        dataset,
        PythonFitConfig(ncell=(1, 1, 1), fit_on=(True, False, False), regularization=1.0),
    )

    assert np.isfinite(result.coefficients).all()
    assert result.diagnostics.regularization == 1.0
    assert result.coefficients[0] == pytest.approx(result.coefficients[1])
    assert abs(result.coefficients[0]) < 1.0


def test_compute_goal_function_uses_multibinit_factors():
    features = FitFeatureMatrices(
        energy=np.array([[0.5], [1.0]]),
        forces=np.array([[[[1.0], [0.0], [0.0]]], [[[0.0], [2.0], [0.0]]]]),
        stress=np.array([[[1.0], [0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [2.0], [0.0], [0.0], [0.0], [0.0]]]),
    )
    dataset = _dataset_for_solver(
        ntime=2,
        force_diff=np.array([[[2.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]]]),
        stress_diff=np.array([[3.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 4.0, 0.0, 0.0, 0.0, 0.0]]),
        energy_diff=np.array([1.5, 1.0]),
        sqomega=np.array([4.0, 9.0]),
    )

    goal = compute_goal_function(np.array([1.0]), features, dataset, weights=np.array([1.0, 2.0]))

    assert goal.force == pytest.approx((1.0**2 * 1.0 + (-1.0) ** 2 * 2.0) / (3 * 1 * 2))
    assert goal.stress == pytest.approx((4.0 * 2.0**2 * 1.0 + 9.0 * 2.0**2 * 2.0) / (6 * 2))
    assert goal.energy == pytest.approx(((1.0**2) / 2.0 * 1.0 + (0.0**2) / 3.0 * 2.0) / 2)
    assert goal.force_stress == pytest.approx(goal.force + goal.stress)


def test_solver_rejects_dataset_shape_mismatch():
    features = FitFeatureMatrices(
        energy=np.zeros((2, 1)),
        forces=np.zeros((2, 3, 3, 1)),
        stress=np.zeros((2, 6, 1)),
    )
    dataset = _dataset_for_solver(ntime=2, force_diff=np.zeros((3, 2, 3)))

    with pytest.raises(ValueError, match="force_diff must have shape"):
        solve_weighted_least_squares(features, dataset, PythonFitConfig(ncell=(1, 1, 1)))


def test_solver_rejects_invalid_weights_and_regularization():
    features = FitFeatureMatrices(
        energy=np.zeros((1, 1)),
        forces=np.zeros((1, 1, 3, 1)),
        stress=np.zeros((1, 6, 1)),
    )
    dataset = _dataset_for_solver()

    with pytest.raises(ValueError, match="weights must be finite and non-negative"):
        solve_weighted_least_squares(features, dataset, weights=np.array([-1.0]))
    with pytest.raises(ValueError, match="regularization must be finite"):
        PythonFitConfig(ncell=(1, 1, 1), regularization=np.nan)
    with pytest.raises(ValueError, match="regularization must be finite"):
        solve_weighted_least_squares(features, dataset, regularization=np.inf)


def test_compute_goal_function_rejects_non_vector_coefficients():
    features = FitFeatureMatrices(
        energy=np.zeros((1, 1)),
        forces=np.zeros((1, 1, 3, 1)),
        stress=np.zeros((1, 6, 1)),
    )
    dataset = _dataset_for_solver()

    with pytest.raises(ValueError, match="coefficients must have shape"):
        compute_goal_function(np.zeros((1, 1)), features, dataset)


def test_xml_basis_adapter_preserves_terms_and_round_trips(tmp_path):
    xml = tmp_path / "basis.xml"
    xml.write_text(
        """<?xml version="1.0" ?>
<Heff_definition>
  <coefficient number="7" value=" 1.2500000000E+00" text="u2eta">
    <term weight=" -2.500000">
      <displacement_diff atom_a="1" atom_b="2" direction="x" power="2">
        <cell_a>0 0 0</cell_a>
        <cell_b>1 0 -1</cell_b>
      </displacement_diff>
      <strain power=" 1" voigt=" 4" />
    </term>
  </coefficient>
</Heff_definition>
""",
        encoding="utf-8",
    )

    basis = load_xml_basis(xml)

    assert basis == [
        XmlBasisFunction(
            number=7,
            value=1.25,
            text="u2eta",
            terms=(
                {
                    "weight": -2.5,
                    "displacements": (
                        {
                            "atom_a": 1,
                            "atom_b": 2,
                            "direction": "x",
                            "power": 2,
                            "cell_a": (0, 0, 0),
                            "cell_b": (1, 0, -1),
                        },
                    ),
                    "strains": ({"power": 1, "voigt": 4},),
                },
            ),
        )
    ]

    out = tmp_path / "fitted.xml"
    write_fitted_xml(out, basis, fitted_values=[3.5])
    coeffs = read_coefficient_xml(out)

    assert len(coeffs) == 1
    assert coeffs[0].number == 7
    assert coeffs[0].value == pytest.approx(3.5)
    assert coeffs[0].terms[0].displacements[0]["cell_b"] == [1, 0, -1]
    assert coeffs[0].terms[0].strains[0] == {"power": 1, "voigt": 4}


def test_xml_basis_adapter_preserves_batio3_counts():
    xml = Path(__file__).parent.parent / "examples/BaHfO3_training/real_training_run/wrapper_run_qgrid/BaTiO3_fit_coeffs.xml"
    parsed = read_coefficient_xml(xml)
    basis = load_xml_basis(xml)

    assert len(basis) == len(parsed)
    assert [len(item.terms) for item in basis] == [len(coeff.terms) for coeff in parsed]
    assert sum(len(term["displacements"]) for item in basis for term in item.terms) == sum(
        len(term.displacements) for coeff in parsed for term in coeff.terms
    )
    assert sum(len(term["strains"]) for item in basis for term in item.terms) == sum(
        len(term.strains) for coeff in parsed for term in coeff.terms
    )


def test_basis_to_coefficients_rejects_wrong_value_count():
    basis = [XmlBasisFunction(number=1, value=0.0, text="", terms=())]

    with pytest.raises(ValueError, match="fitted_values must have shape"):
        basis_to_coefficients(basis, fitted_values=[1.0, 2.0])


def test_evaluate_basis_features_matches_displacement_hand_calculation():
    basis = [
        XmlBasisFunction(
            number=1,
            value=0.0,
            text="quadratic",
            terms=(
                {
                    "weight": 2.0,
                    "displacements": (
                        {
                            "atom_a": 0,
                            "atom_b": 1,
                            "direction": "x",
                            "power": 2,
                            "cell_a": (0, 0, 0),
                            "cell_b": (0, 0, 0),
                        },
                    ),
                    "strains": (),
                },
            ),
        )
    ]
    dataset = _dataset_for_features(displacement=np.array([[[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]]))

    features = evaluate_basis_features(basis, dataset, ncell=(1, 1, 1))

    assert features.energy.shape == (1, 1)
    assert features.forces.shape == (1, 2, 3, 1)
    assert features.stress.shape == (1, 6, 1)
    np.testing.assert_allclose(features.energy[:, 0], [8.0])
    np.testing.assert_allclose(features.forces[0, :, :, 0], [[8.0, 0.0, 0.0], [-8.0, 0.0, 0.0]])
    np.testing.assert_allclose(features.stress[:, :, 0], 0.0)


def test_evaluate_basis_features_uses_fortran_displacement_orientation():
    basis = [
        XmlBasisFunction(
            number=1,
            value=0.0,
            text="linear",
            terms=(
                {
                    "weight": 1.0,
                    "displacements": (
                        {
                            "atom_a": 0,
                            "atom_b": 1,
                            "direction": "x",
                            "power": 1,
                            "cell_a": (0, 0, 0),
                            "cell_b": (0, 0, 0),
                        },
                    ),
                    "strains": (),
                },
            ),
        )
    ]
    dataset = _dataset_for_features(displacement=np.array([[[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]]))

    features = evaluate_basis_features(basis, dataset, ncell=(1, 1, 1))

    np.testing.assert_allclose(features.energy[:, 0], [-2.0])
    np.testing.assert_allclose(features.forces[0, :, :, 0], [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])


def test_evaluate_basis_features_handles_strain_stress_feature():
    basis = [
        XmlBasisFunction(
            number=1,
            value=0.0,
            text="eta",
            terms=({"weight": 3.0, "displacements": (), "strains": ({"power": 1, "voigt": 1},)},),
        )
    ]
    dataset = _dataset_for_features(displacement=np.zeros((1, 1, 3)), strain=np.array([[0.2, 0.0, 0.0, 0.0, 0.0, 0.0]]), ucvol=np.array([10.0]))

    features = evaluate_basis_features(basis, dataset, ncell=(1, 1, 1))

    np.testing.assert_allclose(features.energy[:, 0], [0.6])
    np.testing.assert_allclose(features.stress[0, :, 0], [0.36, 0.0, 0.0, 0.0, 0.0, 0.0])


def test_evaluate_xml_loaded_basis_shapes_with_multiple_frames():
    xml = Path(__file__).parent.parent / "examples/BaHfO3_training/real_training_run/wrapper_run_qgrid/BaTiO3_fit_coeffs.xml"
    basis = load_xml_basis(xml)[:3]
    dataset = _dataset_for_features(displacement=np.zeros((2, 40, 3)), ntime=2, strain=np.zeros((2, 6)), ucvol=np.ones(2))

    features = evaluate_basis_features(basis, dataset, ncell=(2, 2, 2))

    assert features.energy.shape == (2, 3)
    assert features.forces.shape == (2, 40, 3, 3)
    assert features.stress.shape == (2, 6, 3)


def test_evaluate_basis_features_uses_supercell_builder_cell_order():
    basis = [
        XmlBasisFunction(
            number=1,
            value=0.0,
            text="shift-x",
            terms=(
                {
                    "weight": 1.0,
                    "displacements": (
                        {
                            "atom_a": 0,
                            "atom_b": 0,
                            "direction": "x",
                            "power": 2,
                            "cell_a": (0, 0, 0),
                            "cell_b": (1, 0, 0),
                        },
                    ),
                    "strains": (),
                },
            ),
        )
    ]
    displacement = np.zeros((1, 8, 3), dtype=float)
    displacement[0, :, 0] = np.arange(8, dtype=float)
    dataset = _dataset_for_features(displacement=displacement)

    features = evaluate_basis_features(basis, dataset, ncell=(2, 2, 2))

    assert features.energy[0, 0] == pytest.approx(128.0)


def test_fit_multibinit_model_python_wires_pipeline_without_subprocess(tmp_path, monkeypatch):
    ddb = tmp_path / "input.ddb"
    hist = tmp_path / "training_HIST.nc"
    basis_xml = tmp_path / "basis.xml"
    output_xml = tmp_path / "fitted.xml"
    ddb.write_text("ddb placeholder", encoding="utf-8")
    _write_force_fit_hist(hist)
    basis_xml.write_text(
        """<?xml version="1.0" ?>
<Heff_definition>
  <coefficient number="1" value=" 0.0000000000E+00" text="linear">
    <term weight=" 1.000000">
      <displacement_diff atom_a="0" atom_b="1" direction="x" power="1">
        <cell_a>0 0 0</cell_a>
        <cell_b>0 0 0</cell_b>
      </displacement_diff>
    </term>
  </coefficient>
</Heff_definition>
""",
        encoding="utf-8",
    )
    reference = TrainingFrame(
        rprimd=np.eye(3),
        xred=np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        xcart=np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        energy=0.0,
        forces=np.zeros((2, 3)),
        stress=np.zeros(6),
    )
    monkeypatch.setattr("pymultibinit.training._reference_frame_from_ddb", lambda _ddb, _ncell: reference)
    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("subprocess used")))

    result = fit_multibinit_model_python(
        ddb=ddb,
        hist=hist,
        basis_xml=basis_xml,
        output_xml=output_xml,
        config=PythonFitConfig(ncell=(1, 1, 1), fit_on=(True, False, False), selection="greedy", ncoeff=1),
    )

    assert result.output_xml == str(output_xml.resolve())
    assert result.ncoeff == 1
    assert result.nframes == 1
    np.testing.assert_allclose(result.coefficients, [-2.0])
    assert read_coefficient_xml(output_xml)[0].value == pytest.approx(-2.0)


def test_fit_multibinit_model_python_reports_missing_paths(tmp_path):
    with pytest.raises(FileNotFoundError, match="DDB file not found"):
        fit_multibinit_model_python(tmp_path / "missing.ddb", tmp_path / "missing.nc", tmp_path / "missing.xml")

    ddb = tmp_path / "input.ddb"
    hist = tmp_path / "training_HIST.nc"
    basis = tmp_path / "basis.xml"
    ddb.write_text("ddb", encoding="utf-8")
    hist.write_text("hist", encoding="utf-8")
    basis.write_text("xml", encoding="utf-8")
    with pytest.raises(TypeError, match="config must be a PythonFitConfig"):
        fit_multibinit_model_python(ddb, hist, basis, config={"ncell": (1, 1, 1)})


def test_select_greedy_coefficients_chooses_expected_terms_deterministically():
    features = FitFeatureMatrices(
        energy=np.zeros((1, 3)),
        forces=np.array([[[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]]]),
        stress=np.zeros((1, 6, 3)),
    )
    dataset = _dataset_for_solver(force_diff=np.array([[[2.0, 3.0, 0.0]]]))

    result = select_greedy_coefficients(
        features,
        dataset,
        PythonFitConfig(ncell=(1, 1, 1), fit_on=(True, False, False), ncoeff=2, selection="greedy"),
    )

    assert result.selected == (1, 0)
    np.testing.assert_allclose(result.coefficients, [2.0, 3.0, 0.0], atol=1e-12)
    assert len(result.steps) == 2
    assert result.steps[-1]["train_rmse"]["forces_ha_bohr"] == pytest.approx(0.0)


def test_select_greedy_coefficients_reports_validation_rmse():
    features = FitFeatureMatrices(
        energy=np.zeros((1, 2)),
        forces=np.array([[[[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]]]),
        stress=np.zeros((1, 6, 2)),
    )
    dataset = _dataset_for_solver(force_diff=np.array([[[2.0, 0.0, 0.0]]]))
    validation = _dataset_for_solver(force_diff=np.array([[[3.0, 0.0, 0.0]]]))

    result = select_greedy_coefficients(
        features,
        dataset,
        PythonFitConfig(ncell=(1, 1, 1), fit_on=(True, False, False), ncoeff=1, selection="greedy"),
        validation_features=features,
        validation_dataset=validation,
    )

    assert result.selected == (0,)
    assert result.steps[0]["train_rmse"]["forces_ha_bohr"] == pytest.approx(0.0)
    assert result.steps[0]["validation_rmse"]["forces_ha_bohr"] == pytest.approx(np.sqrt(1.0 / 3.0))


def test_select_greedy_coefficients_large_basis_path_matches_expected_terms():
    forces = np.zeros((1, 1, 3, 1001))
    forces[0, 0, 0, 1000] = 1.0
    forces[0, 0, 1, 999] = 1.0
    features = FitFeatureMatrices(
        energy=np.zeros((1, 1001)),
        forces=forces,
        stress=np.zeros((1, 6, 1001)),
    )
    dataset = _dataset_for_solver(force_diff=np.array([[[2.0, 3.0, 0.0]]]))

    result = select_greedy_coefficients(
        features,
        dataset,
        PythonFitConfig(ncell=(1, 1, 1), fit_on=(True, False, False), ncoeff=2, selection="greedy"),
    )

    assert result.selected == (999, 1000)
    np.testing.assert_allclose(result.coefficients[[999, 1000]], [3.0, 2.0], atol=1e-12)
    assert result.steps[-1]["train_rmse"]["forces_ha_bohr"] == pytest.approx(0.0)


def test_fit_screened_greedy_uses_screening_weights_and_candidate_pool(monkeypatch):
    dataset = training_mod.TrainingDataset(
        displacement=np.zeros((4, 1, 3)),
        du_delta=np.zeros((4, 6, 1, 3)),
        strain=np.zeros((4, 6)),
        ucvol=np.ones(4),
        sqomega=np.ones(4),
        energy_diff=np.zeros(4),
        force_diff=np.zeros((4, 1, 3)),
        stress_diff=np.zeros((4, 6)),
    )
    basis = [object(), object(), object(), object()]
    seen_screening_weights = []
    seen_pool = []

    def fake_evaluate_basis_features(chunk_basis, chunk_dataset, ncell, backend="auto", memmap_dir=None):
        seen_pool.append(len(chunk_basis))
        ntime = chunk_dataset.displacement.shape[0]
        ncoeff = len(chunk_basis)
        return FitFeatureMatrices(
            energy=np.zeros((ntime, ncoeff)),
            forces=np.zeros((ntime, 1, 3, ncoeff)),
            stress=np.zeros((ntime, 6, ncoeff)),
        )

    def fake_rhs_diagonal_target(features, chunk_dataset, config, weights):
        seen_screening_weights.append(None if weights is None else np.asarray(weights).copy())
        ncoeff = features.energy.shape[1]
        return np.arange(1, ncoeff + 1, dtype=float), np.ones(ncoeff), 1.0

    def fake_select_greedy_coefficients(features, full_dataset, config, weights=None):
        assert np.asarray(weights).tolist() == [1.0, 2.0, 3.0, 4.0]
        return training_mod.GreedySelectionResult(
            selected=(0,),
            coefficients=np.array([5.0, 0.0]),
            diagnostics=training_mod.FitDiagnostics(
                goal=training_mod.GoalFunctionComponents(0.0, 0.0, 0.0, 0.0),
                residual_norm=0.0,
                matrix_rank=1,
                condition_number=1.0,
                regularization=0.0,
                info=0,
            ),
            steps=({"selected": 0},),
        )

    monkeypatch.setattr(training_mod, "evaluate_basis_features", fake_evaluate_basis_features)
    monkeypatch.setattr(training_mod, "_greedy_rhs_diagonal_target", fake_rhs_diagonal_target)
    monkeypatch.setattr(training_mod, "select_greedy_coefficients", fake_select_greedy_coefficients)

    result = training_mod._fit_screened_greedy(
        basis,
        dataset,
        PythonFitConfig(
            ncell=(1, 1, 1),
            selection="screened_greedy",
            ncoeff=1,
            candidate_pool_size=2,
            feature_chunk_size=2,
            screening_frame_count=2,
        ),
        weights=np.array([1.0, 2.0, 3.0, 4.0]),
    )

    assert [item.tolist() for item in seen_screening_weights[:2]] == [[1.0, 4.0], [1.0, 4.0]]
    assert seen_pool == [2, 2, 2]
    assert result.selected == (1,)
    assert result.coefficients[1] == pytest.approx(5.0)
    assert result.steps[0]["screened_pool_size"] == 2


def test_select_greedy_coefficients_respects_constraints_and_singular_candidates():
    features = FitFeatureMatrices(
        energy=np.zeros((1, 4)),
        forces=np.array([[[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]]),
        stress=np.zeros((1, 6, 4)),
    )
    dataset = _dataset_for_solver(force_diff=np.array([[[1.0, 2.0, 0.0]]]))

    result = select_greedy_coefficients(
        features,
        dataset,
        PythonFitConfig(ncell=(1, 1, 1), fit_on=(True, False, False), ncoeff=2, selection="greedy"),
        banned={1},
        preselected={0},
    )

    assert result.selected == (0, 2)
    assert 1 not in result.selected
    assert result.steps[-1]["skipped_singular"] >= 1


def test_select_greedy_coefficients_rejects_impossible_constraints():
    features = FitFeatureMatrices(energy=np.zeros((1, 2)), forces=np.zeros((1, 1, 3, 2)), stress=np.zeros((1, 6, 2)))
    dataset = _dataset_for_solver()

    with pytest.raises(ValueError, match="preselected coefficient count"):
        select_greedy_coefficients(
            features,
            dataset,
            PythonFitConfig(ncell=(1, 1, 1), selection="greedy", ncoeff=1),
            preselected={0, 1},
        )
    with pytest.raises(ValueError, match="Not enough selectable coefficients"):
        select_greedy_coefficients(
            features,
            dataset,
            PythonFitConfig(ncell=(1, 1, 1), selection="greedy", ncoeff=2),
            banned={1},
        )
    with pytest.raises(ValueError, match="Unable to select requested ncoeff"):
        select_greedy_coefficients(
            features,
            dataset,
            PythonFitConfig(ncell=(1, 1, 1), selection="greedy", ncoeff=1),
        )
    with pytest.raises(ValueError, match="Preselected coefficients produce a singular"):
        select_greedy_coefficients(
            features,
            dataset,
            PythonFitConfig(ncell=(1, 1, 1), selection="greedy", ncoeff=1),
            preselected={0},
        )


def _dataset_for_solver(
    ntime=1,
    force_diff=None,
    stress_diff=None,
    energy_diff=None,
    sqomega=None,
):
    force_diff = np.zeros((ntime, 1, 3)) if force_diff is None else force_diff
    stress_diff = np.zeros((ntime, 6)) if stress_diff is None else stress_diff
    energy_diff = np.zeros(ntime) if energy_diff is None else energy_diff
    sqomega = np.ones(ntime) if sqomega is None else sqomega
    return type(
        "Dataset",
        (),
        {
            "energy_diff": np.asarray(energy_diff, dtype=float),
            "force_diff": np.asarray(force_diff, dtype=float),
            "stress_diff": np.asarray(stress_diff, dtype=float),
            "sqomega": np.asarray(sqomega, dtype=float),
        },
    )()


def _dataset_for_features(displacement, ntime=1, strain=None, du_delta=None, ucvol=None):
    displacement = np.asarray(displacement, dtype=float)
    natom = displacement.shape[1]
    strain = np.zeros((ntime, 6), dtype=float) if strain is None else np.asarray(strain, dtype=float)
    du_delta = np.zeros((ntime, 6, natom, 3), dtype=float) if du_delta is None else np.asarray(du_delta, dtype=float)
    ucvol = np.ones(ntime, dtype=float) if ucvol is None else np.asarray(ucvol, dtype=float)
    return type(
        "FeatureDataset",
        (),
        {
            "displacement": displacement,
            "strain": strain,
            "du_delta": du_delta,
            "ucvol": ucvol,
        },
    )()


def _write_force_fit_hist(path: Path) -> Path:
    with netcdf_file(str(path), "w") as nc:
        nc.createDimension("time", 1)
        nc.createDimension("natom", 2)
        nc.createDimension("three", 3)
        nc.createDimension("six", 6)
        rprimd = nc.createVariable("rprimd", "d", ("time", "three", "three"))
        xred = nc.createVariable("xred", "d", ("time", "natom", "three"))
        xcart = nc.createVariable("xcart", "d", ("time", "natom", "three"))
        etotal = nc.createVariable("etotal", "d", ("time",))
        fcart = nc.createVariable("fcart", "d", ("time", "natom", "three"))
        strten = nc.createVariable("strten", "d", ("time", "six"))
        rprimd[:] = np.eye(3)[None, :, :]
        xred[:] = np.array([[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]])
        xcart[:] = np.array([[[0.0, 0.0, 0.0], [0.75, 0.0, 0.0]]])
        etotal[:] = np.array([0.0])
        fcart[:] = np.array([[[2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]]])
        strten[:] = np.zeros((1, 6))
    return path


def test_train_multibinit_model_invokes_binary_and_records_metadata(tmp_path):
    ddb = tmp_path / "input.ddb"
    hist = tmp_path / "training_HIST.nc"
    config = tmp_path / "train.abi"
    ddb.write_text("ddb", encoding="utf-8")
    hist.write_text("hist", encoding="utf-8")
    config.write_text("multibinit input", encoding="utf-8")
    executable = _write_fake_multibinit(
        tmp_path / "multibinit",
        "pathlib.Path('model.conf').write_text('trained', encoding='utf-8')\n"
        "pathlib.Path('argv.json').write_text(__import__('json').dumps(sys.argv), encoding='utf-8')\n"
        "print('training complete')\n"
        "sys.exit(0)\n",
    )

    result = train_multibinit_model(
        ddb=ddb,
        hist=hist,
        config=config,
        output_dir=tmp_path / "out",
        executable=executable,
        extra_args=["--flag", "value"],
    )

    assert result.returncode == 0
    assert result.model_config == str(tmp_path / "out" / "model.conf")
    assert Path(result.log_file).read_text(encoding="utf-8") == "training complete\n"
    argv = json.loads((tmp_path / "out" / "argv.json").read_text(encoding="utf-8"))
    assert argv == [str(executable), str(config), "--flag", "value"]

    metadata = json.loads((tmp_path / "out" / "pymultibinit_training_result.json").read_text(encoding="utf-8"))
    assert metadata["command"] == [str(executable), str(config), "--flag", "value"]
    assert metadata["ddb"] == str(ddb.resolve())
    assert metadata["hist"] == str(hist.resolve())
    assert metadata["config"] == str(config.resolve())


def test_train_multibinit_model_uses_environment_binary(tmp_path, monkeypatch):
    ddb = tmp_path / "input.ddb"
    hist = tmp_path / "training_HIST.nc"
    ddb.write_text("ddb", encoding="utf-8")
    hist.write_text("hist", encoding="utf-8")
    executable = _write_fake_multibinit(
        tmp_path / "env_multibinit",
        "print(os.environ['PYMULTIBINIT_DDB'])\n"
        "print(os.environ['PYMULTIBINIT_HIST'])\n"
        "sys.exit(0)\n",
    )
    monkeypatch.setenv("MULTIBINIT_BINARY", str(executable))

    result = train_multibinit_model(ddb=ddb, hist=hist, output_dir=tmp_path / "out")

    stdout = Path(result.log_file).read_text(encoding="utf-8").splitlines()
    assert stdout == [str(ddb.resolve()), str(hist.resolve())]


def test_train_multibinit_model_missing_binary_raises(tmp_path):
    ddb = tmp_path / "input.ddb"
    hist = tmp_path / "training_HIST.nc"
    ddb.write_text("ddb", encoding="utf-8")
    hist.write_text("hist", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="MULTIBINIT executable not found"):
        train_multibinit_model(
            ddb=ddb,
            hist=hist,
            output_dir=tmp_path / "out",
            executable=tmp_path / "does-not-exist",
        )


def test_train_multibinit_model_nonzero_exit_raises_with_logs(tmp_path):
    ddb = tmp_path / "input.ddb"
    hist = tmp_path / "training_HIST.nc"
    ddb.write_text("ddb", encoding="utf-8")
    hist.write_text("hist", encoding="utf-8")
    executable = _write_fake_multibinit(
        tmp_path / "multibinit",
        "print('bad input', file=sys.stderr)\n"
        "sys.exit(7)\n",
    )

    with pytest.raises(RuntimeError, match="MULTIBINIT training failed with exit code 7"):
        train_multibinit_model(
            ddb=ddb,
            hist=hist,
            output_dir=tmp_path / "out",
            executable=executable,
        )

    assert (tmp_path / "out" / "multibinit.stderr.log").read_text(encoding="utf-8") == "bad input\n"


def test_mbtools_train_cli_invokes_binary(tmp_path, monkeypatch):
    ddb = tmp_path / "input.ddb"
    hist = tmp_path / "training_HIST.nc"
    config = tmp_path / "train.abi"
    ddb.write_text("ddb", encoding="utf-8")
    hist.write_text("hist", encoding="utf-8")
    config.write_text("input", encoding="utf-8")
    executable = _write_fake_multibinit(
        tmp_path / "multibinit",
        "pathlib.Path('model.conf').write_text('trained', encoding='utf-8')\n"
        "sys.exit(0)\n",
    )
    outdir = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mbtools",
            "train",
            str(ddb),
            str(hist),
            "--config",
            str(config),
            "--output-dir",
            str(outdir),
            "--executable",
            str(executable),
        ],
    )

    assert main() == 0
    metadata = json.loads((outdir / "pymultibinit_training_result.json").read_text(encoding="utf-8"))
    assert metadata["model_config"] == str(outdir / "model.conf")


def test_mbtools_train_python_cli_writes_diagnostics(tmp_path, monkeypatch):
    ddb = tmp_path / "input.ddb"
    hist = tmp_path / "training_HIST.nc"
    basis = tmp_path / "basis.xml"
    output_xml = tmp_path / "fit.xml"
    diagnostics = tmp_path / "fit.json"
    ddb.write_text("ddb", encoding="utf-8")
    hist.write_text("hist", encoding="utf-8")
    basis.write_text("xml", encoding="utf-8")

    fake_result = SimpleNamespace(
        coefficients=np.array([1.5]),
        output_xml=str(output_xml),
        ncoeff=1,
        nframes=2,
        ddb=str(ddb),
        hist=str(hist),
        basis_xml=str(basis),
        diagnostics=SimpleNamespace(
            goal=SimpleNamespace(force_stress=1.0, force=0.5, stress=0.25, energy=0.125),
            residual_norm=0.75,
            matrix_rank=1,
            condition_number=2.0,
            regularization=0.0,
            info=0,
        ),
    )

    calls = {}

    def fake_fit(**kwargs):
        calls.update(kwargs)
        output_xml.write_text("<xml />", encoding="utf-8")
        return fake_result

    monkeypatch.setattr("pymultibinit.training.fit_multibinit_model_python", fake_fit)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mbtools",
            "train-python",
            str(ddb),
            str(hist),
            "--basis-xml",
            str(basis),
            "--output-xml",
            str(output_xml),
            "--diagnostics-json",
            str(diagnostics),
            "--ncell",
            "1",
            "1",
            "1",
        ],
    )

    assert main() == 0
    assert calls["ddb"] == str(ddb)
    assert calls["basis_xml"] == str(basis)
    data = json.loads(diagnostics.read_text(encoding="utf-8"))
    assert data["coefficients"] == [1.5]
    assert data["goal"]["force_stress"] == 1.0
    assert data["output_xml"] == str(output_xml)


def test_train_python_diagnostics_json_is_strict_for_infinite_values(tmp_path):
    from pymultibinit.cli import _python_fit_diagnostics

    result = SimpleNamespace(
        coefficients=np.array([1.0]),
        output_xml="fit.xml",
        ncoeff=1,
        nframes=1,
        ddb="d.ddb",
        hist="h.nc",
        basis_xml="b.xml",
        diagnostics=SimpleNamespace(
            goal=SimpleNamespace(force_stress=np.inf, force=0.0, stress=np.nan, energy=1.0),
            residual_norm=np.inf,
            matrix_rank=0,
            condition_number=np.inf,
            regularization=0.0,
            info=1,
        ),
    )

    text = json.dumps(_python_fit_diagnostics(result), allow_nan=False)
    assert "Infinity" not in text
    assert "NaN" not in text


def test_pair_key_normalization_inverse_orientation():
    key, sign = normalize_pair_key(PairKey(direction=0, atom_a=2, atom_b=0, cell_b=(1, 0, 0)))

    assert key == PairKey(direction=0, atom_a=0, atom_b=2, cell_b=(-1, 0, 0))
    assert sign == -1


def test_build_factor_action_map_deterministic_direction_signs():
    key = PairKey(direction=0, atom_a=0, atom_b=1, cell_b=(0, 0, 0))
    actions = build_factor_action_map([key], symrel=[np.diag([-1, 1, 1])])

    transformed, sign = actions[0][key]

    assert transformed == key
    assert sign == -1


def test_canonicalize_monomial_orbit_matches_symmetry_equivalent_terms():
    key_x = PairKey(direction=0, atom_a=0, atom_b=1, cell_b=(0, 0, 0))
    key_y = PairKey(direction=1, atom_a=0, atom_b=1, cell_b=(0, 0, 0))
    actions = build_factor_action_map(
        [key_x, key_y],
        symrel=[np.eye(3, dtype=int), np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])],
    )

    mono_x, sign_x = canonicalize_monomial_orbit(MonomialKey(((key_x, 2),)), actions)
    mono_y, sign_y = canonicalize_monomial_orbit(MonomialKey(((key_y, 2),)), actions)

    assert mono_x == mono_y
    assert sign_x == sign_y == 1


def test_factor_action_map_applies_atom_mapping_and_cell_shift():
    key = PairKey(direction=0, atom_a=0, atom_b=1, cell_b=(0, 0, 0))
    actions = build_factor_action_map(
        [key],
        symrel=[np.eye(3, dtype=int)],
        atom_mappings=[{0: (1, (0, 0, 0)), 1: (0, (1, 0, 0))}],
    )

    transformed, sign = actions[0][key]

    assert transformed == PairKey(direction=0, atom_a=0, atom_b=1, cell_b=(-1, 0, 0))
    assert sign == -1


def test_factor_action_map_accepts_build_atom_mapping_array_format():
    key = PairKey(direction=0, atom_a=0, atom_b=1, cell_b=(0, 0, 0))
    indsym = np.zeros((4, 1, 2), dtype=int)
    indsym[:, 0, 0] = [0, 0, 0, 1]
    indsym[:, 0, 1] = [1, 0, 0, 0]

    actions = build_factor_action_map([key], symrel=[np.eye(3, dtype=int)], atom_mappings=indsym)

    transformed, sign = actions[0][key]
    assert transformed == PairKey(direction=0, atom_a=0, atom_b=1, cell_b=(-1, 0, 0))
    assert sign == -1


def test_factor_action_map_uses_direct_mapping_translations():
    key = PairKey(direction=0, atom_a=0, atom_b=1, cell_b=(0, 0, 0))
    rotation = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    indsym = np.zeros((4, 1, 2), dtype=int)
    indsym[:, 0, 0] = [1, 0, 0, 1]
    indsym[:, 0, 1] = [0, 0, 0, 0]

    actions = build_factor_action_map([key], symrel=[rotation], atom_mappings=indsym)

    transformed, sign = actions[0][key]
    assert transformed == PairKey(direction=1, atom_a=0, atom_b=1, cell_b=(1, 0, 0))
    assert sign == -1


def test_canonicalize_monomial_orbit_has_deterministic_sign_tie():
    key = PairKey(direction=0, atom_a=0, atom_b=1, cell_b=(0, 0, 0))
    monomial = MonomialKey(((key, 1),))
    actions = [{key: (key, 1)}, {key: (key, -1)}]

    assert canonicalize_monomial_orbit(monomial, actions) == (monomial, -1)


def test_generate_displacement_basis_tiny_structure_is_deterministic_and_readable(tmp_path):
    basis = generate_displacement_basis(
        xcart=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        cutoff=1.5,
        power_range=(1, 1),
        include_pure_strain=False,
    )

    assert [item.text for item in basis] == ["u0_0_1", "u1_0_1", "u2_0_1"]
    out = write_fitted_xml(tmp_path / "generated.xml", basis)
    assert len(read_coefficient_xml(out)) == 3


def test_fortran_text_labels_match_multibinit_style():
    basis = generate_displacement_basis(
        xcart=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        cutoff=1.5,
        power_range=(2, 2),
        include_pure_strain=False,
    )

    labeled = with_fortran_text_labels(basis, ["Ti", "O", "O"])

    assert labeled[0].text == "(Ti_x-O1_x)^2"
    assert "(Ti_x-O1_x)^1(Ti_y-O1_y)^1" in [item.text for item in labeled]


def test_generate_displacement_basis_removes_symmetry_duplicate_orbits():
    basis = generate_displacement_basis(
        xcart=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        cutoff=1.5,
        power_range=(1, 1),
        symrel=[np.eye(3, dtype=int), np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])],
        include_pure_strain=False,
    )

    assert [item.text for item in basis] == ["u0_0_1", "u2_0_1"]
    assert len(basis[0].terms) == 2


def test_generate_displacement_basis_low_power_keys_are_stable():
    basis = generate_displacement_basis(
        xcart=np.array([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        cutoff=0.5,
        power_range=(2, 2),
        include_pure_strain=False,
    )

    assert [item.text for item in basis[:3]] == ["u0_0_1^2", "u0_0_1*u1_0_1", "u0_0_1*u2_0_1"]
    assert all(disp["atom_b"] != 2 for item in basis for disp in item.terms[0]["displacements"])


def test_generate_displacement_basis_can_include_linear_strain_couplings():
    basis = generate_displacement_basis(
        xcart=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        cutoff=1.5,
        power_range=(3, 4),
        include_strain_coupling=True,
        strain_voigts=(1, 4),
        include_pure_strain=False,
    )

    texts = [item.text for item in basis]
    assert "u0_0_1^3" in texts
    assert "u0_0_1^4" in texts
    assert "u0_0_1^2*eta1" in texts
    assert "u0_0_1^3*eta4" in texts
    strain_terms = [term for item in basis for term in item.terms if term["strains"]]
    assert strain_terms
    assert {term["strains"][0]["voigt"] for term in strain_terms} == {1, 4}


def test_generate_displacement_basis_includes_periodic_same_atom_pairs():
    basis = generate_displacement_basis(
        xcart=np.array([[0.0, 0.0, 0.0]]),
        cutoff=1.1,
        power_range=(1, 1),
        ncell=(2, 1, 1),
        rprimd=np.eye(3),
        include_pure_strain=False,
    )

    assert any(item.terms[0]["displacements"][0]["cell_b"] == (-1, 0, 0) for item in basis)


def test_displacement_pair_diagnostics_reports_symmetry_closure():
    diagnostics = displacement_pair_diagnostics(
        xcart=np.array([[0.0, 0.0, 0.0]]),
        cutoff=1.1,
        ncell=(2, 1, 1),
        rprimd=np.eye(3),
        symrel=[np.eye(3, dtype=int), np.diag([-1, 1, 1])],
    )

    assert diagnostics["n_factors"] == 3
    assert diagnostics["symmetry_closed"] is True
    assert diagnostics["missing_mapped_factors_count"] == 0


def test_generate_displacement_basis_filters_anti_invariant_linear_terms():
    basis = generate_displacement_basis(
        xcart=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        cutoff=1.5,
        power_range=(1, 1),
        symrel=[np.diag([-1, 1, 1])],
        include_pure_strain=False,
    )

    assert "u0_0_1" not in [item.text for item in basis]


def test_generate_displacement_basis_prunes_disconnected_periodic_monomials():
    basis = generate_displacement_basis(
        xcart=np.array([[0.0, 0.0, 0.0]]),
        cutoff=1.1,
        power_range=(2, 2),
        ncell=(3, 1, 1),
        rprimd=np.eye(3),
        include_pure_strain=False,
    )

    assert all(len({disp["cell_b"] for disp in item.terms[0]["displacements"]}) == 1 for item in basis)


def test_generate_fortran_pair_list_matches_batio3_pair_count_diagnostics():
    lattice = np.eye(3) * 7.49649813
    xred = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
        ]
    )
    xcart = xred @ lattice.T
    symrel = _cubic_signed_permutation_symrel()

    pair_lists = [generate_fortran_pair_list(xcart, xred, 8.0, iatom, symrel, ncell=(1, 1, 1), rprimd=lattice) for iatom in (0, 1, 2)]
    counts = [pair_list.ncoeff_sym for pair_list in pair_lists]
    combination_counts = [
        count_fortran_irreducible_pair_combinations(pair_list, power_range=(3, 4), ncell=(1, 1, 1), rprimd=lattice)
        for pair_list in pair_lists
    ]
    coefficient_counts = [
        count_fortran_displacement_coefficients(pair_list, power_range=(3, 4), ncell=(1, 1, 1), rprimd=lattice)
        for pair_list in pair_lists
    ]

    assert counts == [3, 3, 6]
    assert combination_counts == [25, 25, 182]
    assert coefficient_counts[:2] == [308, 246]
    # O1 still needs the final Fortran symmetry-reduction detail: Python gets 384 vs Fortran 365.
    assert coefficient_counts[2] == 384


def _cubic_signed_permutation_symrel():
    ops = []
    for permutation in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
        for signs in ((sx, sy, sz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)):
            matrix = np.zeros((3, 3), dtype=int)
            for row, column in enumerate(permutation):
                matrix[row, column] = signs[row]
            ops.append(matrix)
    return np.array(ops, dtype=int)


def test_mbtools_train_python_cli_reports_invalid_basis(tmp_path, monkeypatch, capsys):
    ddb = tmp_path / "input.ddb"
    hist = tmp_path / "training_HIST.nc"
    ddb.write_text("ddb", encoding="utf-8")
    hist.write_text("hist", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["mbtools", "train-python", str(ddb), str(hist), str(tmp_path / "missing.xml")])

    assert main() == 1
    assert "Basis XML file not found" in capsys.readouterr().err
