import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np


AUDIT = Path(__file__).parents[2] / "debugs/BFO_arijit_harmonic_update/audit_standalone_parity.py"
SPEC = importlib.util.spec_from_file_location("bfo_standalone_audit", AUDIT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot load audit module: {AUDIT}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_test_set_input_uses_hist_stress_convention_without_fitting():
    text = MODULE.test_input(dipdip=True, asr=2)
    assert "fit_coeff 0" in text
    assert "test_effpot 1" in text
    assert "ts_option 0" in text


def test_test_set_input_supports_disabled_asr_minus_one():
    assert "asr -1" in MODULE.test_input(dipdip=True, asr=-1)


def test_test_set_files_use_sixth_legacy_slot():
    lines = MODULE.test_files("in", "out", "ddb", "hist").splitlines()
    assert lines == ["in", "out", "ddb", "", "", "hist"]


def test_audit_defaults_to_the_ordered_bfo_hist_fixture():
    assert MODULE.run_audit.__defaults__[1] == (
        MODULE.ROOT / "debugs/BFO_arijit_harmonic/BFO-harmonic-ordered-again.nc"
    )


def test_parse_test_goal_requires_test_set_marker():
    with unittest.TestCase().assertRaisesRegex(RuntimeError, "TEST - SET"):
        MODULE.parse_test_goal("Energy : 1.0E+00")


def test_parse_test_goal_keeps_multibinit_eV_squared_per_angstrom_squared_units():
    text = (
        "TEST - SET Option\n"
        " Energy          :   1.0000000000000000E+00\n"
        " Forces+Stresses :   2.0000000000000000E+00\n"
        " Forces          :   3.0000000000000000E+00\n"
        " Stresses        :   4.0000000000000000E+00\n"
    )
    assert MODULE.parse_test_goal(text) == {
        "energy_eV2_A2": 1.0,
        "forces_stress_eV2_A2": 2.0,
        "forces_eV2_A2": 3.0,
        "stress_eV2_A2": 4.0,
    }


def test_evaluator_outputs_are_normalized_to_hist_atomic_units():
    energy, forces, stress = MODULE.evaluator_to_hist_units(
        MODULE.HARTREE_TO_EV,
        np.array([[MODULE.HARTREE_TO_EV / MODULE.BOHR_TO_ANGSTROM, 0.0, 0.0]]),
        np.full(6, MODULE.HARTREE_TO_EV / MODULE.BOHR_TO_ANGSTROM ** 3),
    )
    assert energy == 1.0
    np.testing.assert_allclose(forces, [[1.0, 0.0, 0.0]])
    np.testing.assert_allclose(stress, np.ones(6))


def test_parse_test_goal_from_actual_abinit_pymb_execution():
    result = MODULE.run_standalone(
        MODULE.HERE / "BFO-ref-arijit.ddb.out",
        MODULE.ROOT / "debugs/BFO_arijit_harmonic/BFO-harmonic-ordered-again.nc",
        dipdip=True,
        asr=2,
    )
    assert all(np.isfinite(value) and value >= 0.0 for value in result.values())


def test_direct_ddb_asr_minus_one_rejection_is_explicit():
    with unittest.TestCase().assertRaisesRegex(RuntimeError, "Wrong value for asr: -1"):
        MODULE.run_standalone_artifact(
            MODULE.HERE / "BFO-ref-arijit.ddb.out",
            MODULE.ROOT / "debugs/BFO_arijit_harmonic/BFO-harmonic-ordered-again.nc",
            dipdip=True,
            asr=-1,
        )


def test_standalone_test_artifact_has_per_frame_energy_force_and_stress():
    output = MODULE.run_standalone_artifact(
        MODULE.HERE / "BFO-ref-arijit.ddb.out",
        MODULE.ROOT / "debugs/BFO_arijit_harmonic/BFO-harmonic-ordered-again.nc",
        dipdip=True,
        asr=2,
    )
    assert output["energy_Ha"].shape == (4319,)
    assert output["forces_Ha_Bohr"].shape == (4319, 40, 3)
    assert output["stress_Ha_Bohr3"].shape == (4319, 6)


def test_standalone_shift_and_permutation_controls_use_temporary_histories():
    ddb = MODULE.HERE / "BFO-ref-arijit.ddb.out"
    hist = MODULE.ROOT / "debugs/BFO_arijit_harmonic/BFO-harmonic-ordered-again.nc"
    baseline = MODULE.run_standalone_artifact(ddb, hist, dipdip=True, asr=2)
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        shifted = directory / "shifted.nc"
        permuted = directory / "permuted.nc"
        MODULE.write_transformed_hist(hist, shifted, shift_frame=1)
        MODULE.write_transformed_hist(
            hist, permuted, permutation=np.roll(np.arange(40), 1)
        )
        shifted_output = MODULE.run_standalone_artifact(ddb, shifted, dipdip=True, asr=2)
        permuted_output = MODULE.run_standalone_artifact(ddb, permuted, dipdip=True, asr=2)
    assert np.max(abs(shifted_output["energy_Ha"] - baseline["energy_Ha"])) > 1e-6
    np.testing.assert_allclose(permuted_output["energy_Ha"], baseline["energy_Ha"])
    np.testing.assert_allclose(permuted_output["forces_Ha_Bohr"], baseline["forces_Ha_Bohr"])
    np.testing.assert_allclose(permuted_output["stress_Ha_Bohr3"], baseline["stress_Ha_Bohr3"])


def test_test_only_wrapping_and_mapping_controls_are_measurable():
    controls = MODULE.test_only_controls(
        MODULE.HERE / "BFO-ref-arijit.ddb.out",
        MODULE.ROOT / "debugs/BFO_arijit_harmonic/BFO-harmonic-ordered-again.nc",
    )
    wrapping = controls["wrapping"]
    invariant = wrapping["minimum_image_invariance"]
    raw = wrapping["raw_minus_minimum_image"]
    assert invariant["energy_abs_eV"] < 1e-8
    assert invariant["force_max_abs_eV_A"] < 1e-7
    assert invariant["stress_max_abs_eV_A3"] < 1e-10
    assert abs(raw["energy_eV"]) > 1e-3
    assert raw["force_max_abs_eV_A"] > 1e-3
    assert raw["stress_max_abs_eV_A3"] > 1e-6
    permutation = controls["mapping_permutation"]
    assert permutation["mapping_is_bijection"]
    assert permutation["energy_abs_eV"] < 1e-12
    assert permutation["stress_max_abs_eV_A3"] < 1e-12
    assert permutation["model_force_permutation_max_abs_eV_A"] < 1e-12
    assert permutation["hist_force_inverse_mapping_max_abs_Ha_Bohr"] < 1e-12
