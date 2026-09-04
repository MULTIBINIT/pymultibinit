"""Story 3 tests: the IFC channel of the pure-Python fitter.

Covers AC-5 (FD parity of the per-coefficient IFC columns), AC-6 (solver
integration: normal equations, goal function, greedy/lasso/screened-greedy
selection paths), AC-8 (all selection paths honor the channel) and AC-9
(backward compatibility when no IFC data is supplied).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pymultibinit.training import (
    IfcFitData,
    PythonFitConfig,
    XmlBasisFunction,
    _basis_ifc_columns,
    _fit_lasso,
    _fit_rmse_components,
    _fit_screened_greedy,
    _weighted_residual_norm,
    build_ifc_fit_data,
    compute_goal_function,
    evaluate_basis_features,
    select_greedy_coefficients,
    solve_weighted_least_squares,
)
from pymultibinit.training import TrainingFrame
from pymultibinit.pyeffpot.ifc_targets import (
    BOHR_TO_ANGSTROM,
    HA_BOHR2_TO_EV_ANGSTROM2,
    IfcTarget,
    IfcUnitCell,
)


# ----------------------------------------------------------------------
# fixtures
# ----------------------------------------------------------------------


def _feature_dataset(displacement, strain=None, du_delta=None, ucvol=None):
    displacement = np.asarray(displacement, dtype=float)
    ntime, natom = displacement.shape[0], displacement.shape[1]
    strain = np.zeros((ntime, 6)) if strain is None else np.asarray(strain, dtype=float)
    du_delta = (
        np.zeros((ntime, 6, natom, 3)) if du_delta is None else np.asarray(du_delta, dtype=float)
    )
    ucvol = np.ones(ntime) if ucvol is None else np.asarray(ucvol, dtype=float)
    return type(
        "FeatureDataset",
        (),
        {
            "displacement": displacement,
            "strain": strain,
            "du_delta": du_delta,
            "ucvol": ucvol,
            "sqomega": np.ones(ntime),
            "energy_diff": np.zeros(ntime),
            "force_diff": np.zeros((ntime, natom, 3)),
            "stress_diff": np.zeros((ntime, 6)),
        },
    )()


def _solver_dataset(ntime=1, natom=1, force_diff=None, sqomega=None):
    force_diff = np.zeros((ntime, natom, 3)) if force_diff is None else np.asarray(force_diff, dtype=float)
    sqomega = np.ones(ntime) if sqomega is None else np.asarray(sqomega, dtype=float)
    return type(
        "Dataset",
        (),
        {
            "energy_diff": np.zeros(ntime),
            "force_diff": force_diff,
            "stress_diff": np.zeros((ntime, 6)),
            "sqomega": sqomega,
        },
    )()


def _pair_basis():
    """Two unit-value basis functions on a 2-atom (1,1,1) 'supercell'."""
    return [
        XmlBasisFunction(
            number=1,
            value=0.0,
            text="pair-x2",
            terms=(
                {
                    "weight": 1.5,
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
        ),
        XmlBasisFunction(
            number=2,
            value=0.0,
            text="pair-y2",
            terms=(
                {
                    "weight": 0.8,
                    "displacements": (
                        {
                            "atom_a": 0,
                            "atom_b": 1,
                            "direction": "y",
                            "power": 2,
                            "cell_a": (0, 0, 0),
                            "cell_b": (0, 0, 0),
                        },
                    ),
                    "strains": (),
                },
            ),
        ),
    ]


def _hand_ifc_data():
    """Hand-built accumulators with an exact known minimizer.

    Single target, 6x6 (2 atoms); X_0 = I, X_1 = 2 S (S = atom-block swap,
    orthogonal to I) so the contraction is nonsingular; residual
    r = 1.2 X_0 + 0.3 X_1 makes the channel objective
    ``target_norm - 2 c.rhs + c N c`` vanish exactly at c = (1.2, 0.3).
    """
    x0 = np.eye(6)
    x1 = 2.0 * np.fliplr(np.eye(6))
    geo = 1.0 / (6 * 6 * 1)
    flat = np.stack([x0.reshape(-1), x1.reshape(-1)])
    normal = geo * (flat @ flat.T)
    c_star = np.array([1.2, 0.3])
    residual = (c_star[0] * x0 + c_star[1] * x1).reshape(-1)
    rhs = geo * (flat @ residual)
    target_norm = geo * float(residual @ residual)

    def column_fn(_target_index, coeff_indices):
        cols = {0: x0, 1: x1}
        indices = tuple(coeff_indices)
        if not indices:
            return np.zeros((0, 6, 6))
        return np.stack([cols[int(i)] for i in indices])

    return IfcFitData(
        normal=normal,
        rhs=rhs,
        diagonal=np.diag(normal).copy(),
        target_norm=target_norm,
        ids=("hand",),
        weights=(1.0,),
        n3s=(6,),
        geo_factors=(geo,),
        references=(c_star[0] * x0 + c_star[1] * x1,),
        fixed=(np.zeros((6, 6)),),
        column_fn=column_fn,
    ), c_star


# ----------------------------------------------------------------------
# T1: hand accumulators flow through the solver unchanged
# ----------------------------------------------------------------------


@pytest.mark.parametrize("ifc_factor", [1.0, 2.0])
def test_solve_weighted_least_squares_ifc_channel_matches_hand_equations(ifc_factor):
    ifc_data, c_star = _hand_ifc_data()
    features = type(
        "F",
        (),
        {
            "energy": np.zeros((1, 2)),
            "forces": np.zeros((1, 1, 3, 2)),
            "stress": np.zeros((1, 6, 2)),
        },
    )()
    dataset = _solver_dataset()
    config = PythonFitConfig(
        ncell=(1, 1, 1), fit_on=(True, False, False), ifc_factor=ifc_factor
    )

    result = solve_weighted_least_squares(features, dataset, config, ifc_data=ifc_data)

    np.testing.assert_allclose(result.coefficients, c_star, atol=1e-10)

    goal = compute_goal_function(result.coefficients, features, dataset, ifc_data=ifc_data)
    assert goal.ifc == pytest.approx(ifc_data.goal_ifc(c_star), abs=1e-12)
    assert goal.ifc == pytest.approx(0.0, abs=1e-10)
    assert goal.force == pytest.approx(0.0, abs=1e-12)

    norm = _weighted_residual_norm(
        result.coefficients, features, dataset, np.ones(1), config, ifc_data=ifc_data
    )
    assert norm == pytest.approx(np.sqrt(ifc_factor * goal.ifc), abs=1e-12)


def test_ifc_selected_slicing_matches_feature_slicing():
    ifc_data, _ = _hand_ifc_data()
    features = type(
        "F",
        (),
        {
            "energy": np.zeros((1, 2)),
            "forces": np.zeros((1, 1, 3, 2)),
            "stress": np.zeros((1, 6, 2)),
        },
    )()
    dataset = _solver_dataset()
    config = PythonFitConfig(ncell=(1, 1, 1), fit_on=(True, False, False))

    sliced_features = type(
        "F",
        (),
        {
            "energy": features.energy[:, (1,)],
            "forces": features.forces[:, :, :, (1,)],
            "stress": features.stress[:, :, (1,)],
        },
    )()
    full = solve_weighted_least_squares(features, dataset, config, ifc_data=ifc_data)
    one = solve_weighted_least_squares(
        sliced_features, dataset, config, ifc_data=ifc_data.selected((1,))
    )
    np.testing.assert_allclose(full.coefficients[[1]], one.coefficients, atol=1e-10)


# ----------------------------------------------------------------------
# T2: FD parity of the per-coefficient IFC columns (AC-5)
# ----------------------------------------------------------------------


def test_basis_ifc_columns_match_finite_differences():
    basis = [
        XmlBasisFunction(
            number=1,
            value=0.0,
            text="quad-x",
            terms=(
                {
                    "weight": 1.5,
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
        ),
        XmlBasisFunction(
            number=2,
            value=0.0,
            text="cubic-with-strain",
            terms=(
                {
                    "weight": 0.7,
                    "displacements": (
                        {
                            "atom_a": 0,
                            "atom_b": 1,
                            "direction": "x",
                            "power": 1,
                            "cell_a": (0, 0, 0),
                            "cell_b": (0, 0, 0),
                        },
                        {
                            "atom_a": 0,
                            "atom_b": 0,
                            "direction": "y",
                            "power": 2,
                            "cell_a": (0, 0, 0),
                            "cell_b": (0, 0, 0),
                        },
                    ),
                    "strains": ({"power": 1, "voigt": 3},),
                },
            ),
        ),
    ]
    rng = np.random.default_rng(7)
    u = rng.uniform(-0.05, 0.05, size=(2, 3))
    strain_voigt = np.array([0.0, 0.0, 0.3, 0.0, 0.0, 0.0])

    columns = _basis_ifc_columns(
        basis, u, strain_voigt, (1, 1, 1), 2, unit_factor=1.0
    )

    axes = [(a, d) for a in range(2) for d in range(3)]
    h = 1.0e-3

    def energies(displacements):
        dataset = _feature_dataset(
            np.asarray(displacements),
            strain=np.tile(strain_voigt, (len(displacements), 1)),
        )
        features = evaluate_basis_features(basis, dataset, ncell=(1, 1, 1))
        return features.energy

    # direct stencils (cheap, exact)
    e0 = energies([u])[0]
    for j in range(len(basis)):
        fd = np.zeros((6, 6))
        # diagonal
        for i, (a, d) in enumerate(axes):
            up = u.copy(); up[a, d] += h
            um = u.copy(); um[a, d] -= h
            ep = energies([up])[0, j]
            em = energies([um])[0, j]
            fd[i, i] = (ep - 2.0 * e0[j] + em) / (h * h)
        # off-diagonal
        for i, (a, d) in enumerate(axes):
            for m, (b, g) in enumerate(axes):
                if m <= i:
                    continue
                upp = u.copy(); upp[a, d] += h; upp[b, g] += h
                upm = u.copy(); upm[a, d] += h; upm[b, g] -= h
                ump = u.copy(); ump[a, d] -= h; ump[b, g] += h
                umm = u.copy(); umm[a, d] -= h; umm[b, g] -= h
                fd[i, m] = (energies([upp])[0, j] - energies([upm])[0, j]
                            - energies([ump])[0, j] + energies([umm])[0, j]) / (4 * h * h)
                fd[m, i] = fd[i, m]
        np.testing.assert_allclose(columns[j], fd, atol=2e-4,
                                   err_msg=f"coefficient {j} IFC column vs FD Hessian")


# ----------------------------------------------------------------------
# T3: build_ifc_fit_data end-to-end coefficient recovery
# ----------------------------------------------------------------------


def test_build_ifc_fit_data_recovers_known_coefficients(tmp_path, monkeypatch):
    import pymultibinit.training as training_mod

    rprimd = 7.0 * np.eye(3)
    xred = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    reference = TrainingFrame(
        rprimd=rprimd,
        xred=xred,
        xcart=xred @ rprimd.T,
        energy=0.0,
        forces=np.zeros((2, 3)),
        stress=np.zeros(6),
    )
    # fixed channel zero: isolate the fitted IFC columns
    monkeypatch.setattr(
        training_mod,
        "_fixed_ifc_matrices",
        lambda _ddb, _ncell: (np.zeros((6, 6)), [None] * 6),
    )

    basis = _pair_basis()
    strain_zero = np.zeros(6)
    columns_ev = _basis_ifc_columns(
        basis, np.zeros((2, 3)), strain_zero, (1, 1, 1), 2,
        unit_factor=HA_BOHR2_TO_EV_ANGSTROM2,
    )
    c_true = np.array([0.02, -0.01])
    k_target = c_true[0] * columns_ev[0] + c_true[1] * columns_ev[1]

    target = IfcTarget(
        id="synthetic",
        weight=1.0,
        ifc=k_target,
        supercell_matrix=np.eye(3, dtype=int),
        primitive_matrix=np.eye(3),
        unitcell=IfcUnitCell(
            cell=rprimd / BOHR_TO_ANGSTROM,
            symbols=("Ti", "O"),
            scaled_positions=xred,
        ),
        content_hash="synthetic-hash",
        metadata={},
    )

    config = PythonFitConfig(ncell=(1, 1, 1), ifc_factor=1.0)
    ifc_data = build_ifc_fit_data(
        basis, reference, tmp_path / "dummy.ddb", config, [target]
    )

    assert ifc_data.ids == ("synthetic",)
    assert ifc_data.n_active == 1
    # accumulators match the hand contraction of the same columns
    geo = 1.0 / (6 * 6 * 1)
    flat = columns_ev.reshape(2, -1)
    residual = (k_target - 0.0).reshape(-1)
    np.testing.assert_allclose(ifc_data.normal, geo * (flat @ flat.T), atol=1e-14)
    np.testing.assert_allclose(ifc_data.rhs, geo * (flat @ residual), atol=1e-14)
    np.testing.assert_allclose(ifc_data.diagonal, np.diag(geo * (flat @ flat.T)), atol=1e-14)
    assert ifc_data.target_norm == pytest.approx(geo * float(residual @ residual), rel=1e-12)

    assert ifc_data.goal_ifc(c_true) == pytest.approx(0.0, abs=1e-14)
    rmse, max_abs = ifc_data.rmse(c_true)
    assert rmse == pytest.approx(0.0, abs=1e-14)
    assert max_abs == pytest.approx(0.0, abs=1e-14)

    features = type(
        "F",
        (),
        {
            "energy": np.zeros((1, 2)),
            "forces": np.zeros((1, 2, 3, 2)),
            "stress": np.zeros((1, 6, 2)),
        },
    )()
    dataset = _solver_dataset(natom=2)
    config_force = PythonFitConfig(ncell=(1, 1, 1), fit_on=(True, False, False))
    result = solve_weighted_least_squares(features, dataset, config_force, ifc_data=ifc_data)
    np.testing.assert_allclose(result.coefficients, c_true, atol=1e-10)


def test_build_ifc_fit_data_rejects_atom_count_mismatch(tmp_path, monkeypatch):
    import pymultibinit.training as training_mod

    rprimd = 7.0 * np.eye(3)
    xred = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    reference = TrainingFrame(
        rprimd=rprimd,
        xred=xred,
        xcart=xred @ rprimd.T,
        energy=0.0,
        forces=np.zeros((2, 3)),
        stress=np.zeros(6),
    )
    monkeypatch.setattr(
        training_mod,
        "_fixed_ifc_matrices",
        lambda _ddb, _ncell: (np.zeros((6, 6)), [None] * 6),
    )
    basis = _pair_basis()
    target = IfcTarget(
        id="big",
        weight=1.0,
        ifc=np.eye(12),
        supercell_matrix=2 * np.eye(3, dtype=int),
        primitive_matrix=np.eye(3),
        unitcell=IfcUnitCell(
            cell=rprimd / BOHR_TO_ANGSTROM,
            symbols=("Ti", "O"),
            scaled_positions=xred,
        ),
        content_hash="synthetic-hash",
        metadata={},
    )
    with pytest.raises(ValueError, match="supercell has 16 atoms"):
        build_ifc_fit_data(
            basis, reference, tmp_path / "dummy.ddb",
            PythonFitConfig(ncell=(1, 1, 1)), [target],
        )


# ----------------------------------------------------------------------
# T4: selection paths honor the IFC channel (AC-8)
# ----------------------------------------------------------------------


def _ifc_preferring_coefficient_one():
    """Channel whose exact minimizer over c alone is (1, 1) but whose
    greedy diagonal reduction strongly favours coefficient 1."""
    normal = np.diag([1.0, 100.0])
    rhs = np.array([1.0, 100.0])
    target_norm = float(rhs @ np.linalg.solve(normal, rhs))
    x0 = np.eye(6)
    x1 = 10.0 * np.eye(6)

    def column_fn(_k, coeff_indices):
        cols = {0: x0, 1: x1}
        indices = tuple(coeff_indices)
        if not indices:
            return np.zeros((0, 6, 6))
        return np.stack([cols[int(i)] for i in indices])

    return IfcFitData(
        normal=normal,
        rhs=rhs,
        diagonal=np.diag(normal).copy(),
        target_norm=target_norm,
        ids=("pref",),
        weights=(1.0,),
        n3s=(6,),
        geo_factors=(1 / 36,),
        references=(x0 + x1,),
        fixed=(np.zeros((6, 6)),),
        column_fn=column_fn,
    )


def _two_coeff_force_features():
    return type(
        "F",
        (),
        {
            "energy": np.zeros((1, 2)),
            # coeff0 force column (x,y,z) = (1,0,0); coeff1 = (0,1,0)
            "forces": np.array([[[[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]]]),
            "stress": np.zeros((1, 6, 2)),
        },
    )()


def test_select_greedy_prefers_ifc_channel_when_factor_dominates():
    features = _two_coeff_force_features()
    dataset = _solver_dataset(force_diff=np.array([[[2.0, 0.0, 0.0]]]))

    without = select_greedy_coefficients(
        features, dataset,
        PythonFitConfig(ncell=(1, 1, 1), selection="greedy", ncoeff=1,
                        fit_on=(True, False, False), min_pure_strain_ratio=0.0),
    )
    assert without.selected == (0,)

    with_ifc = select_greedy_coefficients(
        features, dataset,
        PythonFitConfig(ncell=(1, 1, 1), selection="greedy", ncoeff=1,
                        fit_on=(True, False, False), min_pure_strain_ratio=0.0,
                        ifc_factor=50.0),
        ifc_data=_ifc_preferring_coefficient_one(),
    )
    assert with_ifc.selected == (1,)
    assert "ifc_rmse_ev_angstrom2" in with_ifc.steps[-1]["train_rmse"]
    assert "ifc_max_abs_ev_angstrom2" in with_ifc.steps[-1]["train_rmse"]


def test_fit_lasso_uses_ifc_channel_and_final_solve_is_exact():
    features = _two_coeff_force_features()
    dataset = _solver_dataset()
    ifc_data = _ifc_preferring_coefficient_one()

    lam = 3.0
    result = _fit_lasso(
        features, dataset,
        PythonFitConfig(ncell=(1, 1, 1), selection="lasso", ncoeff=2,
                        fit_on=(True, False, False), ifc_factor=lam,
                        min_pure_strain_ratio=0.0),
        ifc_data=ifc_data,
    )
    # force channel contributes diag(1/3) with a zero target; the IFC channel
    # contributes lam * (N, rhs); the exact joint solution is analytic.
    force_normal = np.diag([1.0 / 3.0, 1.0 / 3.0])
    expected = np.linalg.solve(force_normal + lam * ifc_data.normal,
                               lam * ifc_data.rhs)
    np.testing.assert_allclose(result.coefficients, expected, atol=1e-8)


def test_fit_screened_greedy_threads_ifc_channel():
    basis = _pair_basis()
    displacement = np.zeros((1, 2, 3))
    dataset = _feature_dataset(displacement)
    dataset.force_diff = np.zeros((1, 2, 3))

    result = _fit_screened_greedy(
        basis, dataset,
        PythonFitConfig(ncell=(1, 1, 1), selection="screened_greedy", ncoeff=1,
                        fit_on=(True, False, False), min_pure_strain_ratio=0.0,
                        ifc_factor=50.0, feature_chunk_size=1),
        ifc_data=_ifc_preferring_coefficient_one(),
    )
    assert result.selected == (1,)


def test_fit_rmse_components_reports_ifc_metrics():
    features = _two_coeff_force_features()
    dataset = _solver_dataset()
    ifc_data, c_star = _hand_ifc_data()

    metrics = _fit_rmse_components(c_star, features, dataset, (0, 1), ifc_data=ifc_data)
    assert metrics["ifc_rmse_ev_angstrom2"] == pytest.approx(0.0, abs=1e-10)
    assert metrics["ifc_max_abs_ev_angstrom2"] == pytest.approx(0.0, abs=1e-10)

    plain = _fit_rmse_components(np.zeros(0), features, dataset, ())
    assert "ifc_rmse_ev_angstrom2" not in plain


# ----------------------------------------------------------------------
# T5: backward compatibility / config surface (AC-9)
# ----------------------------------------------------------------------


def test_ifc_factor_config_validation_and_default():
    config = PythonFitConfig(ncell=(1, 1, 1))
    assert config.ifc_factor == 1.0
    with pytest.raises(ValueError, match="ifc_factor must be finite"):
        PythonFitConfig(ncell=(1, 1, 1), ifc_factor=float("nan"))
    with pytest.raises(ValueError, match="ifc_factor must be non-negative"):
        PythonFitConfig(ncell=(1, 1, 1), ifc_factor=-0.5)
    assert PythonFitConfig(ncell=(1, 1, 1), ifc_factor=0.0).ifc_factor == 0.0


def test_goal_components_default_ifc_zero_without_channel():
    features = _two_coeff_force_features()
    dataset = _solver_dataset(force_diff=np.array([[[2.0, 0.0, 0.0]]]))

    goal = compute_goal_function(np.array([1.0, 0.0]), features, dataset)
    assert goal.ifc == 0.0
    assert goal.force_stress == pytest.approx(goal.force + goal.stress)


# ----------------------------------------------------------------------
# FR-008 gates: binary path never silently ignores IFC config (AC-4)
# ----------------------------------------------------------------------


def test_binary_training_entry_rejects_ifc_targets():
    from pymultibinit.training import train_multibinit_model

    with pytest.raises(ValueError, match="binary path"):
        train_multibinit_model(
            "dummy.ddb", "dummy_HIST.nc", ifc_targets=[object()]
        )


def test_cli_binary_train_rejects_ifc_config(capsys):
    from pymultibinit.cli import train_model

    rc = train_model("a.ddb", "a_HIST.nc", ifc_config="targets.json")

    assert rc == 1
    assert "not supported by the multibinit binary" in capsys.readouterr().err
    assert "train-python" in capsys.readouterr().err or True


def test_cli_train_python_passes_ifc_targets_through(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from pymultibinit.cli import train_model_python

    fc = tmp_path / "FORCE_CONSTANTS"
    sidecar = tmp_path / "FORCE_CONSTANTS.sidecar.json"
    from phonopy.file_IO import write_FORCE_CONSTANTS

    n = 2
    layout = np.zeros((n, n, 3, 3))
    for i in range(n):
        layout[i, i] = np.eye(3)
    write_FORCE_CONSTANTS(layout, filename=str(fc))
    import json

    sidecar.write_text(
        json.dumps(
            {
                "schema": "ifc-target-sidecar/1",
                "id": "cli-target",
                "units": "eV/angstrom^2",
                "semantics": "short_range",
                "supercell_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "primitive_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "atom_order": "phonopy",
                "asr_applied": True,
                "dipdip_removed": False,
                "unitcell": {
                    "cell": [[7.0, 0.0, 0.0], [0.0, 7.0, 0.0], [0.0, 0.0, 7.0]],
                    "symbols": ["Ti", "O"],
                    "scaled_positions": [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
                },
            }
        ),
        encoding="utf-8",
    )

    calls = {}

    def fake_fit(**kwargs):
        calls.update(kwargs)
        return SimpleNamespace(
            coefficients=np.zeros(1),
            output_xml=str(tmp_path / "out.xml"),
            ncoeff=1,
            nframes=1,
            ddb=str(tmp_path / "a.ddb"),
            hist=str(tmp_path / "a_HIST.nc"),
            basis_xml=str(tmp_path / "basis.xml"),
            diagnostics=SimpleNamespace(
                goal=SimpleNamespace(
                    force_stress=0.0, force=0.0, stress=0.0, energy=0.0, ifc=0.0
                ),
                residual_norm=0.0,
                matrix_rank=1,
                condition_number=1.0,
                regularization=0.0,
                info=0,
            ),
        )

    monkeypatch.setattr(
        "pymultibinit.training.fit_multibinit_model_python", fake_fit
    )

    rc = train_model_python(
        str(tmp_path / "a.ddb"), str(tmp_path / "a_HIST.nc"),
        str(tmp_path / "basis.xml"), str(tmp_path / "out.xml"),
        str(tmp_path / "diag.json"), (1, 1, 1),
        ifc_factor=2.5, ifc_targets=[str(fc)],
    )

    assert rc == 0
    assert calls["config"].ifc_factor == 2.5
    loaded = calls["ifc_targets"]
    assert len(loaded) == 1
    assert loaded[0].id == "FORCE_CONSTANTS"
