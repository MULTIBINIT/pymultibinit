import os
from pathlib import Path

import numpy as np
import pytest

from pymultibinit.pyeffpot.ddb_parser_complete import read_ddb
from pymultibinit.pyeffpot.potential import EffectivePotential
from pymultibinit.wrapper_cffi import MultibinitWrapperCFFI


def test_full_qgrid_supercell_displacement_parity():
    ddb_path = (
        Path(__file__).parents[2]
        / "atomchain/examples/08_training_set_strategies_batio3"
        / "batio3_npt_md_relaxed/ddb/model.ddb"
    )
    if not ddb_path.exists():
        pytest.skip(f"DDB file not found: {ddb_path}")
    if not os.environ.get("LIBABINIT_PATH"):
        pytest.skip("LIBABINIT_PATH is required for CFFI parity")

    fortran = None
    try:
        fortran = MultibinitWrapperCFFI()
        fortran.init_from_params(str(ddb_path), ncell=(2, 2, 2), ngqpt=(2, 2, 2), dipdip=0)
    except Exception as exc:
        pytest.skip(f"CFFI libabinit is not available: {exc}")
    assert fortran is not None

    _, _, reference_positions, reference_lattice = fortran.get_supercell_structure()
    python = EffectivePotential.from_files(str(ddb_path), ncell=(2, 2, 2), dipdip=False)
    positions = reference_positions.copy()
    positions[0, 0] += 0.01

    energy_fortran, forces_fortran, _ = fortran.evaluate(positions, reference_lattice)
    energy_python, forces_python, _ = python.evaluate(positions, reference_lattice)

    assert energy_python == pytest.approx(energy_fortran, abs=1e-12)
    assert np.max(np.abs(forces_python - forces_fortran)) < 1e-12
    if fortran is not None:
        fortran.free()


def test_dipdip_bec_dielectric_ddb_matches_cffi_contribution():
    ddb_path = Path(__file__).parents[1] / "examples/BaHfO3_example/BaHfO3_DDB"
    if not ddb_path.exists():
        pytest.skip(f"DDB file not found: {ddb_path}")
    if not os.environ.get("LIBABINIT_PATH"):
        pytest.skip("LIBABINIT_PATH is required for CFFI parity")

    unitcell = read_ddb(str(ddb_path))
    assert unitcell.epsilon_inf is not None
    assert unitcell.zeff is not None
    assert np.max(np.abs(unitcell.zeff)) > 1e-8

    fortran_without = None
    fortran_with = None
    try:
        fortran_without = MultibinitWrapperCFFI()
        fortran_without.init_from_params(str(ddb_path), ncell=(1, 1, 1), ngqpt=(4, 4, 4), dipdip=0)
        fortran_with = MultibinitWrapperCFFI()
        fortran_with.init_from_params(str(ddb_path), ncell=(1, 1, 1), ngqpt=(4, 4, 4), dipdip=1)
    except Exception as exc:
        pytest.skip(f"CFFI libabinit is not available: {exc}")
    assert fortran_without is not None
    assert fortran_with is not None

    _, _, reference_positions, reference_lattice = fortran_without.get_supercell_structure()
    positions = reference_positions.copy()
    positions[0, 0] += 0.001

    energy_fortran_without, forces_fortran_without, _ = fortran_without.evaluate(positions, reference_lattice)
    energy_fortran_with, forces_fortran_with, _ = fortran_with.evaluate(positions, reference_lattice)
    python_without = EffectivePotential.from_files(str(ddb_path), ncell=(1, 1, 1), dipdip=False)
    python_with = EffectivePotential.from_files(str(ddb_path), ncell=(1, 1, 1), dipdip=True)
    energy_python_without, forces_python_without, _ = python_without.evaluate(positions, reference_lattice)
    energy_python_with, forces_python_with, _ = python_with.evaluate(positions, reference_lattice)

    force_delta_fortran = forces_fortran_with - forces_fortran_without
    force_delta_python = forces_python_with - forces_python_without
    assert (energy_python_with - energy_python_without) == pytest.approx(
        energy_fortran_with - energy_fortran_without, abs=1e-12
    )
    assert np.max(np.abs(force_delta_python - force_delta_fortran)) < 1e-12
    if fortran_without is not None:
        fortran_without.free()
    if fortran_with is not None:
        fortran_with.free()
