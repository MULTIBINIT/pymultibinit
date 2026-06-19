from pathlib import Path

import numpy as np

from pymultibinit import MultibinitPotential
from pymultibinit.pyeffpot.xml_parser import read_coefficient_xml
from pymultibinit.training import PythonFitConfig, fit_multibinit_model_python


def test_batio3_pure_python_training_and_evaluation_workflow(tmp_path):
    root = Path(__file__).resolve().parents[1]
    ddb = root / "examples/BaHfO3_example/BaHfO3_DDB"
    hist = root / "examples/BaHfO3_training/real_training_run/BaTiO3_multibinit_HIST.nc"
    basis_xml = root / "examples/BaHfO3_training/real_training_run/wrapper_run_qgrid/BaTiO3_fit_coeffs.xml"
    output_xml = tmp_path / "BaHfO3_fit_python.xml"

    result = fit_multibinit_model_python(
        ddb=ddb,
        hist=hist,
        basis_xml=basis_xml,
        output_xml=output_xml,
        config=PythonFitConfig(
            ncell=(2, 2, 2),
            selection="greedy",
            ncoeff=3,
            regularization=1e-8,
        ),
    )

    assert result.nframes == 8
    assert result.ncoeff == 3
    assert result.diagnostics.info == 0
    assert result.diagnostics.matrix_rank == 3
    assert np.isfinite(result.diagnostics.residual_norm)
    assert np.count_nonzero(np.abs(result.coefficients) > 0.0) == 3
    assert len(read_coefficient_xml(output_xml)) == 3

    potential = MultibinitPotential.from_pyeffpot(str(ddb), xml_file=str(output_xml), ncell=(2, 2, 2))
    structure = potential.get_supercell_structure()
    assert structure is not None
    positions, lattice, _ = structure

    energy, forces, stress = potential.evaluate(positions, lattice, skip_atom_matching=True)

    assert positions.shape == (40, 3)
    assert np.isfinite(energy)
    assert forces.shape == (40, 3)
    assert stress.shape == (6,)
    assert np.all(np.isfinite(forces))
    assert np.all(np.isfinite(stress))
    assert np.linalg.norm(forces) < 1e-8
