"""
Unit tests for dipole-dipole utilities.

How to run:
    pytest pymultibinit/tests/test_dipdip.py -v
"""

import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from pymultibinit.pyeffpot.dipdip import (
    dipole_dipole_tensor,
    dipole_dipole_ifc_block,
    build_dipole_dipole_ifcs,
)


def test_dipole_tensor_zero_self_term():
    tensor = dipole_dipole_tensor(np.zeros(3), np.eye(3))
    assert np.allclose(tensor, 0.0)


def test_dipole_ifc_block_shape():
    zeff = np.eye(3)
    block = dipole_dipole_ifc_block(np.array([1.0, 0.0, 0.0]), zeff, zeff, np.eye(3))
    assert block.shape == (3, 3)


def test_build_dipdip_ifcs_shape():
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    zeff = np.stack([np.eye(3), np.eye(3)], axis=2)
    out = build_dipole_dipole_ifcs(positions, np.eye(3), zeff)
    assert out.shape == (3, 2, 3, 2)
    assert np.allclose(out[:, 0, :, 0], 0.0)


def test_dipdip_active_for_ncell_below_ngqpt():
    """Regression: dipdip must stay active when ncell < ngqpt.

    A 2026-08-03 gate (any(ncell > ngqpt)) silently zeroed ewald_atmfrc for
    e.g. BaHfO3 ncell=(2,2,2) with ngqpt=(4,4,4), disabling dipole-dipole.
    The binary computes dipdip for any ncell (exact parity verified against
    libabinit 2026-08-26).
    """
    from pathlib import Path

    from pymultibinit import MultibinitPotential

    ddb = Path(__file__).resolve().parents[1] / "examples/BaHfO3_example/BaHfO3_DDB"
    if not ddb.exists():
        import pytest

        pytest.skip("BaHfO3 DDB fixture missing")
    pot = MultibinitPotential.from_pyeffpot(
        ddb_file=str(ddb), xml_file=None, ncell=(2, 2, 2), dipdip=True
    )
    ewald = pot._pyeffpot_potential.supercell.ifcs_sc.ewald_atmfrc
    assert ewald is not None
    assert abs(ewald).max() > 1e-6, "dipdip was silently disabled (ewald all zero)"
