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
