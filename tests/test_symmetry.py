"""Unit tests for symmetry operations in pyeffpot."""

import numpy as np
import pytest

from pymultibinit.pyeffpot.symmetry import (
    get_reciprocal_symmetry,
    find_equivalent_atom,
    build_atom_mapping,
    symmetry_to_cartesian,
    check_q_symmetry,
    find_symmetry_for_qpoint,
)


class TestReciprocalSymmetry:
    """Tests for get_reciprocal_symmetry function."""

    def test_identity_rotation(self):
        """Identity rotation should return identity."""
        symrel = np.eye(3, dtype=int)
        symrec = get_reciprocal_symmetry(symrel)
        np.testing.assert_array_equal(symrec, symrel)

    def test_c4_rotation(self):
        """C4 rotation around z-axis: symrec = (S^-1)^T (ABINIT mati3inv)."""
        symrel = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ], dtype=int)
        symrec = get_reciprocal_symmetry(symrel)
        expected = np.linalg.inv(symrel).T
        np.testing.assert_allclose(symrec, expected, atol=1e-10)

    def test_multiple_symmetries(self):
        """Test with multiple symmetry operations."""
        symrel = np.array([
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
        ], dtype=int)
        symrec = get_reciprocal_symmetry(symrel)
        assert symrec.shape == (2, 3, 3)


class TestFindEquivalentAtom:
    """Tests for find_equivalent_atom function."""

    def test_find_same_atom(self):
        """Finding an atom at its own position."""
        xred_all = np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
        ])
        iatom, trans = find_equivalent_atom(np.array([0.0, 0.0, 0.0]), xred_all)
        assert iatom == 0
        np.testing.assert_array_equal(trans, [0, 0, 0])

    def test_find_with_translation(self):
        """Finding atom with lattice translation."""
        xred_all = np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
        ])
        iatom, trans = find_equivalent_atom(np.array([1.0, 0.0, 0.0]), xred_all)
        assert iatom == 0
        np.testing.assert_array_equal(trans, [1, 0, 0])


class TestBuildAtomMapping:
    """Tests for build_atom_mapping function."""

    def test_identity_symmetry(self):
        """Identity symmetry should map atoms to themselves."""
        xred = np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
        ])
        rotations = np.array([np.eye(3, dtype=int)])
        translations = np.zeros((1, 3))
        
        indsym = build_atom_mapping(xred, rotations, translations)
        
        assert indsym.shape == (4, 1, 2)
        assert indsym[3, 0, 0] == 0
        assert indsym[3, 0, 1] == 1


class TestCheckQSymmetry:
    """Tests for check_q_symmetry function."""

    def test_check_q_symmetry_direct(self):
        """Direct symmetry: q = S @ q."""
        q = np.array([0.25, 0.0, 0.0])
        q_ref = np.array([0.25, 0.0, 0.0])
        symrel = np.eye(3, dtype=int)
        
        is_direct, is_inverse = check_q_symmetry(q, q_ref, symrel)
        assert is_direct is True
        assert is_inverse is False

    def test_check_q_symmetry_time_reversal(self):
        """Time reversal: q = -S @ q."""
        q = np.array([-0.25, 0.0, 0.0])
        q_ref = np.array([0.25, 0.0, 0.0])
        symrel = np.eye(3, dtype=int)
        
        is_direct, is_inverse = check_q_symmetry(q, q_ref, symrel)
        assert is_direct is False
        assert is_inverse is True


class TestFindSymmetryForQpoint:
    """Tests for find_symmetry_for_qpoint function."""

    def test_find_symmetry_for_qpoint(self):
        """Find symmetry mapping between q-points."""
        q_target = np.array([0.25, 0.0, 0.0])
        q_ibz = np.array([0.25, 0.0, 0.0])
        rotations = np.array([np.eye(3, dtype=int)])
        
        isym, time_reversal = find_symmetry_for_qpoint(q_target, q_ibz, rotations)
        assert isym == 0
        assert time_reversal is False

    def test_find_symmetry_with_time_reversal(self):
        """Find symmetry with time reversal."""
        q_target = np.array([-0.25, 0.0, 0.0])
        q_ibz = np.array([0.25, 0.0, 0.0])
        rotations = np.array([np.eye(3, dtype=int)])
        
        isym, time_reversal = find_symmetry_for_qpoint(q_target, q_ibz, rotations)
        assert isym == 0
        assert time_reversal is True


class TestSymmetryToCartesian:
    """Tests for symmetry_to_cartesian function."""

    def test_cubic_lattice(self):
        """Cubic lattice: S_cart = S for orthogonal lattice."""
        symrel = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ], dtype=int)
        rprimd = np.eye(3)
        gprimd = 2 * np.pi * np.eye(3)
        
        symcart = symmetry_to_cartesian(symrel, rprimd, gprimd)
        expected = symrel.astype(float)
        np.testing.assert_allclose(symcart, expected, atol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
