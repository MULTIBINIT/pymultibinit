"""
Test atom matching functionality.

This module tests the atom ordering and PBC handling features.
"""
import pytest
import numpy as np
from pymultibinit.atom_matching import (
    find_atom_mapping_pbc,
    apply_mapping_to_positions,
    apply_inverse_mapping_to_forces,
    validate_mapping
)


def test_find_atom_mapping_identity():
    """Test that identical structures produce identity mapping."""
    positions = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 2.0, 2.0],
        [0.0, 4.0, 0.0],
        [2.0, 6.0, 2.0]
    ])
    lattice = np.array([
        [4.0, 0.0, 0.0],
        [0.0, 8.0, 0.0],
        [0.0, 0.0, 4.0]
    ])
    
    mapping, inverse_mapping = find_atom_mapping_pbc(positions, positions, lattice)
    
    # Should be identity mapping
    assert np.array_equal(mapping, np.arange(4))
    assert np.array_equal(inverse_mapping, np.arange(4))


def test_find_atom_mapping_reordered():
    """Test atom matching with reordered atoms."""
    # Reference structure
    positions_ref = np.array([
        [0.0, 0.0, 0.0],  # Atom 0
        [2.0, 2.0, 2.0],  # Atom 1
        [0.0, 4.0, 0.0],  # Atom 2
        [2.0, 6.0, 2.0]   # Atom 3
    ])
    
    # Input structure (atoms 2 and 3 swapped)
    positions_input = np.array([
        [0.0, 0.0, 0.0],  # Atom 0 -> ref 0
        [2.0, 2.0, 2.0],  # Atom 1 -> ref 1
        [2.0, 6.0, 2.0],  # Atom 2 -> ref 3
        [0.0, 4.0, 0.0]   # Atom 3 -> ref 2
    ])
    
    lattice = np.array([
        [4.0, 0.0, 0.0],
        [0.0, 8.0, 0.0],
        [0.0, 0.0, 4.0]
    ])
    
    mapping, inverse_mapping = find_atom_mapping_pbc(
        positions_input, positions_ref, lattice
    )
    
    # Verify mapping
    # ref[0] matches input[0], ref[1] matches input[1]
    # ref[2] matches input[3], ref[3] matches input[2]
    expected_mapping = np.array([0, 1, 3, 2])
    assert np.array_equal(mapping, expected_mapping)
    
    # Verify inverse mapping
    # input[0]->ref[0], input[1]->ref[1], input[2]->ref[3], input[3]->ref[2]
    expected_inverse = np.array([0, 1, 3, 2])
    assert np.array_equal(inverse_mapping, expected_inverse)
    
    # Verify reordering works
    positions_reordered = positions_input[mapping]
    np.testing.assert_allclose(positions_reordered, positions_ref, atol=1e-10)


def test_find_atom_mapping_pbc_wrapped():
    """Test atom matching with PBC wrapping."""
    # Reference structure
    positions_ref = np.array([
        [0.5, 0.5, 0.5],  # Atom in center
        [3.5, 3.5, 3.5]
    ])
    
    # Input structure (same atoms but wrapped across PBC)
    positions_input = np.array([
        [0.5, 0.5, 0.5],      # Same position
        [3.5 - 4.0, 3.5, 3.5]  # Wrapped in x direction: -0.5
    ])
    
    lattice = np.array([
        [4.0, 0.0, 0.0],
        [0.0, 4.0, 0.0],
        [0.0, 0.0, 4.0]
    ])
    
    mapping, inverse_mapping = find_atom_mapping_pbc(
        positions_input, positions_ref, lattice, tolerance=0.5
    )
    
    # Should still match correctly due to PBC
    assert mapping[0] == 0
    assert mapping[1] == 1


def test_apply_mapping_to_positions():
    """Test position reordering."""
    positions = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    
    mapping = np.array([2, 0, 1])  # Rotate indices
    
    positions_reordered = apply_mapping_to_positions(positions, mapping)
    
    expected = np.array([
        [0.0, 0.0, 1.0],  # positions[2]
        [1.0, 0.0, 0.0],  # positions[0]
        [0.0, 1.0, 0.0]   # positions[1]
    ])
    
    np.testing.assert_array_equal(positions_reordered, expected)


def test_apply_inverse_mapping_to_forces():
    """Test force mapping back to input order."""
    # Forces in reference order
    forces_ref = np.array([
        [1.0, 0.0, 0.0],  # Force on ref atom 0
        [0.0, 1.0, 0.0],  # Force on ref atom 1
        [0.0, 0.0, 1.0]   # Force on ref atom 2
    ])
    
    # Inverse mapping: input[i] corresponds to ref[inverse_mapping[i]]
    # input[0]->ref[1], input[1]->ref[2], input[2]->ref[0]
    inverse_mapping = np.array([1, 2, 0])
    
    forces_input_order = apply_inverse_mapping_to_forces(forces_ref, inverse_mapping)
    
    # forces_input_order[0] should be forces_ref[1]
    # forces_input_order[1] should be forces_ref[2]
    # forces_input_order[2] should be forces_ref[0]
    expected = np.array([
        [0.0, 1.0, 0.0],  # forces_ref[1]
        [0.0, 0.0, 1.0],  # forces_ref[2]
        [1.0, 0.0, 0.0]   # forces_ref[0]
    ])
    
    np.testing.assert_array_equal(forces_input_order, expected)


def test_round_trip_mapping():
    """Test that forward and inverse mappings are consistent."""
    positions_ref = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
        [3.0, 3.0, 3.0]
    ])
    
    positions_input = np.array([
        [2.0, 2.0, 2.0],  # -> ref[2]
        [0.0, 0.0, 0.0],  # -> ref[0]
        [3.0, 3.0, 3.0],  # -> ref[3]
        [1.0, 1.0, 1.0]   # -> ref[1]
    ])
    
    lattice = np.eye(3) * 5.0
    
    mapping, inverse_mapping = find_atom_mapping_pbc(
        positions_input, positions_ref, lattice
    )
    
    # Forward mapping: reorder input to match reference
    positions_reordered = apply_mapping_to_positions(positions_input, mapping)
    np.testing.assert_allclose(positions_reordered, positions_ref, atol=1e-10)
    
    # Create forces in reference order
    forces_ref = np.random.rand(4, 3)
    
    # Apply inverse mapping to get forces in input order
    forces_input = apply_inverse_mapping_to_forces(forces_ref, inverse_mapping)
    
    # Verify: force on input atom i should match force on corresponding ref atom
    for i_input in range(4):
        i_ref = inverse_mapping[i_input]
        np.testing.assert_array_equal(forces_input[i_input], forces_ref[i_ref])


def test_validate_mapping():
    """Test mapping validation."""
    positions_ref = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0]
    ])
    
    positions_input = np.array([
        [1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0]
    ])
    
    lattice = np.eye(3) * 5.0
    
    # Correct mapping
    correct_mapping = np.array([1, 0])
    assert validate_mapping(positions_input, positions_ref, correct_mapping, lattice)
    
    # Incorrect mapping
    incorrect_mapping = np.array([0, 1])
    assert not validate_mapping(positions_input, positions_ref, incorrect_mapping, lattice, tolerance=0.1)


def test_matching_with_small_displacements():
    """Test that small displacements don't break matching."""
    positions_ref = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0]
    ])
    
    # Slightly displaced
    positions_input = np.array([
        [2.01, 0.01, 0.0],  # Matches ref[1] with small displacement
        [0.01, 0.01, 0.0]   # Matches ref[0] with small displacement
    ])
    
    lattice = np.eye(3) * 5.0
    
    mapping, _ = find_atom_mapping_pbc(
        positions_input, positions_ref, lattice, tolerance=0.1
    )
    
    # Should still match correctly
    assert mapping[0] == 1  # ref[0] matches input[1]
    assert mapping[1] == 0  # ref[1] matches input[0]


def test_matching_failure_on_incompatible_structures():
    """Test that matching fails with clear error for incompatible structures."""
    positions_ref = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0]
    ])
    
    # Completely different structure
    positions_input = np.array([
        [10.0, 10.0, 10.0],
        [11.0, 11.0, 11.0]
    ])
    
    lattice = np.eye(3) * 5.0
    
    with pytest.raises(RuntimeError, match="Atom matching failed"):
        find_atom_mapping_pbc(
            positions_input, positions_ref, lattice, tolerance=0.1
        )


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
