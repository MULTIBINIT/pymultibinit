"""
Atom matching utilities for MULTIBINIT structures.

This module provides functions to match atoms between input structures and
the MULTIBINIT internal supercell reference, handling atom ordering differences
and periodic boundary conditions.

Key functions:
- find_atom_mapping_pbc: Find atom correspondence with PBC handling
- is_identity_mapping_no_pbc_shift: Check if mapping is trivial (optimization)
- apply_mapping_to_positions: Reorder positions
- apply_inverse_mapping_to_forces: Map forces back to input order
"""
import numpy as np
from typing import Tuple, Optional
import warnings


def find_atom_mapping_pbc(
    positions_input: np.ndarray,
    positions_ref: np.ndarray,
    lattice: np.ndarray,
    tolerance: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find mapping between input atoms and reference atoms using minimum image convention.
    
    This implements the same algorithm as MULTIBINIT's effective_potential_file_mapHistToRef
    function (m_effective_potential_file.F90:200-350).
    
    Algorithm:
    1. For each reference atom, compute distance to all input atoms
    2. Apply minimum image convention (wrap to [-0.5, 0.5] in fractional coords)
    3. Find closest match by absolute distance
    4. Record mapping and any PBC shifts needed
    
    Parameters
    ----------
    positions_input : np.ndarray, shape (natom, 3)
        Input atomic positions in Cartesian coordinates (Angstrom or Bohr)
    positions_ref : np.ndarray, shape (natom, 3)
        Reference atomic positions in Cartesian coordinates
    lattice : np.ndarray, shape (3, 3)
        Lattice vectors as row vectors (same units as positions)
    tolerance : float
        Maximum allowed distance for a match (same units as positions).
        Raises error if closest match exceeds this.
        
    Returns
    -------
    mapping : np.ndarray, shape (natom,), dtype=int
        mapping[i] = j means reference atom i matches input atom j
        To reorder input to match reference: positions_reordered = positions_input[mapping]
    inverse_mapping : np.ndarray, shape (natom,), dtype=int
        Inverse mapping for reverse operations (forces back to input order)
        inverse_mapping[j] = i means input atom j corresponds to reference atom i
        
    Raises
    ------
    ValueError
        If natom differs between input and reference, or if no valid mapping found
    RuntimeError
        If closest match distance exceeds tolerance
        
    Examples
    --------
    >>> positions_input = np.array([[0.0, 0.0, 0.0], [0.0, 4.0, 0.0]])
    >>> positions_ref = np.array([[0.0, 4.0, 0.0], [0.0, 0.0, 0.0]])  # Reversed order
    >>> lattice = np.array([[4.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 4.0]])
    >>> mapping, inv_mapping = find_atom_mapping_pbc(positions_input, positions_ref, lattice)
    >>> print(mapping)  # [1, 0] - ref[0] matches input[1], ref[1] matches input[0]
    >>> positions_reordered = positions_input[mapping]
    >>> np.allclose(positions_reordered, positions_ref)
    True
    """
    natom_input = positions_input.shape[0]
    natom_ref = positions_ref.shape[0]
    
    if natom_input != natom_ref:
        raise ValueError(
            f"Number of atoms differs: input has {natom_input}, "
            f"reference has {natom_ref}"
        )
    
    natom = natom_ref
    
    # Convert to fractional coordinates
    lattice_inv = np.linalg.inv(lattice)
    xred_input = positions_input @ lattice_inv  # (natom, 3)
    xred_ref = positions_ref @ lattice_inv
    
    # Initialize mapping arrays
    mapping = np.zeros(natom, dtype=np.int32)
    max_distances = np.zeros(natom)
    
    # For each reference atom, find closest input atom
    for ia in range(natom):
        # Compute fractional coordinate differences
        # list_reddist[ib] = xred_input[ib] - xred_ref[ia]
        dr_frac = xred_input - xred_ref[ia]  # (natom, 3)
        
        # Apply minimum image convention: wrap to [-0.5, 0.5]
        # This is the key PBC handling logic from MULTIBINIT
        dr_frac_wrapped = dr_frac - np.round(dr_frac)
        
        # Convert to Cartesian distance
        dr_cart = dr_frac_wrapped @ lattice  # (natom, 3)
        distances = np.linalg.norm(dr_cart, axis=1)  # (natom,)
        
        # Find closest match
        ib_closest = np.argmin(distances)
        min_distance = distances[ib_closest]
        
        # Store mapping
        mapping[ia] = ib_closest
        max_distances[ia] = min_distance
        
    # Check for valid mapping
    if np.max(max_distances) > tolerance:
        worst_idx = np.argmax(max_distances)
        raise RuntimeError(
            f"Atom matching failed: reference atom {worst_idx} has closest "
            f"match at distance {max_distances[worst_idx]:.6f}, which exceeds "
            f"tolerance {tolerance:.6f}. This likely means the structures are "
            f"incompatible or have very different geometries."
        )
    
    # Check for uniqueness (each input atom matched to at most one ref atom)
    unique_matches = np.unique(mapping)
    if len(unique_matches) < natom:
        # Some input atoms were matched multiple times
        warnings.warn(
            f"Non-unique atom matching: {natom - len(unique_matches)} reference "
            f"atoms map to the same input atoms. This may indicate duplicate atoms "
            f"or a degenerate structure.",
            RuntimeWarning
        )
    
    # Compute inverse mapping
    inverse_mapping = np.zeros(natom, dtype=np.int32)
    for ia in range(natom):
        ib = mapping[ia]
        inverse_mapping[ib] = ia
    
    return mapping, inverse_mapping


def apply_mapping_to_positions(
    positions: np.ndarray,
    mapping: np.ndarray
) -> np.ndarray:
    """
    Reorder positions according to mapping.
    
    Parameters
    ----------
    positions : np.ndarray, shape (natom, 3)
        Input positions
    mapping : np.ndarray, shape (natom,)
        Mapping array from find_atom_mapping_pbc
        
    Returns
    -------
    positions_reordered : np.ndarray, shape (natom, 3)
        Reordered positions
    """
    return positions[mapping]


def apply_inverse_mapping_to_forces(
    forces_reordered: np.ndarray,
    inverse_mapping: np.ndarray
) -> np.ndarray:
    """
    Map forces back from reference order to input order.
    
    After evaluation, forces are in the reference (internal) order.
    This function maps them back to the original input order.
    
    Parameters
    ----------
    forces_reordered : np.ndarray, shape (natom, 3)
        Forces in reference (internal) order
    inverse_mapping : np.ndarray, shape (natom,)
        Inverse mapping from find_atom_mapping_pbc
        
    Returns
    -------
    forces_original : np.ndarray, shape (natom, 3)
        Forces in original input order
        
    Examples
    --------
    >>> # Reference order forces
    >>> forces_ref = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    >>> inverse_mapping = np.array([1, 0])  # input[0]->ref[1], input[1]->ref[0]
    >>> forces_input = apply_inverse_mapping_to_forces(forces_ref, inverse_mapping)
    >>> # forces_input[0] = forces_ref[1], forces_input[1] = forces_ref[0]
    """
    natom = forces_reordered.shape[0]
    forces_original = np.zeros_like(forces_reordered)
    
    for ia in range(natom):
        # inverse_mapping[ia] tells us which reference atom corresponds to input atom ia
        i_ref = inverse_mapping[ia]
        forces_original[ia] = forces_reordered[i_ref]
    
    return forces_original


def validate_mapping(
    positions_input: np.ndarray,
    positions_ref: np.ndarray,
    mapping: np.ndarray,
    lattice: np.ndarray,
    tolerance: float = 1e-3
) -> bool:
    """
    Validate that a mapping correctly matches atoms within tolerance.
    
    Parameters
    ----------
    positions_input : np.ndarray, shape (natom, 3)
        Input positions
    positions_ref : np.ndarray, shape (natom, 3)
        Reference positions
    mapping : np.ndarray, shape (natom,)
        Mapping to validate
    lattice : np.ndarray, shape (3, 3)
        Lattice vectors
    tolerance : float
        Maximum allowed distance (same units as positions)
        
    Returns
    -------
    is_valid : bool
        True if mapping is valid within tolerance
    """
    positions_reordered = positions_input[mapping]
    
    # Convert to fractional for PBC-aware comparison
    lattice_inv = np.linalg.inv(lattice)
    xred_reordered = positions_reordered @ lattice_inv
    xred_ref = positions_ref @ lattice_inv
    
    # Compute differences with PBC
    dr_frac = xred_reordered - xred_ref
    dr_frac_wrapped = dr_frac - np.round(dr_frac)
    dr_cart = dr_frac_wrapped @ lattice
    
    distances = np.linalg.norm(dr_cart, axis=1)
    max_distance = np.max(distances)
    
    return max_distance < tolerance


def is_identity_mapping_no_pbc_shift(
    positions_input: np.ndarray,
    positions_ref: np.ndarray,
    mapping: np.ndarray,
    lattice: np.ndarray,
    tolerance: float = 1e-6
) -> bool:
    """
    Check if mapping is identity (no reordering) and no PBC shifts are needed.
    
    This function checks two conditions:
    1. Mapping is identity: mapping[i] == i for all i
    2. No PBC shifts needed: positions match without wrapping
    
    If both conditions are true, we can skip force remapping for efficiency.
    
    Parameters
    ----------
    positions_input : np.ndarray, shape (natom, 3)
        Input positions
    positions_ref : np.ndarray, shape (natom, 3)
        Reference positions
    mapping : np.ndarray, shape (natom,)
        Atom mapping
    lattice : np.ndarray, shape (3, 3)
        Lattice vectors
    tolerance : float
        Distance tolerance for checking if positions match
        
    Returns
    -------
    is_identity : bool
        True if mapping is identity and no PBC shifts needed
    """
    natom = len(mapping)
    
    # Check 1: Is mapping identity?
    if not np.array_equal(mapping, np.arange(natom)):
        return False
    
    # Check 2: Are positions identical without PBC wrapping?
    # Compute direct Cartesian distance
    dr_cart = positions_input - positions_ref
    distances = np.linalg.norm(dr_cart, axis=1)
    
    if np.max(distances) < tolerance:
        # Positions are identical, no PBC shift needed
        return True
    
    # Positions differ - need to check if it's just PBC wrapping
    # If PBC wrapping is needed, we still need to pass shifted positions
    # to MULTIBINIT, so return False
    return False


def get_reference_structure_info(positions_ref: np.ndarray, lattice: np.ndarray) -> str:
    """
    Generate a human-readable summary of the reference structure.
    
    Parameters
    ----------
    positions_ref : np.ndarray, shape (natom, 3)
        Reference positions
    lattice : np.ndarray, shape (3, 3)
        Lattice vectors
        
    Returns
    -------
    info : str
        Multi-line string with structure information
    """
    natom = positions_ref.shape[0]
    lattice_inv = np.linalg.inv(lattice)
    xred_ref = positions_ref @ lattice_inv
    
    info = f"Reference Structure Information:\n"
    info += f"  Number of atoms: {natom}\n"
    info += f"  Lattice vectors (row format):\n"
    for i in range(3):
        info += f"    [{lattice[i, 0]:10.6f} {lattice[i, 1]:10.6f} {lattice[i, 2]:10.6f}]\n"
    info += f"\n  First 5 atoms (Cartesian):\n"
    for i in range(min(5, natom)):
        info += f"    Atom {i:3d}: [{positions_ref[i, 0]:10.6f} {positions_ref[i, 1]:10.6f} {positions_ref[i, 2]:10.6f}]\n"
    info += f"\n  First 5 atoms (Fractional):\n"
    for i in range(min(5, natom)):
        info += f"    Atom {i:3d}: [{xred_ref[i, 0]:10.6f} {xred_ref[i, 1]:10.6f} {xred_ref[i, 2]:10.6f}]\n"
    
    return info
