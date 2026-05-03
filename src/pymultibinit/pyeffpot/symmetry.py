"""
Symmetry operations for dynamical matrix rotation.

This module provides symmetry-related functions for:
1. Getting symmetry operations from crystal structure (using spglib)
2. Building atom mapping (indsym equivalent in ABINIT)
3. Rotating dynamical matrices using symmetry operations

References:
- abinit/src/72_response/m_ddb.F90 (symdm9 subroutine, lines 6712-6966)
- abinit/src/44_abitools/m_dynmat.F90 (symdyma, d2sym3)
- abinit/shared/common/src/32_util/m_symtk.F90 (symatm, littlegroup_q)
"""

from typing import Tuple, Optional
import importlib
import numpy as np

try:
    spglib = importlib.import_module("spglib")
    HAS_SPGLIB = True
except ImportError:
    spglib = None
    HAS_SPGLIB = False


def get_symmetry_from_crystal(
    lattice: np.ndarray,
    positions: np.ndarray,
    numbers: np.ndarray,
    symprec: float = 1e-5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get symmetry operations from crystal structure using spglib.
    
    Parameters
    ----------
    lattice : (3, 3) array
        Lattice vectors in Cartesian coordinates (Bohr or Angstrom)
    positions : (natom, 3) array
        Atomic positions in fractional coordinates
    numbers : (natom,) array
        Atomic numbers for each atom
    symprec : float
        Symmetry tolerance
        
    Returns
    -------
    rotations : (nsym, 3, 3) int array
        Rotation matrices in fractional coordinates (symrel in ABINIT)
    translations : (nsym, 3) float array
        Fractional translations (tnons in ABINIT)
        
    Notes
    -----
    In ABINIT notation:
    - symrel: Real space rotation matrices (integer 3x3)
    - symrec: Reciprocal space rotation = inverse transpose of symrel
    - tnons: Non-symmorphic translations (fractional)
    """
    if not HAS_SPGLIB:
        raise ImportError("spglib is required for symmetry analysis. Install with: pip install spglib")
    if spglib is None:
        raise ImportError("spglib is required for symmetry analysis. Install with: pip install spglib")
    
    cell = (lattice, positions, numbers)
    symmetry = spglib.get_symmetry(cell, symprec=symprec)
    
    if symmetry is None:
        raise ValueError("Could not determine symmetry from crystal structure")
    
    rotations = symmetry['rotations']  # (nsym, 3, 3) integer
    translations = symmetry['translations']  # (nsym, 3) float
    
    return rotations, translations


def get_reciprocal_symmetry(symrel: np.ndarray) -> np.ndarray:
    """
    Compute reciprocal space symmetry matrices from real space matrices.
    
    symrec = inverse(symrel)^T
    
    For orthogonal integer matrices, symrec = symrel.
    
    Parameters
    ----------
    symrel : (nsym, 3, 3) or (3, 3) int array
        Real space rotation matrices
        
    Returns
    -------
    symrec : same shape as symrel
        Reciprocal space rotation matrices
    """
    single = symrel.ndim == 2
    if single:
        symrel = symrel[np.newaxis, :, :]
    
    nsym = symrel.shape[0]
    symrec = np.zeros_like(symrel, dtype=float)
    
    for isym in range(nsym):
        # symrec = inverse(symrel)^T
        # For integer matrices with det = ±1, this gives integer result
        symrec[isym] = np.linalg.inv(symrel[isym]).T
    
    # Round to integers if close
    symrec_int = np.round(symrec).astype(int)
    if np.allclose(symrec, symrec_int):
        symrec = symrec_int
    else:
        # Keep as float for non-orthogonal symmetries
        pass
    
    if single:
        symrec = symrec[0]
    
    return symrec


def find_equivalent_atom(
    xred_target: np.ndarray,
    xred_all: np.ndarray,
    tol: float = 1e-6
) -> Tuple[int, np.ndarray]:
    """
    Find atom equivalent to target position within unit cell.
    
    Parameters
    ----------
    xred_target : (3,) array
        Target position in fractional coordinates (may be outside [0,1))
    xred_all : (natom, 3) array
        All atomic positions in fractional coordinates
    tol : float
        Tolerance for position matching
        
    Returns
    -------
    iatom : int
        Index of equivalent atom
    translation : (3,) int array
        Integer translation to bring target to equivalent atom
    """
    natom = len(xred_all)
    
    # Wrap target to [0, 1)
    xred_wrapped = xred_target - np.floor(xred_target)
    
    min_dist = float('inf')
    best_iatom = 0
    best_trans = np.zeros(3, dtype=int)
    
    for iat in range(natom):
        # Difference in fractional coordinates
        diff = xred_target - xred_all[iat]
        
        # Wrapped difference to [-0.5, 0.5)
        diff_wrapped = diff - np.round(diff)
        dist = np.linalg.norm(diff_wrapped)
        
        if dist < min_dist:
            min_dist = dist
            best_iatom = iat
            best_trans = np.round(diff).astype(int)
    
    if min_dist > tol:
        raise ValueError(
            f"Could not find equivalent atom for position {xred_target}. "
            f"Minimum distance: {min_dist}"
        )
    
    return best_iatom, best_trans


def build_atom_mapping(
    xred: np.ndarray,
    rotations: np.ndarray,
    translations: np.ndarray,
    tol: float = 1e-6
) -> np.ndarray:
    """
    Build atom mapping for each symmetry operation (indsym in ABINIT).
    
    For each symmetry isym and atom iatom, indsym stores:
    - indsym[0:3, isym, iatom]: Translation vector (integer lattice translations)
    - indsym[3, isym, iatom]: Atom index that iatom maps to under INVERSE symmetry
    
    Parameters
    ----------
    xred : (natom, 3) array
        Atomic positions in fractional coordinates
    rotations : (nsym, 3, 3) int array
        Rotation matrices (symrel)
    translations : (nsym, 3) float array
        Fractional translations (tnons)
    tol : float
        Tolerance for atom matching
        
    Returns
    -------
    indsym : (4, nsym, natom) int array
        Atom mapping array:
        - indsym[0:3, isym, i] = translation vector
        - indsym[3, isym, i] = equivalent atom index
        
    References
    ----------
    ABINIT m_symtk.F90:2023-2163 (symatm subroutine)
    """
    nsym = len(rotations)
    natom = len(xred)
    indsym = np.zeros((4, nsym, natom), dtype=int)
    
    # Get reciprocal space rotations (symrec = inverse transpose of symrel)
    symrec = get_reciprocal_symmetry(rotations)
    
    for isym in range(nsym):
        S_inv = symrec[isym].T  # ABINIT symatm applies transpose(symrec)
        tau = translations[isym]  # Fractional translation (tnons)
        
        for iat in range(natom):
            # Apply inverse symmetry: x' = S^{-1} @ (x - tau)
            # This gives the position of the atom after applying inverse symmetry
            xred_new = S_inv @ (xred[iat] - tau)
            
            # Find equivalent atom in the unit cell
            jat, trans = find_equivalent_atom(xred_new, xred, tol)
            
            indsym[3, isym, iat] = jat
            indsym[0:3, isym, iat] = trans
    
    return indsym


def symmetry_to_cartesian(
    symrel: np.ndarray,
    rprimd: np.ndarray,
    gprimd: np.ndarray
) -> np.ndarray:
    """
    Transform symmetry matrix from fractional to Cartesian coordinates.
    
    symcart = rprimd @ symrel @ gprimd
    
    This gives the symmetry operation in Cartesian coordinates.
    
    Parameters
    ----------
    symrel : (3, 3) array
        Rotation matrix in fractional coordinates
    rprimd : (3, 3) array
        Real space lattice vectors (Cartesian)
    gprimd : (3, 3) array
        Reciprocal space lattice vectors (Cartesian, 2π included)
        
    Returns
    -------
    symcart : (3, 3) array
        Rotation matrix in Cartesian coordinates
    """
    return rprimd @ symrel @ np.linalg.inv(rprimd)


def rotate_dynamical_matrix(
    dynmat_ibz: np.ndarray,
    q_ibz: np.ndarray,
    symrel: np.ndarray,
    tnons: np.ndarray,
    indsym_i: np.ndarray,
    rprimd: np.ndarray,
    time_reversal: bool = False
) -> np.ndarray:
    """
    Rotate dynamical matrix from q_ibz to q = S @ q_ibz.
    
    This is a wrapper around rotate_dynamical_matrix_full.
    """
    return rotate_dynamical_matrix_full(
        dynmat_ibz, q_ibz, symrel, tnons, indsym_i, rprimd, time_reversal
    )


def rotate_dynamical_matrix_full(
    dynmat_ibz: np.ndarray,
    q_ibz: np.ndarray,
    symrel: np.ndarray,
    tnons: np.ndarray,
    indsym_i: np.ndarray,
    rprimd: np.ndarray,
    time_reversal: bool = False,
    q_target: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Full rotation of dynamical matrix with atom mapping and phase factors.
    
    Implements the complete ABINIT symdm9 algorithm:
    
    D_ij'(q) = phase_i * phase_j * Σ_{αβ} S_{iα} S_{jβ} D_{αβ}(S^{-1}q)
    
    where:
    - phase_i = exp(2πi * q · τ_i) with τ_i from indsym
    - S is the Cartesian rotation matrix
    - α, β run over mapped atoms
    
    Parameters
    ----------
    dynmat_ibz : (natom, 3, natom, 3, 2) array
        Dynamical matrix at q_ibz (IBZ q-point)
    q_ibz : (3,) array
        IBZ q-point in fractional coordinates
    symrel : (3, 3) int array
        Rotation matrix in fractional coordinates
    tnons : (3,) array
        Fractional translation for this symmetry
    indsym_i : (4, natom) array
        Atom mapping for this symmetry:
        - indsym_i[0:3, i] = translation for atom i
        - indsym_i[3, i] = mapped atom index for atom i
    rprimd : (3, 3) array
        Real space lattice vectors (Cartesian)
    time_reversal : bool
        If True, apply time reversal (complex conjugation)
        
    Returns
    -------
    dynmat_rot : (natom, 3, natom, 3, 2) array
        Rotated dynamical matrix at q = S @ q_ibz
        
    References
    ----------
    ABINIT m_ddb.F90:6931-6953
    """
    natom = dynmat_ibz.shape[0]
    
    # Convert to complex
    D_ibz = dynmat_ibz[..., 0] + 1j * dynmat_ibz[..., 1]
    
    # If time reversal, take complex conjugate first
    if time_reversal:
        D_ibz = np.conj(D_ibz)
    
    # S_cart = R @ S_frac @ inv(R) where R has lattice vectors in columns.
    # In this parser, rprimd has vectors in rows, so R = rprimd.T.
    S_cart = rprimd.T @ symrel.astype(float) @ np.linalg.inv(rprimd.T)
    
    # Compute target q-point for phase factors if not provided
    if q_target is None:
        symrec = get_reciprocal_symmetry(symrel)
        q_target = symrec @ q_ibz
        if time_reversal:
            q_target = -q_target
    
    # Compute phase factors for each atom.
    # ABINIT symdm9 uses the source DDB q-vector qq in these phases, not the
    # target q-point after applying the reciprocal-space symmetry. With time
    # reversal, qq is negated before both conjugating the matrix and evaluating
    # the atom-translation phases.
    q_phase = -q_ibz if time_reversal else q_ibz
    phases = np.zeros(natom, dtype=complex)
    for ia in range(natom):
        trans = indsym_i[0:3, ia]
        arg = 2 * np.pi * np.dot(q_phase, trans)
        phases[ia] = np.exp(1j * arg)
    
    # Rotate dynamical matrix
    D_rot = np.zeros((natom, 3, natom, 3), dtype=complex)
    
    for ia in range(natom):
        for ib in range(natom):
            # Mapped atom indices
            ja = indsym_i[3, ia]  # Atom that ia maps to (image)
            jb = indsym_i[3, ib]
            
            # Phase factor term: exp(i 2pi q_target . (trans_ia - trans_ib))
            phase_factor = phases[ia] * np.conj(phases[ib])
            
            # Get the source 3x3 block from IBZ point
            D_src = D_ibz[ja, :, jb, :]
            
            # Apply Cartesian rotation: S @ D @ S^T
            D_rot_block = S_cart @ D_src @ S_cart.T
            
            # Apply phase
            D_rot[ia, :, ib, :] = phase_factor * D_rot_block
    
    # Convert back to [real, imag] format
    dynmat_rot = np.zeros((natom, 3, natom, 3, 2))
    dynmat_rot[..., 0] = np.real(D_rot)
    dynmat_rot[..., 1] = np.imag(D_rot)
    
    return dynmat_rot


def check_q_symmetry(
    q: np.ndarray,
    q_ref: np.ndarray,
    symrel: np.ndarray,
    tol: float = 1e-8
) -> Tuple[bool, bool]:
    """
    Check if q is related to q_ref by symmetry.
    
    Parameters
    ----------
    q : (3,) array
        Target q-point in fractional coordinates
    q_ref : (3,) array
        Reference q-point in fractional coordinates
    symrel : (3, 3) int array
        Rotation matrix
    tol : float
        Tolerance for comparison
        
    Returns
    -------
    is_direct : bool
        True if q = S @ q_ref (mod G)
    is_inverse : bool
        True if q = -S @ q_ref (mod G), i.e., with time reversal
    """
    # Get reciprocal space rotation
    symrec = get_reciprocal_symmetry(symrel)
    
    # Check direct: q = symrec @ q_ref
    q_sym = symrec @ q_ref
    diff = q - q_sym
    diff = diff - np.round(diff)  # Wrap to [-0.5, 0.5)
    is_direct = np.max(np.abs(diff)) < tol
    
    # Check with time reversal: q = -symrec @ q_ref
    q_sym_tr = -q_sym
    diff_tr = q - q_sym_tr
    diff_tr = diff_tr - np.round(diff_tr)
    is_inverse = np.max(np.abs(diff_tr)) < tol
    
    return is_direct, is_inverse


def find_symmetry_for_qpoint(
    q_target: np.ndarray,
    q_ibz: np.ndarray,
    rotations: np.ndarray,
    tol: float = 1e-8
) -> Tuple[int, bool]:
    """
    Find symmetry operation and time-reversal flag that maps q_ibz to q_target.
    
    Parameters
    ----------
    q_target : (3,) array
        Target q-point in fractional coordinates
    q_ibz : (3,) array
        IBZ q-point to map from
    rotations : (nsym, 3, 3) int array
        All symmetry rotations
    tol : float
        Tolerance for comparison
        
    Returns
    -------
    isym : int
        Index of symmetry operation (-1 if not found)
    time_reversal : bool
        Whether time reversal is needed
        
    Raises
    ------
    ValueError
        If no matching symmetry is found
    """
    nsym = len(rotations)
    symrec = get_reciprocal_symmetry(rotations)
    
    for isym in range(nsym):
        # Check direct
        q_sym = symrec[isym] @ q_ibz
        diff = q_target - q_sym
        diff = diff - np.round(diff)
        if np.max(np.abs(diff)) < tol:
            return isym, False
        
        # Check with time reversal
        q_sym_tr = -q_sym
        diff_tr = q_target - q_sym_tr
        diff_tr = diff_tr - np.round(diff_tr)
        if np.max(np.abs(diff_tr)) < tol:
            return isym, True
    
    raise ValueError(
        f"Could not find symmetry mapping q_ibz={q_ibz} to q_target={q_target}"
    )


def expand_zeff_by_symmetry(
    zeff: np.ndarray,
    rotations: np.ndarray,
    indsym: np.ndarray,
    rprimd: np.ndarray
) -> np.ndarray:
    """
    Expand Born effective charges from irreducible to all atoms.
    
    If atom i has no Born charges (all zero or just ionic charge),
    find a symmetry that maps a known atom j to i, and rotate its Born charge.
    
    Parameters
    ----------
    zeff : (natom, 3, 3) array
        Initial Born charges (may be partially filled)
    rotations : (nsym, 3, 3) int array
        Rotation matrices (frac)
    indsym : (4, nsym, natom) array
        Atom mapping (direct mapping: indsym[3, isym, ja] = ja' where S(ja) = ja' + ntrans)
    rprimd : (3, 3) array
        Real space lattice vectors (Cartesian)
        
    Returns
    -------
    zeff_expanded : (natom, 3, 3) array
    """
    natom = zeff.shape[0]
    nsym = rotations.shape[0]
    zeff_out = np.copy(zeff)
    
    # Identify which atoms have Born charges
    # Electronics part of Born charges is usually significantly non-zero.
    def is_filled(z):
        return np.max(np.abs(z)) > 1e-12

    filled = [is_filled(zeff[i, :, :]) for i in range(natom)]
    
    if all(filled):
        return zeff_out
        
    for i in range(natom):
        if filled[i]: 
            continue
        
        # Try to find a symmetry that maps a filled atom j to i
        found = False
        for isym in range(nsym):
            # In ABINIT indsym, indsym[3, isym, ja] = ja' where S(ja) = ja' + ntrans
            # So if we want to fill atom i, we need ja such that S(ja) maps to i.
            for ja in range(natom):
                if filled[ja] and indsym[3, isym, ja] == i:
                    # i = image of ja under S
                    # Z_i = S_cart @ Z_ja @ S_cart.T
                    S_cart = rprimd.T @ rotations[isym].astype(float) @ np.linalg.inv(rprimd.T)
                    zeff_out[i, :, :] = S_cart @ zeff_out[ja, :, :] @ S_cart.T
                    filled[i] = True
                    found = True
                    break
            if found: 
                break
            
    return zeff_out
