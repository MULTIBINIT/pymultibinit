"""
Phonon utilities for calculating frequencies from DDB data.

This module provides functions for:
- Converting dynamical matrices from reduced to Cartesian coordinates
- Calculating phonon frequencies at arbitrary q-points
- Mass weighting and diagonalization

The coordinate transformation is crucial because:
- DDB stores 2nd derivatives in REDUCED coordinates (u)
- Frequency calculation requires CARTESIAN coordinates (x = u × acell)
- Transformation: D_cart = (acell²/2) × D_reduced

Example:
    >>> from pymultibinit.pyeffpot import read_ddb
    >>> from pymultibinit.pyeffpot.phonon import calculate_phonon_frequencies
    >>> 
    >>> u = read_ddb('BaTiO3.DDB')
    >>> gamma_idx = 3  # Gamma point
    >>> dynmat = u.dynmat[gamma_idx, :, :, :, :, 0]  # Real part
    >>> 
    >>> frequencies = calculate_phonon_frequencies(
    ...     dynmat, u.amu, u.typat, u.acell
    ... )
    >>> print(f"Phonon frequencies: {frequencies} cm⁻¹")
"""
import numpy as np
from typing import Optional

# Physical constants (atomic units)
AMU_EMASS = 1822.888484264545  # 1 amu in electron masses
HA_CMM1 = 219474.6313705  # Hartree to cm⁻¹ conversion


def reduced_to_cartesian(dynmat_reduced: np.ndarray, acell: float) -> np.ndarray:
    """
    Convert dynamical matrix from reduced to Cartesian coordinates.
    
    The DDB stores 2nd derivatives in reduced coordinates (fractional).
    Frequency calculation requires Cartesian coordinates (Bohr).
    
    Transformation:
        D_cartesian = (acell²/2) × D_reduced
    
    The factor of 2 comes from ABINIT's specific definition of 2nd derivatives.
    
    Args:
        dynmat_reduced: Dynamical matrix in reduced coordinates
                       Shape: (natom, 3, natom, 3) or (nqpt, natom, 3, natom, 3, 2)
        acell: Lattice parameter in Bohr (for cubic cells)
               For non-cubic cells, this should be handled differently
    
    Returns:
        Dynamical matrix in Cartesian coordinates (same shape as input)
    
    Example:
        >>> dynmat_cart = reduced_to_cartesian(dynmat_ddb, acell=0.189)
    """
    coord_factor = acell**2 / 2.0
    return dynmat_reduced * coord_factor


def mass_weight_dynamical_matrix(
    dynmat: np.ndarray,
    amu: np.ndarray,
    typat: np.ndarray,
    amu_emass: float = AMU_EMASS
) -> np.ndarray:
    """
    Apply mass weighting to dynamical matrix.
    
    The mass-weighted dynamical matrix is:
        D_mass = M^(-1/2) × D × M^(-1/2)
    
    where M is the diagonal mass matrix.
    
    Args:
        dynmat: Dynamical matrix, shape (3*natom, 3*natom)
        amu: Atomic masses in amu, shape (ntypat,)
        typat: Atom types (1-indexed), shape (natom,)
        amu_emass: Conversion factor from amu to electron mass
    
    Returns:
        Mass-weighted dynamical matrix, shape (3*natom, 3*natom)
    """
    natom = len(typat)
    assert dynmat.shape == (3*natom, 3*natom)
    
    # Build mass vector (replicate for 3 directions per atom)
    mass_vec = np.array([amu[typ-1] for typ in typat for _ in range(3)])
    
    # Apply mass weighting: D_ij / sqrt(m_i * m_j) / amu_emass
    mass_weighted = np.zeros((3*natom, 3*natom))
    for i in range(3*natom):
        for j in range(3*natom):
            fac = 1.0 / np.sqrt(mass_vec[i] * mass_vec[j]) / amu_emass
            mass_weighted[i, j] = dynmat[i, j] * fac
    
    # Make Hermitian (symmetric for real matrix)
    mass_weighted = 0.5 * (mass_weighted + mass_weighted.T)
    
    return mass_weighted


def eigenvalues_to_frequencies(
    eigenvalues: np.ndarray,
    ha_cmm1: float = HA_CMM1
) -> np.ndarray:
    """
    Convert eigenvalues to phonon frequencies.
    
    Following ABINIT's convention:
    - For positive eigenvalues: ω = +√λ
    - For negative eigenvalues: ω = -√|λ| (imaginary mode)
    - For near-zero eigenvalues: ω = 0
    
    Args:
        eigenvalues: Eigenvalues of mass-weighted dynamical matrix
        ha_cmm1: Conversion factor from Hartree to cm⁻¹
    
    Returns:
        Phonon frequencies in cm⁻¹
    """
    frequencies = np.zeros_like(eigenvalues)
    
    for i, ev in enumerate(eigenvalues):
        if ev >= 1.0e-16:
            frequencies[i] = np.sqrt(ev) * ha_cmm1
        elif ev >= -1.0e-16:
            frequencies[i] = 0.0
        else:
            frequencies[i] = -np.sqrt(-ev) * ha_cmm1
    
    return frequencies


def calculate_phonon_frequencies(
    dynmat_reduced: np.ndarray,
    amu: np.ndarray,
    typat: np.ndarray,
    acell: float,
    return_eigenvalues: bool = False
) -> np.ndarray:
    """
    Calculate phonon frequencies from DDB dynamical matrix.
    
    This is the main function that performs the complete calculation:
    1. Convert from reduced to Cartesian coordinates
    2. Reshape to 2D matrix
    3. Apply mass weighting
    4. Diagonalize
    5. Convert eigenvalues to frequencies
    
    Args:
        dynmat_reduced: Dynamical matrix from DDB in reduced coordinates
                       Shape: (natom, 3, natom, 3)
        amu: Atomic masses in amu, shape (ntypat,)
        typat: Atom types (1-indexed), shape (natom,)
        acell: Lattice parameter in Bohr
        return_eigenvalues: If True, also return eigenvalues
    
    Returns:
        Phonon frequencies in cm⁻¹, shape (3*natom,)
        If return_eigenvalues is True, returns (frequencies, eigenvalues)
    
    Example:
        >>> u = read_ddb('BaTiO3.DDB')
        >>> dynmat = u.dynmat[3, :, :, :, :, 0]  # Gamma point, real part
        >>> freq = calculate_phonon_frequencies(dynmat, u.amu, u.typat, u.acell)
        >>> print(f"Optical modes: {freq[3:]} cm⁻¹")
    """
    natom = dynmat_reduced.shape[0]
    
    # Step 1: Convert coordinates
    dynmat_cart = reduced_to_cartesian(dynmat_reduced, acell)
    
    # Step 2: Reshape to 2D
    dynmat_2d = dynmat_cart.reshape(3*natom, 3*natom)
    
    # Step 3: Mass weighting
    mass_weighted = mass_weight_dynamical_matrix(dynmat_2d, amu, typat)
    
    # Step 4: Diagonalize
    eigenvalues = np.linalg.eigvalsh(mass_weighted)
    
    # Step 5: Convert to frequencies
    frequencies = eigenvalues_to_frequencies(eigenvalues)
    
    if return_eigenvalues:
        return frequencies, eigenvalues
    return frequencies
