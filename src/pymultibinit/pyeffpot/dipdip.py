"""
General dipole-dipole interaction utilities.

These routines are independent of any specific parser or storage format.
They operate on Cartesian positions, dielectric tensor, and Born charges.

For supercells, we compute dipole-dipole interactions at Gamma point using
Ewald summation to handle the long-range Coulomb interactions.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Optional


def dipole_dipole_tensor(rij_cart: np.ndarray, epsilon_inf: np.ndarray) -> np.ndarray:
    """Return a simple screened dipole-dipole interaction tensor.

    Parameters
    ----------
    rij_cart
        Cartesian separation vector in Bohr.
    epsilon_inf
        High-frequency dielectric tensor, shape (3, 3).
    
    Returns
    -------
    np.ndarray
        Dipole-dipole interaction tensor (3, 3).
    """
    rij_cart = np.asarray(rij_cart, dtype=float)
    epsilon_inf = np.asarray(epsilon_inf, dtype=float)

    rnorm = np.linalg.norm(rij_cart)
    if rnorm < 1e-14:
        return np.zeros((3, 3), dtype=float)

    inv_eps = np.linalg.inv(epsilon_inf)
    rhat = rij_cart / rnorm
    prefactor = 1.0 / (rnorm ** 3)
    return prefactor * (3.0 * np.outer(rhat, rhat) - np.eye(3)) @ inv_eps


def dipole_dipole_ifc_block(
    rij_cart: np.ndarray,
    zeff_i: np.ndarray,
    zeff_j: np.ndarray,
    epsilon_inf: np.ndarray,
) -> np.ndarray:
    """Build the 3x3 dipole-dipole IFC block between two atoms."""
    tensor = dipole_dipole_tensor(rij_cart, epsilon_inf)
    return zeff_i.T @ tensor @ zeff_j


def build_dipole_dipole_ifcs_simple(
    positions_cart: np.ndarray,
    epsilon_inf: np.ndarray,
    zeff: np.ndarray,
) -> np.ndarray:
    """Build pairwise dipole-dipole IFC blocks for one cell (simple real-space only).

    Returns
    -------
    np.ndarray
        Array with shape (3, natom, 3, natom).
    """
    positions_cart = np.asarray(positions_cart, dtype=float)
    natom = positions_cart.shape[0]
    out = np.zeros((3, natom, 3, natom), dtype=float)

    for i in range(natom):
        for j in range(natom):
            if i == j:
                continue
            block = dipole_dipole_ifc_block(
                positions_cart[j] - positions_cart[i],
                zeff[:, :, i],
                zeff[:, :, j],
                epsilon_inf,
            )
            out[:, i, :, j] = block

    return out


def ewald_dipole_dipole_gamma(
    positions_cart: np.ndarray,
    lattice_vectors: np.ndarray,
    epsilon_inf: np.ndarray,
    zeff: np.ndarray,
    eta: Optional[float] = None,
    nreal: int = 3,
    nrecip: int = 10,
) -> np.ndarray:
    """
    Compute dipole-dipole IFCs at Gamma using Ewald summation.
    
    This is the proper way to handle long-range dipole-dipole interactions
    in a periodic supercell.
    
    Parameters
    ----------
    positions_cart
        Atomic positions in Cartesian coordinates (Bohr), shape (natom, 3).
    lattice_vectors
        Lattice vectors in Bohr, shape (3, 3).
    epsilon_inf
        High-frequency dielectric tensor, shape (3, 3).
    zeff
        Born effective charges, shape (3, 3, natom).
    eta
        Ewald screening parameter (1/Bohr). Controls real/recip balance.
        If None, automatically chosen based on cell size.
    nreal
        Number of real-space lattice vector shells.
    nrecip
        Number of reciprocal-space G-vector shells.
    
    Returns
    -------
    np.ndarray
        Dipole-dipole IFCs, shape (3, natom, 3, natom).
    
    Notes
    -----
    The Ewald method splits the slowly-converging Coulomb sum into:
    - Real-space part: screened by complementary error function
    - Reciprocal-space part: Fourier transform of screened potential
    
    At Gamma point (q=0), the reciprocal part includes the non-analytic
    contribution that gives rise to LO-TO splitting.
    """
    positions_cart = np.asarray(positions_cart, dtype=float)
    lattice_vectors = np.asarray(lattice_vectors, dtype=float)
    epsilon_inf = np.asarray(epsilon_inf, dtype=float)
    zeff = np.asarray(zeff, dtype=float)
    
    natom = positions_cart.shape[0]
    ifcs = np.zeros((3, natom, 3, natom), dtype=float)
    
    # Volume
    volume = np.abs(np.linalg.det(lattice_vectors))
    
    # Auto-select eta if not provided
    if eta is None:
        # Typical choice: eta ~ (N/V)^(1/3) where N is number of cells
        # For a supercell, use moderate screening
        eta_val = 1.0 / np.power(volume, 1.0/3.0) * 2.0
    else:
        eta_val = float(eta)
    
    # Reciprocal lattice vectors
    rec_lattice = 2 * np.pi * np.linalg.inv(lattice_vectors).T
    
    # Build real-space lattice vectors
    real_R = _build_lattice_shells(lattice_vectors, nreal)
    
    # Build reciprocal lattice vectors (exclude Gamma)
    recip_G = _build_lattice_shells(rec_lattice, nrecip)
    recip_G = recip_G[np.linalg.norm(recip_G, axis=1) > 1e-10]
    
    inv_eps = np.linalg.inv(epsilon_inf)
    
    # Try to import scipy for erfc, fall back to simple implementation
    try:
        from scipy.special import erfc
    except ImportError:
        erfc = _erfc_approx
    
    # Real-space contribution
    for i in range(natom):
        for j in range(natom):
            for R in real_R:
                rij = positions_cart[j] - positions_cart[i] + R
                rnorm = np.linalg.norm(rij)
                
                if rnorm < 1e-10:
                    continue
                
                # Screened dipole-dipole tensor
                rhat = rij / rnorm
                
                # Real-space Ewald factor: erfc(eta*r) / r^3 + screening terms
                erfc_eta_r = float(erfc(eta_val * rnorm))
                exp_eta_r2 = np.exp(-(eta_val * rnorm)**2)
                
                # Full tensor: (3*rhat*rhat - I) * [erfc(eta*r)/r^3 + 2*eta/sqrt(pi)*exp(-(eta*r)^2)/r^2]
                factor1 = erfc_eta_r / rnorm**3
                factor2 = 2.0 * eta_val / np.sqrt(np.pi) * exp_eta_r2 / rnorm**2
                factor = factor1 + factor2
                
                dd_tensor = factor * (3.0 * np.outer(rhat, rhat) - np.eye(3))
                
                # Add to IFCs
                block = zeff[:, :, i].T @ dd_tensor @ zeff[:, :, j]
                ifcs[:, i, :, j] += block
    
    # Reciprocal-space contribution (at Gamma, this is the non-analytic part)
    for G in recip_G:
        Gnorm = np.linalg.norm(G)
        if Gnorm < 1e-10:
            continue
            
        # Reciprocal-space factor: 4*pi/V * exp(-G^2/(4*eta^2)) / G^2
        factor = 4.0 * np.pi / volume * np.exp(-Gnorm**2 / (4.0 * eta_val**2)) / Gnorm**2
        
        # Tensor: G*G^T screened by dielectric
        G_eps = G @ inv_eps
        GG = np.outer(G, G_eps)
        
        # Add contribution
        for i in range(natom):
            for j in range(natom):
                # For Gamma point, phase = 1 for all pairs
                block = factor * zeff[:, :, i].T @ GG @ zeff[:, :, j]
                ifcs[:, i, :, j] += block
    
    return ifcs


def _build_lattice_shells(lattice_vectors: np.ndarray, nshells: int) -> np.ndarray:
    """Build lattice vectors up to nshells."""
    vectors = []
    for i in range(-nshells, nshells + 1):
        for j in range(-nshells, nshells + 1):
            for k in range(-nshells, nshells + 1):
                R = i * lattice_vectors[0] + j * lattice_vectors[1] + k * lattice_vectors[2]
                vectors.append(R)
    return np.array(vectors)


def _erfc_approx(x: float) -> float:
    """Approximate complementary error function (if scipy not available)."""
    # Simple approximation for moderate x
    if x < 0:
        return 2.0 - _erfc_approx(-x)
    if x > 6:
        return 0.0
    
    # Abramowitz and Stegun approximation
    t = 1.0 / (1.0 + 0.3275911 * x)
    return (
        0.254829592 * t
        - 0.284496736 * t**2
        + 1.421413741 * t**3
        - 1.453152027 * t**4
        + 1.061405429 * t**5
    ) * np.exp(-x * x)


def ewald_dipole_dipole_for_rpoint(
    positions1_cart: np.ndarray,
    positions2_cart: np.ndarray,
    epsilon_inf: np.ndarray,
    zeff1: np.ndarray,
    zeff2: np.ndarray,
    lattice_vectors: np.ndarray,
    volume: float,
    eta: Optional[float] = None,
) -> np.ndarray:
    """
    Compute dipole-dipole IFCs for a specific R-point using Ewald summation.
    
    This computes the contribution to the interatomic force constants from
    atom pairs where atom i is in the reference cell and atom j is in the
    cell shifted by R.
    
    For R=[0,0,0]: computes the on-site and same-cell contributions
    For R≠[0,0,0]: computes the contribution from that specific R-vector
    
    Parameters
    ----------
    positions1_cart
        Positions of atoms in cell 1 (Cartesian, Bohr), shape (natom1, 3).
    positions2_cart
        Positions of atoms in cell 2 (Cartesian, Bohr), shape (natom2, 3).
        For R≠0, this should be positions1_cart + R_cart.
    epsilon_inf
        High-frequency dielectric tensor, shape (3, 3).
    zeff1
        Born effective charges for cell 1, shape (3, 3, natom1).
    zeff2
        Born effective charges for cell 2, shape (3, 3, natom2).
    lattice_vectors
        Supercell lattice vectors in Bohr, shape (3, 3).
    volume
        Supercell volume in Bohr^3.
    eta
        Ewald screening parameter. If None, auto-selected.
    
    Returns
    -------
    np.ndarray
        Dipole-dipole IFCs, shape (3, natom1, 3, natom2).
    """
    positions1_cart = np.asarray(positions1_cart, dtype=float)
    positions2_cart = np.asarray(positions2_cart, dtype=float)
    epsilon_inf = np.asarray(epsilon_inf, dtype=float)
    zeff1 = np.asarray(zeff1, dtype=float)
    zeff2 = np.asarray(zeff2, dtype=float)
    
    natom1 = positions1_cart.shape[0]
    natom2 = positions2_cart.shape[0]
    dyddt = np.zeros((3, natom1, 3, natom2), dtype=float)
    
    same_cell = np.allclose(positions1_cart, positions2_cart)
    
    if eta is None:
        gmet = np.linalg.inv(lattice_vectors) @ np.linalg.inv(lattice_vectors).T
        rmet = lattice_vectors @ lattice_vectors.T
        direct = np.sum(rmet)
        recip = np.sum(gmet)
        eta_val = np.pi * 100.0 / 33.0 * np.sqrt(1.69 * recip / direct)
    else:
        eta_val = float(eta)
    
    inv_eps = np.linalg.inv(epsilon_inf)
    det_eps = np.linalg.det(epsilon_inf)
    inv_det_eps = 1.0 / np.sqrt(det_eps)
    
    reta = np.sqrt(eta_val)
    reta3 = -eta_val * reta
    fact2 = 2.0 / np.sqrt(np.pi)
    fac = 4.0 / 3.0 / np.sqrt(np.pi)
    
    try:
        from scipy.special import erfc
    except ImportError:
        erfc = _erfc_approx
    
    # Real-space contribution: only the specific R-point
    for i in range(natom1):
        for j in range(natom2):
            rij = positions2_cart[j] - positions1_cart[i]
            rr = reta * rij  # reta = sqrt(eta)
            xx = inv_eps @ rr
            
            y2 = np.dot(rr, xx)
            
            if same_cell and i == j:
                for mu in range(3):
                    for nu in range(3):
                        dyddt[mu, i, nu, j] += fac * reta3 * inv_eps[nu, mu] * inv_det_eps
                continue
            
            if y2 < 1e-24:
                continue
            
            yy = np.sqrt(y2)
            invy = 1.0 / yy
            invy2 = invy * invy
            derfc_yy = erfc(yy)
            
            term2 = derfc_yy * invy * invy2
            term3 = fact2 * np.exp(-y2) * invy2
            term4 = -(term2 + term3)
            term5 = (3.0 * term2 + term3 * (3.0 + 2.0 * y2)) * invy2
            
            for mu in range(3):
                for nu in range(3):
                    dyddt[mu, i, nu, j] += xx[nu] * xx[mu] * term5 + term4 * inv_eps[nu, mu]
    
    # Reciprocal-space contribution (ABINIT style)
    # gsq = G.T @ epsilon @ G (NOT inv_eps!)
    rec_lattice = 2 * np.pi * np.linalg.inv(lattice_vectors).T
    
    for ig1 in range(-10, 11):
        for ig2 in range(-10, 11):
            for ig3 in range(-10, 11):
                G = ig1 * rec_lattice[0] + ig2 * rec_lattice[1] + ig3 * rec_lattice[2]
                
                # gsq = G.T @ epsilon @ G (ABINIT formula)
                gsq = G @ epsilon_inf @ G
                
                if gsq < 1e-10:
                    continue
                
                # arg1 = (2*pi)^2 * gsq / (4*eta)
                arg1 = (2.0 * np.pi)**2 * gsq / (4.0 * eta_val)
                
                # Skip if exp(-arg1) is too small
                if arg1 > 20:
                    continue
                
                factor = np.exp(-arg1) / gsq
                
                for i in range(natom1):
                    for j in range(natom2):
                        G_rij = np.dot(G, positions2_cart[j] - positions1_cart[i])
                        phase = np.cos(G_rij)
                        
                        for mu in range(3):
                            for nu in range(3):
                                dyddt[mu, i, nu, j] += factor * phase * G[mu] * G[nu]
    
    fact1 = 4.0 * np.pi / volume
    dyddt *= fact1
    
    ifcs = np.zeros((3, natom1, 3, natom2), dtype=float)
    for i in range(natom1):
        for j in range(natom2):
            for mu in range(3):
                for nu in range(3):
                    for ii in range(3):
                        for jj in range(3):
                            ifcs[mu, i, nu, j] += zeff1[ii, mu, i] * zeff2[jj, nu, j] * dyddt[ii, i, jj, j]
    
    return ifcs


def ewald_dipole_dipole_two_cells(
    positions1_cart: np.ndarray,
    positions2_cart: np.ndarray,
    epsilon_inf: np.ndarray,
    zeff1: np.ndarray,
    zeff2: np.ndarray,
    lattice_vectors: np.ndarray,
    volume: float,
    eta: Optional[float] = None,
    nreal: int = 3,
    nrecip: int = 10,
) -> np.ndarray:
    """
    Compute dipole-dipole IFCs between two cells using Ewald summation.
    
    This matches ABINIT's ewald9 function for computing dipole-dipole
    between atoms in different cells (for R ≠ 0).
    
    Parameters
    ----------
    positions1_cart
        Positions of atoms in cell 1 (Cartesian, Bohr), shape (natom1, 3).
    positions2_cart
        Positions of atoms in cell 2 (Cartesian, Bohr), shape (natom2, 3).
    epsilon_inf
        High-frequency dielectric tensor, shape (3, 3).
    zeff1
        Born effective charges for cell 1, shape (3, 3, natom1).
    zeff2
        Born effective charges for cell 2, shape (3, 3, natom2).
    lattice_vectors
        Supercell lattice vectors in Bohr, shape (3, 3).
    volume
        Supercell volume in Bohr^3.
    eta
        Ewald screening parameter. If None, auto-selected.
    nreal
        Number of real-space lattice shells.
    nrecip
        Number of reciprocal-space G-vector shells.
    
    Returns
    -------
    np.ndarray
        Dipole-dipole IFCs, shape (3, natom1, 3, natom2).
    """
    positions1_cart = np.asarray(positions1_cart, dtype=float)
    positions2_cart = np.asarray(positions2_cart, dtype=float)
    epsilon_inf = np.asarray(epsilon_inf, dtype=float)
    zeff1 = np.asarray(zeff1, dtype=float)
    zeff2 = np.asarray(zeff2, dtype=float)
    
    natom1 = positions1_cart.shape[0]
    natom2 = positions2_cart.shape[0]
    ifcs = np.zeros((3, natom1, 3, natom2), dtype=float)
    
    if eta is None:
        eta_val = 1.0 / np.power(volume, 1.0/3.0) * 2.0
    else:
        eta_val = float(eta)
    
    rec_lattice = 2 * np.pi * np.linalg.inv(lattice_vectors).T
    
    real_R = _build_lattice_shells(lattice_vectors, nreal)
    recip_G = _build_lattice_shells(rec_lattice, nrecip)
    recip_G = recip_G[np.linalg.norm(recip_G, axis=1) > 1e-10]
    
    inv_eps = np.linalg.inv(epsilon_inf)
    
    try:
        from scipy.special import erfc
    except ImportError:
        erfc = _erfc_approx
    
    for i in range(natom1):
        for j in range(natom2):
            for R in real_R:
                rij = positions2_cart[j] - positions1_cart[i] + R
                rnorm = np.linalg.norm(rij)
                
                if rnorm < 1e-10:
                    continue
                
                rhat = rij / rnorm
                erfc_eta_r = erfc(eta_val * rnorm)
                exp_eta_r2 = np.exp(-(eta_val * rnorm)**2)
                
                factor1 = erfc_eta_r / rnorm**3
                factor2 = 2.0 * eta_val / np.sqrt(np.pi) * exp_eta_r2 / rnorm**2
                factor = factor1 + factor2
                
                dd_tensor = factor * (3.0 * np.outer(rhat, rhat) - np.eye(3))
                
                block = zeff1[:, :, i].T @ dd_tensor @ zeff2[:, :, j]
                ifcs[:, i, :, j] += block
    
    for G in recip_G:
        Gnorm = np.linalg.norm(G)
        if Gnorm < 1e-10:
            continue
        
        factor = 4.0 * np.pi / volume * np.exp(-Gnorm**2 / (4.0 * eta_val**2)) / Gnorm**2
        
        G_eps = G @ inv_eps
        GG = np.outer(G, G_eps)
        
        for i in range(natom1):
            for j in range(natom2):
                block = factor * zeff1[:, :, i].T @ GG @ zeff2[:, :, j]
                ifcs[:, i, :, j] += block
    
    return ifcs


def build_dipole_dipole_ifcs(
    positions_cart: np.ndarray,
    epsilon_inf: np.ndarray,
    zeff: np.ndarray,
    lattice_vectors: Optional[np.ndarray] = None,
    use_ewald: bool = False,
) -> np.ndarray:
    """
    Build dipole-dipole IFCs for a supercell.
    
    Parameters
    ----------
    positions_cart
        Atomic positions in Bohr, shape (natom, 3).
    epsilon_inf
        Dielectric tensor, shape (3, 3).
    zeff
        Born effective charges, shape (3, 3, natom).
    lattice_vectors
        Lattice vectors in Bohr, shape (3, 3). Required for Ewald.
    use_ewald
        If True, use Ewald summation (recommended for supercells).
        If False, use simple real-space only.
    
    Returns
    -------
    np.ndarray
        Dipole-dipole IFCs, shape (3, natom, 3, natom).
    
    Notes
    -----
    For supercells, you should typically use Ewald summation (use_ewald=True)
    to properly handle the long-range nature of dipole-dipole interactions.
    The simple real-space only method is provided for comparison and testing.
    """
    if use_ewald and lattice_vectors is not None:
        return ewald_dipole_dipole_gamma(
            positions_cart, lattice_vectors, epsilon_inf, zeff
        )
    else:
        return build_dipole_dipole_ifcs_simple(
            positions_cart, epsilon_inf, zeff
        )
