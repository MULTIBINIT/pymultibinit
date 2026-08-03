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
from .datastructures import UnitcellData


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
        Array with shape (natom, 3, natom, 3).
    """
    positions_cart = np.asarray(positions_cart, dtype=float)
    natom = positions_cart.shape[0]
    out = np.zeros((natom, 3, natom, 3), dtype=float)

    for i in range(natom):
        for j in range(natom):
            if i == j:
                continue
            block = dipole_dipole_ifc_block(
                positions_cart[j] - positions_cart[i],
                zeff[i, :, :],
                zeff[j, :, :],
                epsilon_inf,
            )
            out[i, :, j, :] = block

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
        Born effective charges, shape (natom, 3, 3).
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
        Dipole-dipole IFCs, shape (natom, 3, natom, 3).

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
    ifcs = np.zeros((natom, 3, natom, 3), dtype=float)

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
    gnorms = np.linalg.norm(recip_G, axis=1)
    recip_G = recip_G[gnorms > 1e-10]
    gnorms = gnorms[gnorms > 1e-10]

    inv_eps = np.linalg.inv(epsilon_inf)

    # Try to import scipy for erfc, fall back to simple implementation
    try:
        from scipy.special import erfc
    except ImportError:
        erfc = _erfc_approx

    # Real-space contribution
    for R in real_R:
        rij_all = positions_cart[None, :, :] - (positions_cart[:, None, :] - R)
        rnorms = np.linalg.norm(rij_all, axis=-1)

        # Mask out zero distances
        mask = rnorms > 1e-10
        if not np.any(mask):
            continue

        rnorms_m = rnorms[mask]
        rij_m = rij_all[mask]
        rhat_m = rij_m / rnorms_m[:, None]

        erfc_val = erfc(eta_val * rnorms_m)
        exp_eta_r2 = np.exp(-(eta_val * rnorms_m)**2)
        factor = erfc_val / rnorms_m**3 + 2.0 * eta_val / np.sqrt(np.pi) * exp_eta_r2 / rnorms_m**2

        # Tensor: factor * (3*rhat*rhat - I)
        # Using a temporary for the tensor calculation
        # Each block is (3,3)
        ident = np.eye(3)
        tensors = factor[:, None, None] * (3.0 * np.einsum('mi,mj->mij', rhat_m, rhat_m) - ident[None, :, :])

        # Map back to ifcs using mask
        idx_i, idx_j = np.where(mask)

        # Vectorized block multiplication:
        # block[i,j] = zeff[i].T @ tensor[m] @ zeff[j]
        # Using einsum: 'imk, mkp, jpr -> imjr'
        blocks = np.einsum('mik, mkp, mpr -> mir', zeff[idx_i].transpose(0, 2, 1), tensors, zeff[idx_j])
        ifcs[idx_i, :, idx_j, :] += blocks

    # Reciprocal-space contribution
    if len(recip_G) > 0:
        inv_eps = np.linalg.inv(epsilon_inf)

        # G_eps = G @ inv_eps
        G_eps = recip_G @ inv_eps
        # GG = np.outer(G, G_eps) for each G
        GGs = np.einsum('gi,gj->gij', recip_G, G_eps)

        # factor = 4 * PI / V * exp(-G^2/(4*eta^2)) / G^2
        factors = (4.0 * np.pi / volume) * np.exp(-gnorms**2 / (4.0 * eta_val**2)) / gnorms**2

        # Sum over all G: factor * GG
        GG_total = np.einsum('g,gij->ij', factors, GGs)

        # Add contribution to all pairs (Gamma point phase is 1)
        # ifcs[i, :, j, :] += zeff[i].T @ GG_total @ zeff[j]
        # einsum: 'imk, kp, jpr -> imjr'
        blocks = np.einsum('imk, kp, jpr -> imjr', zeff.transpose(0, 2, 1), GG_total, zeff)
        ifcs += blocks

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
        Born effective charges for cell 1, shape (natom1, 3, 3).
    zeff2
        Born effective charges for cell 2, shape (natom2, 3, 3).
    lattice_vectors
        Supercell lattice vectors in Bohr, shape (3, 3).
    volume
        Supercell volume in Bohr^3.
    eta
        Ewald screening parameter. If None, auto-selected.

    Returns
    -------
    np.ndarray
        Dipole-dipole IFCs, shape (natom1, 3, natom2, 3).
    """
    positions1_cart = np.asarray(positions1_cart, dtype=float)
    positions2_cart = np.asarray(positions2_cart, dtype=float)
    epsilon_inf = np.asarray(epsilon_inf, dtype=float)
    zeff1 = np.asarray(zeff1, dtype=float)
    zeff2 = np.asarray(zeff2, dtype=float)

    natom1 = positions1_cart.shape[0]
    natom2 = positions2_cart.shape[0]
    dyddt = np.zeros((natom1, 3, natom2, 3), dtype=float)

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

    # Real-space contribution: direct displacement only (R_latt=0)
    rij_all = positions2_cart[None, :, :] - positions1_cart[:, None, :]
    rr_all = reta * rij_all
    xx_all = rr_all @ inv_eps.T

    y2_all = np.einsum('ijk,ijk->ij', rr_all, xx_all)

    if same_cell:
        diag_val = fac * reta3 * inv_eps.T * inv_det_eps
        for i in range(natom1):
            dyddt[i, :, i, :] += diag_val

    mask = y2_all >= 1e-24
    if same_cell:
        np.fill_diagonal(mask, False)

    if np.any(mask):
        y2 = y2_all[mask]
        xx = xx_all[mask]
        yy = np.sqrt(y2)
        invy = 1.0 / yy
        invy2 = invy * invy
        erfc_y = erfc(yy)

        term2 = erfc_y * invy * invy2
        term3 = fact2 * np.exp(-y2) * invy2
        term4 = -(term2 + term3)
        term5 = (3.0 * term2 + term3 * (3.0 + 2.0 * y2)) * invy2

        updates = term5[:, None, None] * np.einsum('mi,mj->mij', xx, xx) + term4[:, None, None] * inv_eps.T[None, :, :]

        idx_i, idx_j = np.where(mask)
        dyddt[idx_i, :, idx_j, :] += updates

    # Reciprocal-space contribution (ABINIT style)
    rec_lattice = 2 * np.pi * np.linalg.inv(lattice_vectors).T

    # Generate G-points grid
    lim = 5
    g1, g2, g3 = np.meshgrid(np.arange(-lim, lim+1), np.arange(-lim, lim+1), np.arange(-lim, lim+1), indexing='ij')
    g_indices = np.stack([g1.flatten(), g2.flatten(), g3.flatten()], axis=1)

    # G = g1*b1 + g2*b2 + g3*b3
    Gs = g_indices @ rec_lattice

    # gsq = G.T @ epsilon @ G
    gsq = np.einsum('gi,ij,gj->g', Gs, epsilon_inf, Gs)

    mask = (gsq > 1e-10)
    # arg1 = (2*pi)^2 * gsq / (4*eta)
    arg1 = gsq / (4.0 * eta_val)
    mask &= (arg1 <= 20)

    if np.any(mask):
        Gs = Gs[mask]
        gsq = gsq[mask]
        arg1 = arg1[mask]

        factors = np.exp(-arg1) / gsq

        # rij = positions2[j] - positions1[i]
        rij = positions2_cart[:, None, :] - positions1_cart[None, :, :] # (natom2, natom1, 3)
        # Note the swap for Broadcasting: G_rij needs to be (natom1, natom2, nG)
        # G_rij = G . rij
        G_rij = np.einsum('gi, kji -> kjg', Gs, rij.transpose(1, 0, 2)) # (natom1, natom2, nG)
        phases = np.cos(G_rij) # (natom1, natom2, nG)

        # dyddt[i, mu, j, nu] += factor * phase * G[mu] * G[nu]
        # Vectorized over G: sum_G factors[G] * phases[i,j,G] * G[mu]*G[nu]
        GGs = np.einsum('gi,gj->gij', Gs, Gs)
        dyddt += np.einsum('g, ijg, gab -> iajb', factors, phases, GGs)

    fact1 = 4.0 * np.pi / volume
    dyddt *= fact1

    # ifcs[i, mu, j, nu] = sum_{alpha, beta} zeff1[i, alpha, mu] * dyddt[i, alpha, j, beta] * zeff2[j, beta, nu]
    ifcs = np.einsum('iam,iajb,jbn->imjn', zeff1, dyddt, zeff2)

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
        Born effective charges for cell 1, shape (natom1, 3, 3).
    zeff2
        Born effective charges for cell 2, shape (natom2, 3, 3).
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
        Dipole-dipole IFCs, shape (natom1, 3, natom2, 3).
    """
    positions1_cart = np.asarray(positions1_cart, dtype=float)
    positions2_cart = np.asarray(positions2_cart, dtype=float)
    epsilon_inf = np.asarray(epsilon_inf, dtype=float)
    zeff1 = np.asarray(zeff1, dtype=float)
    zeff2 = np.asarray(zeff2, dtype=float)

    natom1 = positions1_cart.shape[0]
    natom2 = positions2_cart.shape[0]
    ifcs = np.zeros((natom1, 3, natom2, 3), dtype=float)

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
                erfc_eta_r = erfc(float(eta_val * rnorm))
                exp_eta_r2 = np.exp(-(eta_val * rnorm)**2)

                factor1 = erfc_eta_r / rnorm**3
                factor2 = 2.0 * eta_val / np.sqrt(np.pi) * exp_eta_r2 / rnorm**2
                factor = factor1 + factor2

                dd_tensor = factor * (3.0 * np.outer(rhat, rhat) - np.eye(3))

                block = zeff1[i, :, :].T @ dd_tensor @ zeff2[j, :, :]
                ifcs[i, :, j, :] += block

    for G in recip_G:
        Gnorm = np.linalg.norm(G)
        if Gnorm < 1e-10:
            continue

        factor = 4.0 * np.pi / volume * np.exp(-Gnorm**2 / (4.0 * eta_val**2)) / Gnorm**2

        G_eps = G @ inv_eps
        GG = np.outer(G, G_eps)

        for i in range(natom1):
            for j in range(natom2):
                block = factor * zeff1[i, :, :].T @ GG @ zeff2[j, :, :]
                ifcs[i, :, j, :] += block

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
        Born effective charges, shape (natom, 3, 3).
    lattice_vectors
        Lattice vectors in Bohr, shape (3, 3). Required for Ewald.
    use_ewald
        If True, use Ewald summation (recommended for supercells).
        If False, use simple real-space only.

    Returns
    -------
    np.ndarray
        Dipole-dipole IFCs, shape (natom, 3, natom, 3).

    Notes
    -----
    For supercells, you should typically use Ewald summation (use_ewald=True)
    to properly handle the long-range nature of dipole-dipole interactions.
    The simple real-space only method is provided for comparison and testing.
    """
    zeff = np.asarray(zeff, dtype=float)
    legacy_layout = zeff.ndim == 3 and zeff.shape[0:2] == (3, 3)
    if legacy_layout:
        zeff = np.moveaxis(zeff, 2, 0)

    if use_ewald and lattice_vectors is not None:
        ifcs = ewald_dipole_dipole_gamma(
            positions_cart, lattice_vectors, epsilon_inf, zeff
        )
    else:
        ifcs = build_dipole_dipole_ifcs_simple(
            positions_cart, epsilon_inf, zeff
        )

    if legacy_layout:
        return np.transpose(ifcs, (1, 0, 3, 2))
    return ifcs
def compute_dipdip_dynmat(
    q: np.ndarray,
    unitcell: UnitcellData,
    eta: Optional[float] = None,
    nreal: int = 5,
    nrecip: int = 8,
    sumg0: int = 1,
) -> np.ndarray:
    """
    Compute dipole-dipole dynamical matrix at q using Ewald summation (Convention 1).

    This matches ABINIT's ewald9/dipdip interaction for Fourier interpolation.

    Args:
        q: q-point in reduced coordinates (3,)
        unitcell: UnitcellData
        eta: Ewald screening parameter
        nreal: real-space shells
        nrecip: reciprocal-space shells
        sumg0: if 0, skip G=0 term (for q=0 non-analytic subtraction)
    """
    natom = unitcell.natom
    rprimd = unitcell.rprimd
    xcart = unitcell.xcart
    zeff = unitcell.zeff
    epsilon_inf = unitcell.epsilon_inf

    if zeff is None or epsilon_inf is None:
        return np.zeros((natom, 3, natom, 3), dtype=complex)

    if eta is None:
        rmet = rprimd @ rprimd.T
        gmet_cell = np.linalg.inv(rprimd) @ np.linalg.inv(rprimd).T
        direct = np.sum(rmet)
        recip = np.sum(gmet_cell)
        eta_val = np.pi * 100.0 / 33.0 * np.sqrt(1.69 * recip / direct)
    else:
        eta_val = eta

    vol = np.abs(np.linalg.det(rprimd))

    # Reciprocal lattice
    rec_lattice = 2 * np.pi * np.linalg.inv(rprimd).T
    q_cart = q @ rec_lattice

    dm_dip = np.zeros((natom, 3, natom, 3), dtype=complex)

    # 1. Reciprocal Part
    lim = nrecip
    g1, g2, g3 = np.meshgrid(np.arange(-lim, lim+1), np.arange(-lim, lim+1), np.arange(-lim, lim+1), indexing='ij')
    g_indices = np.stack([g1.flatten(), g2.flatten(), g3.flatten()], axis=1)

    # K = G + q
    Ks_red = g_indices + q
    Ks_cart = Ks_red @ rec_lattice

    # k_eps_k = K . epsilon . K
    k_eps_k = np.einsum('gi,ij,gj->g', Ks_cart, epsilon_inf, Ks_cart)

    mask = k_eps_k > 1e-12
    if sumg0 == 0 and np.linalg.norm(q) < 1e-12:
        g0_idx = np.where(np.all(g_indices == 0, axis=1))[0]
        if len(g0_idx) > 0:
            mask[g0_idx[0]] = False

    Ks = Ks_cart[mask]
    k_eps = k_eps_k[mask]
    G_m = g_indices[mask]

    # Reciprocal sum scaling (ABINIT ewald9 style)
    # factor = 4pi / (vol * sqrt(det_eps))
    det_eps = np.linalg.det(epsilon_inf)
    factor_rec_pre = (4.0 * np.pi / vol)
    factor_rec = factor_rec_pre * np.exp(-k_eps / (4.0 * eta_val)) / k_eps

    diff_tau = xcart[:, np.newaxis, :] - xcart[np.newaxis, :, :]

    phases_rec = np.exp(1j * np.einsum('gi,abi->gab', Ks, diff_tau))

    KKs = np.einsum('gi,gj->gij', Ks, Ks)
    T_rec = np.einsum('g,gab,gij->aibj', factor_rec, phases_rec, KKs)
    dm_dip += np.einsum('iam,iajb,jbn->imjn', zeff, T_rec, zeff)

    # 2. Real-space part
    reta = np.sqrt(eta_val)
    lim_real = nreal
    r1, r2, r3 = np.meshgrid(np.arange(-lim_real, lim_real+1), np.arange(-lim_real, lim_real+1), np.arange(-lim_real, lim_real+1), indexing='ij')
    R_indices = np.stack([r1.flatten(), r2.flatten(), r3.flatten()], axis=1)

    try:
        from scipy.special import erfc
    except ImportError:
        erfc = _erfc_approx

    inv_eps = np.linalg.inv(epsilon_inf)
    det_eps = np.linalg.det(epsilon_inf)
    inv_det_eps = 1.0 / np.sqrt(det_eps)

    # Self-interaction is independent of q
    diag_val = (4.0 / 3.0 / np.pi**0.5) * (reta**3) * inv_det_eps * inv_eps.T

    for R_idx in R_indices:
        R_cart = R_idx @ rprimd

        phase = np.exp(-2j * np.pi * np.dot(q, R_idx))

        rij = R_cart[None, None, :] + diff_tau

        if np.all(R_idx == 0):
            for i in range(natom):
                dm_dip[i, :, i, :] -= np.einsum('mi,mn,np->ip', zeff[i], diag_val, zeff[i])

            mask = np.ones((natom, natom), dtype=bool)
            np.fill_diagonal(mask, False)
        else:
            mask = np.ones((natom, natom), dtype=bool)

        if np.any(mask):
            r_m = rij[mask]
            r_eps_r = np.einsum('mi,ij,mj->m', r_m, inv_eps, r_m)
            r_norm_eps = np.sqrt(r_eps_r)
            y = reta * r_norm_eps

            invy = 1.0 / y
            erfc_y = erfc(y)
            exp_y2 = np.exp(-y**2)
            fact_pi = 2.0 / np.sqrt(np.pi)

            term2 = erfc_y * invy**3
            term3 = fact_pi * exp_y2 * invy**2
            term4 = -(term2 + term3)
            term5 = (3.0 * term2 + term3 * (3.0 + 2.0 * y**2))

            scaled_r = r_m @ inv_eps.T
            dyddt_m = (term5[:, None, None] * np.einsum('mi,mj->mij', scaled_r, scaled_r) / (r_eps_r[:, None, None] + 1e-18)
                       + term4[:, None, None] * inv_eps.T[None, :, :])
            dyddt_m *= -(reta**3) * inv_det_eps

            idx_i, idx_j = np.where(mask)
            for m in range(len(idx_i)):
                ia, ib = idx_i[m], idx_j[m]
                # Each real space shell R has phases exp(i q . R)
                # But rij includes tau_b - tau_a.
                # Abinit convention 1: exp(i q . R)
                block = zeff[ia].T @ dyddt_m[m] @ zeff[ib]
                dm_dip[ia, :, ib, :] += phase * block

    return dm_dip


def _ewald_eta(unitcell: UnitcellData, eta: Optional[float]) -> float:
    if eta is not None:
        return eta

    rprimd = unitcell.rprimd
    rmet = rprimd @ rprimd.T
    gmet_cell = np.linalg.inv(rprimd) @ np.linalg.inv(rprimd).T
    direct = np.sum(rmet)
    recip = np.sum(gmet_cell)
    return float(np.pi * 100.0 / 33.0 * np.sqrt(1.69 * recip / direct))


def _dipdip_g_indices(limit: int) -> np.ndarray:
    g1, g2, g3 = np.meshgrid(
        np.arange(-limit, limit + 1),
        np.arange(-limit, limit + 1),
        np.arange(-limit, limit + 1),
        indexing="ij",
    )
    return np.stack([g1.ravel(), g2.ravel(), g3.ravel()], axis=1)


def _compute_dipdip_reciprocal_part(
    q: np.ndarray,
    unitcell: UnitcellData,
    eta_val: float,
    g_indices: np.ndarray,
    rec_lattice: np.ndarray,
    volume: float,
    sumg0: int,
) -> np.ndarray:
    natom = unitcell.natom
    xcart = unitcell.xcart
    zeff = unitcell.zeff
    epsilon_inf = unitcell.epsilon_inf
    assert zeff is not None and epsilon_inf is not None

    ks_red = g_indices + q
    ks_cart = ks_red @ rec_lattice
    k_eps_k = np.einsum("gi,ij,gj->g", ks_cart, epsilon_inf, ks_cart)

    mask = k_eps_k > 1e-12
    if sumg0 == 0 and np.linalg.norm(q) < 1e-12:
        g0_idx = np.where(np.all(g_indices == 0, axis=1))[0]
        if len(g0_idx) > 0:
            mask[g0_idx[0]] = False

    ks = ks_cart[mask]
    k_eps = k_eps_k[mask]
    factor_rec = (4.0 * np.pi / volume) * np.exp(-k_eps / (4.0 * eta_val)) / k_eps

    diff_tau = xcart[:, np.newaxis, :] - xcart[np.newaxis, :, :]
    phases_rec = np.exp(1j * np.einsum("gi,abi->gab", ks, diff_tau))
    kks = np.einsum("gi,gj->gij", ks, ks)
    t_rec = np.einsum("g,gab,gij->aibj", factor_rec, phases_rec, kks)
    return np.einsum("iam,iajb,jbn->imjn", zeff, t_rec, zeff).reshape(natom, 3, natom, 3)


def _compute_dipdip_reciprocal_parts(
    qpoints: np.ndarray,
    unitcell: UnitcellData,
    eta_val: float,
    g_indices: np.ndarray,
    rec_lattice: np.ndarray,
    volume: float,
    sumg0: int,
) -> np.ndarray:
    """Efficient reciprocal-space dipdip for many q-points.

    Uses the identity D_rec[i,mu,j,nu] = sum_g f(g) ZK[g,i,mu] conj(ZK[g,j,nu])
    where ZK[g,i,mu] = sum_alpha zeff[i,alpha,mu] K[g,alpha] exp(iK·tau_i).
    This avoids creating (nq, nG, natom, natom) phase intermediates.
    """
    natom = unitcell.natom
    xcart = unitcell.xcart
    zeff = unitcell.zeff
    epsilon_inf = unitcell.epsilon_inf
    assert zeff is not None and epsilon_inf is not None

    nq = len(qpoints)
    ng = len(g_indices)

    ks_red = qpoints[:, None, :] + g_indices[None, :, :]          # (nq, nG, 3)
    ks_cart = np.einsum("qgi,ij->qgj", ks_red, rec_lattice)        # (nq, nG, 3)
    k_eps_k = np.einsum("qgi,ij,qgj->qg", ks_cart, epsilon_inf, ks_cart)  # (nq, nG)

    mask = k_eps_k > 1e-12
    if sumg0 == 0:
        gamma_q = np.linalg.norm(qpoints, axis=1) < 1e-12
        g0_idx = np.where(np.all(g_indices == 0, axis=1))[0]
        if len(g0_idx) > 0 and np.any(gamma_q):
            mask[gamma_q, g0_idx[0]] = False

    factor_rec = np.zeros((nq, ng), dtype=float)
    factor_rec[mask] = (
        (4.0 * np.pi / volume)
        * np.exp(-k_eps_k[mask] / (4.0 * eta_val))
        / k_eps_k[mask]
    )
    factor_rec[~mask] = 0.0

    # K · tau for all q, G: (nq, nG, natom)
    k_dot_tau = np.einsum("qgi,ai->qga", ks_cart, xcart)
    phase_i = np.exp(1j * k_dot_tau)                                # (nq, nG, natom)

    # ZK[q, g, i, mu] = sum_alpha zeff[i, alpha, mu] * K[g, alpha] * phase[q, g, i]
    # = einsum('iam,qga->qgim', zeff, phase_i_expanded)
    # But K and phase can be combined: weighted_phase[q,g,i,alpha] = K[q,g,alpha] * phase[q,g,i]
    # Then ZK = einsum('iam,qgia->qgim', zeff, K_phase)
    K_phase = ks_cart[:, :, None, :] * phase_i[:, :, :, None]       # (nq, nG, natom, 3)
    ZK = np.einsum("iam,qgia->qgim", zeff, K_phase)                 # (nq, nG, natom, 3)

    # D_rec[q, i, mu, j, nu] = sum_g factor[q,g] * ZK[q,g,i,mu] * conj(ZK[q,g,j,nu])
    # Equivalent einsum: np.einsum("qgim,qgjn->qimjn", wZK, ZK.conj())
    # which is a contraction over g. Batched GEMM is much faster than einsum:
    # reshape -> (q, natom*3, nG) @ (q, nG, natom*3) -> (q, natom*3, natom*3).
    wZK = ZK * factor_rec[:, :, None, None]                         # (nq, nG, natom, 3)
    wZK2 = wZK.transpose(0, 2, 3, 1).reshape(nq, natom * 3, ng)
    ZKc2 = ZK.conj().reshape(nq, ng, natom * 3)
    result = np.matmul(wZK2, ZKc2).reshape(nq, natom, 3, natom, 3)

    return result


def _precompute_dipdip_real_terms(
    unitcell: UnitcellData,
    eta_val: float,
    nreal: int,
) -> tuple[np.ndarray, np.ndarray]:
    natom = unitcell.natom
    rprimd = unitcell.rprimd
    xcart = unitcell.xcart
    zeff = unitcell.zeff
    epsilon_inf = unitcell.epsilon_inf
    assert zeff is not None and epsilon_inf is not None

    r1, r2, r3 = np.meshgrid(
        np.arange(-nreal, nreal + 1),
        np.arange(-nreal, nreal + 1),
        np.arange(-nreal, nreal + 1),
        indexing="ij",
    )
    r_indices = np.stack([r1.ravel(), r2.ravel(), r3.ravel()], axis=1)
    real_blocks = np.zeros((len(r_indices), natom, 3, natom, 3), dtype=complex)

    try:
        from scipy.special import erfc
    except ImportError:
        erfc = _erfc_approx

    inv_eps = np.linalg.inv(epsilon_inf)
    det_eps = np.linalg.det(epsilon_inf)
    inv_det_eps = 1.0 / np.sqrt(det_eps)
    reta = np.sqrt(eta_val)
    diag_val = (4.0 / 3.0 / np.pi**0.5) * (reta**3) * inv_det_eps * inv_eps.T
    diff_tau = xcart[:, np.newaxis, :] - xcart[np.newaxis, :, :]

    # Vectorized over all (r, i, j). The block formula factorizes:
    #   block[m, b, e] = sum_{a,c,d,f} Z_i[m,a,b] D[m,c,d] Z_j[m,e,f]
    #                   = (sum_a Z_i) * (sum_cd D) * (sum_f Z_j)
    # so only the scalar sum of D over both indices is needed, not the 3x3.
    nR = len(r_indices)
    natom2 = natom * natom
    idx_i, idx_j = np.divmod(np.arange(natom2), natom)

    r_cart = r_indices @ rprimd                                  # (nR, 3)
    rij = r_cart[:, None, None, :] + diff_tau[None]              # (nR, i, j, 3)
    rij2 = rij.reshape(nR, natom2, 3)

    r_eps_r = np.einsum("mpi,ij,mpj->mp", rij2, inv_eps, rij2)   # (nR, N)
    y = reta * np.sqrt(r_eps_r)
    with np.errstate(divide="ignore", invalid="ignore"):
        invy = 1.0 / y
        erfc_y = erfc(y)
        exp_y2 = np.exp(-(y**2))
        fact_pi = 2.0 / np.sqrt(np.pi)
        term2 = erfc_y * invy**3
        term3 = fact_pi * exp_y2 * invy**2
        term4 = -(term2 + term3)
        term5 = 3.0 * term2 + term3 * (3.0 + 2.0 * y**2)

        scaled_r = np.einsum("mpj,ij->mpi", rij2, inv_eps)       # (nR, N, 3)
        dyddt_sum = (
            term5 * (scaled_r.sum(-1) ** 2) / (r_eps_r + 1e-18)
            + term4 * inv_eps.T.sum()
        )
    dyddt_sum *= -(reta**3) * inv_det_eps

    # (sum_a Z_i[m,a,b]) * (sum_f Z_j[m,e,f]) broadcast over r via dyddt_sum
    zsum_i = zeff[idx_i].sum(axis=1)                             # (N, 3)
    zsum_j = zeff[idx_j].sum(axis=2)                             # (N, 3)
    blocks = zsum_i[:, :, None] * zsum_j[:, None, :]             # (N, 3, 3)
    blocks_all = blocks[None] * dyddt_sum[:, :, None, None]      # (nR, N, 3, 3)

    blk2 = blocks_all.reshape(-1, 3, 3)
    r_idx2 = np.repeat(np.arange(nR), natom2)
    i_idx2 = np.tile(idx_i, nR)
    j_idx2 = np.tile(idx_j, nR)
    zero_pos = np.where(np.all(r_indices == 0, axis=1))[0][0]
    keep = (r_idx2 != zero_pos) | (i_idx2 != j_idx2)             # r=0 self-pairs excluded
    np.add.at(
        real_blocks,
        (r_idx2[keep], i_idx2[keep], slice(None), j_idx2[keep], slice(None)),
        blk2[keep],
    )

    # r=0 self-term: subtract zeff[i].T @ diag_val @ zeff[i] on the diagonal
    self_term = np.einsum("ami,mn,anp->aip", zeff, diag_val, zeff)
    real_blocks[zero_pos, np.arange(natom), :, np.arange(natom), :] -= self_term

    return r_indices, real_blocks


_jax_recip_jit = None


def _get_jax_recip():
    """Return a module-level cached JAX JIT of the reciprocal part, or None."""
    global _jax_recip_jit
    if _jax_recip_jit is not None:
        return _jax_recip_jit
    try:
        import jax
        import jax.numpy as jnp

        jax.config.update("jax_enable_x64", True)

        @jax.jit(static_argnums=(8,))
        def _jax_recip(qpoints, g_indices, rec_lattice, epsilon_inf, volume,
                       eta_val, xcart, zeff, sumg0):
            ks_red = qpoints[:, None, :] + g_indices[None, :, :]
            ks_cart = jnp.einsum("qgi,ij->qgj", ks_red, rec_lattice)
            k_eps_k = jnp.einsum("qgi,ij,qgj->qg", ks_cart, epsilon_inf, ks_cart)
            factor = jnp.where(
                k_eps_k > 1e-12,
                (4.0 * jnp.pi / volume) * jnp.exp(-k_eps_k / (4.0 * eta_val)) / k_eps_k,
                0.0,
            )
            if sumg0 == 0:
                gamma_q = jnp.linalg.norm(qpoints, axis=1) < 1e-12
                g0_mask = jnp.all(g_indices == 0, axis=1)
                factor = jnp.where(gamma_q[:, None] & g0_mask[None, :], 0.0, factor)
            k_dot_tau = jnp.einsum("qgi,ai->qga", ks_cart, xcart)
            phase_i = jnp.exp(1j * k_dot_tau)
            K_phase = ks_cart[:, :, None, :] * phase_i[:, :, :, None]
            ZK = jnp.einsum("iam,qgia->qgim", zeff, K_phase)
            wZK = ZK * factor[:, :, None, None]
            return jnp.einsum("qgim,qgjn->qimjn", wZK, ZK.conj())

        _jax_recip_jit = _jax_recip
        return _jax_recip_jit
    except (ImportError, Exception):
        return None


def compute_dipdip_dynmats(
    qpoints: np.ndarray,
    unitcell: UnitcellData,
    eta: Optional[float] = None,
    nreal: int = 5,
    nrecip: int = 8,
    sumg0: int = 1,
) -> np.ndarray:
    """Compute Ewald dipole-dipole dynamical matrices for many q-points.

    Uses a module-level cached JAX JIT for the reciprocal part when available
    (GPU/multicore), falling back to NumPy. Real-space blocks are q-independent
    and precomputed once.
    """
    qpoints = np.asarray(qpoints, dtype=float)
    natom = unitcell.natom
    if unitcell.zeff is None or unitcell.epsilon_inf is None:
        return np.zeros((len(qpoints), natom, 3, natom, 3), dtype=complex)

    eta_val = _ewald_eta(unitcell, eta)
    rprimd = unitcell.rprimd
    volume = np.abs(np.linalg.det(rprimd))
    rec_lattice = 2 * np.pi * np.linalg.inv(rprimd).T
    g_indices = _dipdip_g_indices(nrecip)
    r_indices, real_blocks = _precompute_dipdip_real_terms(unitcell, eta_val, nreal)

    out = None
    jax_recip = _get_jax_recip()
    if jax_recip is not None:
        try:
            import jax.numpy as jnp

            out = np.array(
                jax_recip(
                    jnp.asarray(qpoints), jnp.asarray(g_indices),
                    jnp.asarray(rec_lattice), jnp.asarray(unitcell.epsilon_inf),
                    volume, eta_val, jnp.asarray(unitcell.xcart),
                    jnp.asarray(unitcell.zeff), sumg0,
                )
            )
        except Exception:
            out = None

    if out is None:
        out = _compute_dipdip_reciprocal_parts(
            qpoints, unitcell, eta_val, g_indices, rec_lattice, volume, sumg0
        )

    if len(qpoints) > 0:
        # Per-q einsum: out[iq] += einsum("r,raibj->aibj", phases, real_blocks).
        # All q share the same real_blocks, so vectorize over q with one GEMM:
        # phases_all (nq, nR) @ real_blocks (nR, natom*3*natom*3) -> (nq, ...).
        phases_all = np.exp(-2j * np.pi * (r_indices @ qpoints.T)).T  # (nq, nR)
        out += (phases_all @ real_blocks.reshape(len(r_indices), -1)).reshape(
            len(qpoints), natom, 3, natom, 3
        )

    return out
