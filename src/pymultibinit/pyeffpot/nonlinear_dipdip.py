"""Geometry-dependent periodic point-dipole electrostatics.

The Ewald sum uses the conducting (G=0-excluded) boundary condition.  The
screening parameter is fixed in transformed Bohr coordinates, so it has no
hidden geometry derivative.
"""

from __future__ import annotations

from itertools import product

import numpy as np
from scipy.special import erfc


_EWALD_ETA = 1.0  # Bohr^-1 in x = epsilon_inf^(-1/2) r coordinates.
_EWALD_TOL = 1.0e-13


def _lattice_vectors_within(basis: np.ndarray, radius: float) -> np.ndarray:
    """Enumerate lattice vectors in a sphere, for column-vector bases."""
    inv_basis = np.linalg.inv(basis)
    limits = np.ceil(radius * np.linalg.norm(inv_basis, axis=1)).astype(int)
    ranges = [range(-int(n), int(n) + 1) for n in limits]
    indices = np.asarray(list(product(*ranges)), dtype=int)
    vectors = indices @ basis.T
    return vectors[np.einsum("ij,ij->i", vectors, vectors) <= radius * radius * (1.0 + 1e-12)]


def _evaluate_geometry_dipoles_components(
    positions: np.ndarray,
    lattice: np.ndarray,
    displacements: np.ndarray,
    zeff: np.ndarray,
    epsilon_inf: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate periodic dipole energy, total forces, and affine stress.

    Parameters
    ----------
    positions : (natom, 3) array
        Cartesian positions in Bohr.
    lattice : (3, 3) array
        Lattice vectors in Bohr as columns.
    displacements : (natom, 3) array
        Cartesian displacements in Bohr.  The dipoles are
        ``p_i = zeff_i @ (u_i - mean(u))``.
    zeff : (natom, 3, 3) array
        Born effective charge tensors, fixed in Cartesian coordinates.
    epsilon_inf : (3, 3) array
        Fixed Cartesian high-frequency dielectric tensor.

    Returns
    -------
    energy : float
        Energy in Hartree.
    forces : (natom, 3) array
        Negative total derivatives with respect to both positions and
        displacements, in Hartree/Bohr.
    stress : (3, 3) array
        Symmetric affine derivative divided by cell volume, in Hartree/Bohr^3.
        Positions, lattice vectors, and displacements are transformed together;
        Born tensors and dielectric tensor are held fixed.

    Notes
    -----
    With ``B = epsilon_inf**(-1/2)``, the screened Green function is
    ``det(epsilon_inf)**(-1/2) / |B r|``.  Real-space dipole energy uses the
    negative Hessian of ``erfc(eta |B r|)/|B r|``; the reciprocal term is
    positive.  The standard dipole self term is included and G=0 is omitted.
    """
    positions = np.asarray(positions, dtype=float)
    lattice = np.asarray(lattice, dtype=float)
    displacements = np.asarray(displacements, dtype=float)
    zeff = np.asarray(zeff, dtype=float)
    epsilon_inf = np.asarray(epsilon_inf, dtype=float)

    if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] == 0:
        raise ValueError("positions must have shape (natom, 3) with natom > 0")
    natom = positions.shape[0]
    if lattice.shape != (3, 3):
        raise ValueError("lattice must have shape (3, 3)")
    if displacements.shape != positions.shape:
        raise ValueError("displacements must have the same shape as positions")
    if zeff.shape != (natom, 3, 3):
        raise ValueError(f"zeff must have shape ({natom}, 3, 3)")
    if epsilon_inf.shape != (3, 3):
        raise ValueError("epsilon_inf must have shape (3, 3)")
    if not all(np.all(np.isfinite(a)) for a in (positions, lattice, displacements, zeff, epsilon_inf)):
        raise ValueError("dipole Ewald inputs must be finite")

    volume = float(np.linalg.det(lattice))
    if abs(volume) <= 1.0e-14:
        raise ValueError("lattice must be nonsingular")
    volume = abs(volume)
    if not np.allclose(epsilon_inf, epsilon_inf.T, rtol=1e-10, atol=1e-12):
        raise ValueError("epsilon_inf must be symmetric")
    epsilon_inf = 0.5 * (epsilon_inf + epsilon_inf.T)
    eps_values, eps_vectors = np.linalg.eigh(epsilon_inf)
    if np.any(eps_values <= 0.0):
        raise ValueError("epsilon_inf must be positive definite")
    bmat = (eps_vectors * (1.0 / np.sqrt(eps_values))) @ eps_vectors.T
    binv = (eps_vectors * np.sqrt(eps_values)) @ eps_vectors.T
    det_factor = float(1.0 / np.sqrt(np.prod(eps_values)))

    # Wrap only for summation efficiency; this is a piecewise-constant lattice
    # translation and leaves local position derivatives unchanged.
    fractional = positions @ np.linalg.inv(lattice).T
    fractional -= np.floor(fractional)
    pos = fractional @ lattice.T
    delta = pos[:, None, :] - pos[None, :, :]
    delta_x = delta @ bmat.T
    pair_diameter = float(np.sqrt(np.max(np.einsum("ija,ija->ij", delta_x, delta_x))))

    u0 = displacements - displacements.mean(axis=0, keepdims=True)
    dipoles = np.einsum("iab,ib->ia", zeff, u0)
    qdipoles = dipoles @ bmat.T
    eta = float(_EWALD_ETA)
    if not np.isfinite(eta) or eta <= 0.0:
        raise ValueError("Ewald eta must be positive")
    exponent_cutoff = -np.log(_EWALD_TOL) + 6.0
    real_cutoff = np.sqrt(exponent_cutoff) / eta
    reciprocal_cutoff = 2.0 * eta * np.sqrt(exponent_cutoff)

    energy = 0.0
    forces_geometry = np.zeros_like(positions)
    grad_q = np.zeros_like(qdipoles)
    denergy_dstrain = np.zeros((3, 3), dtype=float)
    identity = np.eye(3)

    # Direct sum: the pair energy is -1/2 q_i.H[erfc(eta*r)/r].q_j.
    direct_basis = bmat @ lattice
    translations = _lattice_vectors_within(direct_basis, real_cutoff + pair_diameter)
    for translation in translations:
        r = delta + translation @ binv.T
        x = r @ bmat.T
        s2 = np.einsum("ija,ija->ij", x, x)
        mask = s2 <= real_cutoff * real_cutoff
        if not np.any(mask):
            continue
        zero = s2 <= 1.0e-24
        allowed_self = zero & (np.arange(natom)[:, None] == np.arange(natom)[None, :])
        if np.any(zero & ~allowed_self):
            raise ValueError("distinct dipoles occupy the same periodic position")
        mask &= ~allowed_self
        if not np.any(mask):
            continue

        safe_s2 = np.where(mask, s2, 1.0)
        s = np.sqrt(safe_s2)
        exp_term = np.exp(-(eta * s) ** 2)
        erfc_term = erfc(eta * s)
        cterm = (2.0 * eta / np.sqrt(np.pi)) * exp_term
        f1 = -(cterm / s + erfc_term / s**2)
        # f1 is d[erfc(eta*s)/s]/ds.
        f2 = 2.0 * eta**2 * cterm + 2.0 * cterm / s**2 + 2.0 * erfc_term / s**3
        f3 = (
            -4.0 * eta**4 * s * cterm
            - 4.0 * eta**2 * cterm / s
            - 6.0 * cterm / s**3
            - 6.0 * erfc_term / s**4
        )
        ufun = f1 / s
        vfun = f2 / s**2 - f1 / s**3
        uder = f2 / s - f1 / s**2
        vder = f3 / s**2 - 3.0 * f2 / s**3 + 3.0 * f1 / s**4

        qi = qdipoles[:, None, :]
        qj = qdipoles[None, :, :]
        qdot = np.einsum("ia,ja->ij", qdipoles, qdipoles)
        qix = np.einsum("ija,ia->ij", x, qdipoles)
        qjx = np.einsum("ija,ja->ij", x, qdipoles)
        hqj = ufun[:, :, None] * qj + (
            vfun * qjx
        )[:, :, None] * x
        pair_scalar = qdot * ufun + qix * qjx * vfun
        pair_grad_x = (
            (qdot * uder / s)[:, :, None] * x
            + vfun[:, :, None] * (qjx[:, :, None] * qi + qix[:, :, None] * qj)
            + (qix * qjx * vder / s)[:, :, None] * x
        )
        pair_grad_r = pair_grad_x @ bmat
        weighted_scalar = np.where(mask, pair_scalar, 0.0)
        weighted_hqj = np.where(mask[:, :, None], hqj, 0.0)
        weighted_grad = np.where(mask[:, :, None], pair_grad_r, 0.0)
        weighted_r = np.where(mask[:, :, None], r, 0.0)

        energy -= 0.5 * det_factor * float(np.sum(weighted_scalar))
        grad_q -= det_factor * np.sum(weighted_hqj, axis=1)
        forces_geometry += det_factor * np.sum(weighted_grad, axis=1)
        denergy_dstrain -= 0.5 * det_factor * np.einsum(
            "ija,ijb->ab", weighted_grad, weighted_r
        )

    # Reciprocal sum in transformed coordinates.  G=0 is excluded.
    reciprocal_basis = 2.0 * np.pi * np.linalg.inv(direct_basis).T
    gx = _lattice_vectors_within(reciprocal_basis, reciprocal_cutoff)
    gx2 = np.einsum("ga,ga->g", gx, gx)
    nonzero = gx2 > 1.0e-24
    gx = gx[nonzero]
    gx2 = gx2[nonzero]
    gcart = gx @ bmat.T
    phase = np.exp(1j * (gcart @ pos.T))
    gdip = gcart @ dipoles.T
    site_terms = gdip * phase
    structure = np.sum(site_terms, axis=1)
    reciprocal_weight = np.exp(-gx2 / (4.0 * eta**2)) / gx2
    reciprocal_prefactor = 2.0 * np.pi / volume
    structure_abs2 = structure.real**2 + structure.imag**2
    energy += reciprocal_prefactor * float(np.dot(reciprocal_weight, structure_abs2))

    structure_site = structure.conj()[:, None] * phase
    gp_reciprocal = 2.0 * reciprocal_prefactor * np.einsum(
        "g,ga,gi->ia",
        reciprocal_weight,
        gcart,
        structure_site.real,
        optimize=True,
    )
    force_reciprocal = 2.0 * reciprocal_prefactor * np.einsum(
        "g,ga,gi->ia",
        reciprocal_weight,
        gcart,
        (structure.conj()[:, None] * site_terms).imag,
        optimize=True,
    )
    grad_q += gp_reciprocal @ binv.T
    forces_geometry += force_reciprocal

    # Explicit reciprocal-lattice affine derivative at fixed dipoles.
    p_phase = phase @ dipoles
    eps_g = gcart @ epsilon_inf.T
    log_weight_derivative = 1.0 / (4.0 * eta**2) + 1.0 / gx2
    denergy_dstrain -= reciprocal_prefactor * float(
        np.dot(reciprocal_weight, structure_abs2)
    ) * identity
    denergy_dstrain += 2.0 * reciprocal_prefactor * np.einsum(
        "g,ga,gb->ab",
        reciprocal_weight * structure_abs2 * log_weight_derivative,
        gcart,
        eps_g,
        optimize=True,
    )
    denergy_dstrain -= 2.0 * reciprocal_prefactor * np.einsum(
        "g,ga,gb->ab",
        reciprocal_weight,
        gcart,
        (structure.conj()[:, None] * p_phase).real,
        optimize=True,
    )

    # The regularized reciprocal self contribution is removed by this term.
    self_coefficient = 2.0 * eta**3 / (3.0 * np.sqrt(np.pi))
    energy -= det_factor * self_coefficient * float(np.sum(qdipoles * qdipoles))
    grad_q -= det_factor * (2.0 * self_coefficient) * qdipoles

    grad_p = grad_q @ bmat
    grad_u = np.einsum("iba,ib->ia", zeff, grad_p)
    grad_u -= grad_u.mean(axis=0, keepdims=True)
    forces = forces_geometry - grad_u

    # Affine displacement chain: du = dF @ u.  The Born tensors are fixed.
    stress_displacement = np.einsum("ia,ib->ab", grad_u, displacements)
    stress_displacement = 0.5 * (
        stress_displacement + stress_displacement.T
    ) / volume
    denergy_dstrain += stress_displacement * volume
    stress = 0.5 * (denergy_dstrain + denergy_dstrain.T) / volume
    return float(energy), forces, stress, -grad_u, stress_displacement
def evaluate_geometry_dipoles(
    positions: np.ndarray,
    lattice: np.ndarray,
    displacements: np.ndarray,
    zeff: np.ndarray,
    epsilon_inf: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return energy, total forces, and affine stress for the dipole model."""
    energy, forces, stress, _, _ = _evaluate_geometry_dipoles_components(
        positions, lattice, displacements, zeff, epsilon_inf
    )
    return energy, forces, stress


def evaluate_geometry_dipoles_components(
    positions: np.ndarray,
    lattice: np.ndarray,
    displacements: np.ndarray,
    zeff: np.ndarray,
    epsilon_inf: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Also return displacement-only forces and their affine stress."""
    return _evaluate_geometry_dipoles_components(
        positions, lattice, displacements, zeff, epsilon_inf
    )
