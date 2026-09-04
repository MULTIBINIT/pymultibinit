"""Analytic second-derivative (Hessian) blocks of the effective potential.

All quantities in the pyeffpot unit family: Bohr / Hartree. Blocks are exact
second partial derivatives of the evaluated energy E(u, eta) with u the
supercell displacement vector (natom_sc, 3) and eta the (3,3) engineering
strain (Voigt order xx,yy,zz,yz,xz,xy; tensor-shear readback).

Channel prefactors (verified against potential.py evaluators):

    harmonic        E = 1/2 u^T Phi u          -> d2E/du2 = Phi (constant)
    elastic         E = 1/2 Nc eta^T C eta     -> d2E/deta2 = Nc C
    internal strain E = 1/2 sum Lambda eta u   -> d2E/deta du = 1/2 Lambda
    phonon-strain   E = 1/6 sum_a eta_a u^T Phi_a u
                                                -> d2E/du2 += (1/3) eta_a Phi_a
                                                -> d2E/deta_a du = (1/3) Phi_a u
    fitted terms    E = c w prod_f (d_f)^(p_f) prod_s eta_(v_s)^(q_s)
        exact product rule over ordered factor pairs (powers are small ints)

Note (energy/force asymmetry): `_evaluate_strain_coupling` implements
forces with prefactor 1/2 while its energy carries 1/6; the exact
derivative of the evaluated energy is 1/3. The blocks below are exact
derivatives OF THE ENERGY; FD validation quantifies the evaluator mismatch.

The clamped-ion (affine) elastic constant is assembled by `elastic_affine`:
under scale_atoms=True homogeneous strain the displacement field transforms
linearly, u(eta) = (I + eta) u(0) (before min-image wrapping; exact for the
FD amplitudes used), hence d2u/deta2 = 0 and

    C0[nu, om] = H_etaeta[nu, om]
               + coupling[nu] . (E^om u) + coupling[om] . (E^nu u)
               + (E^nu u)^T H_uu (E^om u)

with E^nu the engineering Voigt generators (shear tensor entries = 1/2
amplitude), matching atomchain ddb/finite_difference.strain_atoms.

`coupling_fixed_xcart` gives the force response to strain at fixed Cartesian
positions, the object measured by FD of the code's forces at fixed xcart:
    dF/deta_nu = -coupling[nu] - H_uu @ du_delta_nu
with du_delta = `_compute_du_delta` (Fortran Eq. A4).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# Voigt order xx, yy, zz, yz, xz, xy -> symmetric tensor (alpha, beta) pairs
# (same convention as potential._compute_du_delta).
_VOIGT_PAIRS = ((0, 0), (1, 1), (2, 2), (2, 1), (2, 0), (1, 0))


def _engineering_generators() -> np.ndarray:
    """(6, 3, 3) generators E^nu with shear tensor entries = 1/2 amplitude.

    Matches atomchain finite_difference.strain_atoms: deformation I + eps
    built with eps[a, b] += eps_voigt / 2 for shear Voigt components.
    """
    gens = np.zeros((6, 3, 3), dtype=float)
    for nu, (a, b) in enumerate(_VOIGT_PAIRS):
        gens[nu, a, b] += 0.5
        gens[nu, b, a] += 0.5
    return gens


@dataclass
class HessianBlocks:
    """Exact second partials of E(u, eta) at one configuration (Ha/Bohr).

    ifc             (3N, 3N) d2E/du du   (supercell force constants)
    elastic_fixed_u (6, 6)   d2E/deta deta at fixed u (Hartree; the
                              FD-comparable clamped-ion constant is
                              `elastic_affine`, not this)
    coupling        (6, 3N)  d2E/deta du (fixed-u partial derivative)
    forces          (N, 3)   atomic forces F = -dE/du (Ha/Bohr; directly
                              comparable to EffectivePotential.evaluate)
    strain_voigt    (6,)     engineering Voigt strain used
    """

    ifc: np.ndarray
    elastic_fixed_u: np.ndarray
    coupling: np.ndarray
    forces: np.ndarray
    strain_voigt: np.ndarray


def analytic_blocks(potential, u: np.ndarray, eta: np.ndarray,
                    rprimd: Optional[np.ndarray] = None) -> HessianBlocks:
    """Exact analytic second-derivative blocks at (u, eta)."""
    sc = potential.supercell
    natom_sc = sc.natom_sc
    u = np.asarray(u, dtype=float)
    if u.shape != (natom_sc, 3):
        raise ValueError(f"u has shape {u.shape}, expected {(natom_sc, 3)}")
    eta = np.asarray(eta, dtype=float)
    if eta.shape != (3, 3):
        raise ValueError(f"eta has shape {eta.shape}, expected (3, 3)")
    strain_voigt = np.array([
        eta[0, 0], eta[1, 1], eta[2, 2],
        eta[1, 2] + eta[2, 1],
        eta[2, 0] + eta[0, 2],
        eta[0, 1] + eta[1, 0],
    ], dtype=float)
    ifc = np.zeros((3 * natom_sc, 3 * natom_sc), dtype=float)
    elastic = np.zeros((6, 6), dtype=float)
    coupling = np.zeros((6, 3 * natom_sc), dtype=float)
    forces = np.zeros((natom_sc, 3), dtype=float)

    _harmonic_blocks(potential, u, ifc, forces)
    _elastic_blocks(sc, elastic)
    _internal_strain_blocks(sc, strain_voigt, coupling, forces)
    _phonon_strain_blocks(potential, u, strain_voigt, ifc, coupling, forces)
    _fitted_term_blocks(potential, u, strain_voigt, ifc, elastic, coupling, forces)

    # exact symmetry by construction; enforce bit-level against scatter drift
    ifc = 0.5 * (ifc + ifc.T)

    return HessianBlocks(ifc=ifc, elastic_fixed_u=elastic, coupling=coupling,
                         forces=forces, strain_voigt=strain_voigt)


def _harmonic_blocks(potential, u, ifc, forces) -> None:
    """E = 1/2 u^T Phi u -> d2E/du2 = Phi; dE/du = Phi u."""
    phi = potential._phi_matrix
    if phi is None:
        return
    ifc += phi
    forces -= (phi @ u.reshape(-1)).reshape(-1, 3)


def _elastic_blocks(sc, elastic) -> None:
    """E = 1/2 Nc eta^T C eta -> d2E/deta2 = Nc C (no force contribution)."""
    C = getattr(sc.unitcell, "elastic_constants", None)
    if C is None:
        return
    C = np.asarray(C, dtype=float)
    if C.shape == (6, 6):
        elastic += float(sc.ncells) * C


def _internal_strain_blocks(sc, strain_voigt, coupling, forces) -> None:
    """E = 1/2 sum_{alpha,mu,n} Lambda[alpha,mu,n%Nuc] eta_alpha u[n,mu].

    d2E/deta_alpha du[n,mu] = 1/2 Lambda[alpha,mu,n%Nuc]
    dE/du[n,mu]             = 1/2 sum_alpha Lambda[alpha,mu,n%Nuc] eta_alpha
    """
    lam = getattr(sc.unitcell, "strain_coupling", None)
    if lam is None:
        return
    lam = np.asarray(lam, dtype=float)          # (6, 3, natom_uc)
    natom_uc = sc.unitcell.crystal.natom
    natom_sc = sc.natom_sc
    if lam.shape != (6, 3, natom_uc):
        return
    iuc = np.arange(natom_sc) % natom_uc        # supercell builder order:
    # ia_sc = natom_uc * (iz + nz*(iy + ny*ix)) + atom_uc  => iuc = ia_sc % natom_uc
    lam_sc = lam[:, :, iuc]                     # (6, 3, natom_sc)
    coupling += 0.5 * lam_sc.transpose(0, 2, 1).reshape(6, 3 * natom_sc)
    forces -= 0.5 * np.einsum('amn,a->nm', lam_sc, strain_voigt)


def _phonon_strain_blocks(potential, u, strain_voigt, ifc, coupling, forces) -> None:
    """E = 1/6 sum_a eta_a u^T Phi_a u.

    d2E/du2       += (1/3) eta_a Phi_a
    d2E/deta_a du  = (1/3) Phi_a u   (fixed-u partial)
    dE/du          = (1/3) eta_a Phi_a u
    """
    mats = potential._phonon_strain_matrices
    if mats is None:
        return
    u_flat = u.reshape(-1)
    for alpha in range(6):
        mat = mats[alpha]
        if mat is None:
            continue
        ifc += (strain_voigt[alpha] / 3.0) * mat
        g = (1.0 / 3.0) * (mat @ u_flat)
        coupling[alpha] += g
        forces -= (strain_voigt[alpha] * g).reshape(-1, 3)


def _pow_guard(x: np.ndarray, n: int) -> np.ndarray:
    """x**n safe for integer n >= -2 at x == 0 (0**0 = 1, 0**(neg) = 0)."""
    if n >= 0:
        return x ** n
    return np.where(x == 0.0, 0.0, x ** n)


def _fitted_term_blocks(potential, u, strain_voigt, ifc, elastic, coupling,
                        forces) -> None:
    """Exact product-rule Hessian of the fitted polynomial terms.

    Term energy (potential._evaluate_anharmonic):
        E = c * S * sum_o prod_f d_f(o)^(p_f),
        S = prod_s eta_(v_s)^(q_s),
        d_f(o) = u[idx_a[f,o], dir_f] - u[idx_b[f,o], dir_f].

    Derivatives via the product rule over ORDERED factor pairs (so that two
    distinct factors acting on the same flat index accumulate both orderings,
    while a single factor's cross terms are counted once).
    """
    compiled = potential._anharmonic_compiled
    if not compiled:
        return

    for term_info in compiled:
        c = float(term_info['value']) * float(term_info['weight'])
        disps = [d for d in term_info['displacements'] if int(d['power']) != 0]
        strains = [s for s in (term_info.get('strains') or [])
                   if int(s.get('voigt', 0)) > 0 and int(s.get('power', 0)) > 0]
        F = len(disps)
        S = len(strains)
        if F + S == 0:
            continue

        d_vals, idx_a, idx_b, dirs, pows = [], [], [], [], []
        for d in disps:
            ia = np.asarray(d['idx_a'], dtype=int).reshape(-1)
            ib = np.asarray(d['idx_b'], dtype=int).reshape(-1)
            mu = int(d['dir'])
            p = int(d['power'])
            d_vals.append(u[ia, mu] - u[ib, mu])
            idx_a.append(ia)
            idx_b.append(ib)
            dirs.append(mu)
            pows.append(p)
        sv = [int(s['voigt']) - 1 for s in strains]   # 1-based -> 0-based
        sq = [int(s['power']) for s in strains]

        def prod_others(exclude):
            out = np.ones_like(d_vals[0]) if F else np.ones(1)
            for h in range(F):
                if h not in exclude:
                    out = out * d_vals[h] ** pows[h]
            return out

        def s_val(exclude=()):
            val = 1.0
            for t in range(S):
                if t not in exclude:
                    val *= float(_pow_guard(np.array(strain_voigt[sv[t]]), sq[t]))
            return val

        def s_dval(k):
            """d/deta_(v_k) of the strain monomial (per-factor product rule)."""
            qk = sq[k]
            if qk == 0:
                return 0.0
            ek = strain_voigt[sv[k]]
            if ek == 0.0 and qk - 1 < 0:
                return 0.0
            return qk * float(ek ** (qk - 1)) * s_val((k,))

        prod_all = prod_others(()) if F else np.ones(1)
        # pure-strain terms (no displacement factors) still sum over the
        # ncell origins: prod_disp = ones(ncells) in _evaluate_anharmonic.
        sum_prod = float(prod_all.sum()) if F else float(potential.supercell.ncells)

        # ---- d2E/du du : ordered factor pairs ----
        for f in range(F):
            pf = pows[f]
            for g in range(F):
                pg = pows[g]
                if f == g:
                    if pf < 2:
                        continue
                    pref = (c * s_val() * pf * (pf - 1)
                            * _pow_guard(d_vals[f], pf - 2) * prod_others((f,)))
                    _scatter_pair(ifc, idx_a[f], idx_b[f], dirs[f],
                                  idx_a[f], idx_b[f], dirs[f], pref)
                else:
                    pref = (c * s_val() * pf * pg
                            * _pow_guard(d_vals[f], pf - 1)
                            * _pow_guard(d_vals[g], pg - 1)
                            * prod_others((f, g)))
                    _scatter_pair(ifc, idx_a[f], idx_b[f], dirs[f],
                                  idx_a[g], idx_b[g], dirs[g], pref)

        # ---- d2E/deta du ----
        for s_idx in range(S):
            base = c * s_dval(s_idx)
            if base == 0.0:
                continue
            for f in range(F):
                pf = pows[f]
                pref = (base * pf * _pow_guard(d_vals[f], pf - 1)
                        * prod_others((f,)))
                flat_a = 3 * idx_a[f] + dirs[f]
                flat_b = 3 * idx_b[f] + dirs[f]
                np.add.at(coupling[sv[s_idx]], flat_a, pref)
                np.add.at(coupling[sv[s_idx]], flat_b, -pref)

        # ---- d2E/deta deta ----
        for s_i in range(S):
            for s_j in range(S):
                if s_i == s_j:
                    q = sq[s_i]
                    if q < 2:
                        continue
                    ek = strain_voigt[sv[s_i]]
                    if ek == 0.0 and q - 2 < 0:
                        continue
                    val = (c * q * (q - 1) * float(ek ** (q - 2))
                           * s_val((s_i,)) * sum_prod)
                    elastic[sv[s_i], sv[s_i]] += val
                else:
                    di = s_dval(s_i)
                    dj = s_dval(s_j)
                    if di == 0.0 or dj == 0.0:
                        continue
                    # mixed second derivative removes both factors once
                    val = (c * sq[s_i] * sq[s_j]
                           * float(_pow_guard(np.array(strain_voigt[sv[s_i]]), sq[s_i] - 1))
                           * float(_pow_guard(np.array(strain_voigt[sv[s_j]]), sq[s_j] - 1))
                           * s_val((s_i, s_j)) * sum_prod)
                    elastic[sv[s_i], sv[s_j]] += val

        # ---- dE/du (forces): F = -dE/du ----
        sval = s_val()
        for f in range(F):
            pf = pows[f]
            pref = (c * sval * pf * _pow_guard(d_vals[f], pf - 1)
                    * prod_others((f,)))
            flat_a = 3 * idx_a[f] + dirs[f]
            flat_b = 3 * idx_b[f] + dirs[f]
            np.add.at(forces.reshape(-1), flat_a, -pref)
            np.add.at(forces.reshape(-1), flat_b, pref)


def _scatter_pair(ifc, ia1, ib1, mu1, ia2, ib2, mu2, pref) -> None:
    """Scatter pref (ncells,) into the flat Hessian for one ordered pair.

    d = u[a] - u[b] gives second-derivative signs: (+) on (a1,a2) and
    (b1,b2), (-) on (a1,b2) and (b1,a2). Each ordered pair is written once;
    the transposed entry comes from the reversed ordered pair.
    """
    fa = 3 * ia1 + mu1
    fb = 3 * ib1 + mu1
    ga = 3 * ia2 + mu2
    gb = 3 * ib2 + mu2
    np.add.at(ifc, (fa, ga), pref)
    np.add.at(ifc, (fa, gb), -pref)
    np.add.at(ifc, (fb, ga), -pref)
    np.add.at(ifc, (fb, gb), pref)


def elastic_affine(potential, u: np.ndarray, eta: np.ndarray,
                   blocks: Optional[HessianBlocks] = None
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Clamped-ion (affine) elastic constant C0 and the chain-rule correction.

    C0 = elastic_fixed_u + chain with
    chain = coupling.E^omega u |_nu + coupling.E^nu u |_omega
          + (E^nu u)^T ifc (E^omega u)
    (d2u/deta2 = 0 on the affine path; see module docstring).
    Returns (C0 (6,6) Hartree, chain (6,6) Hartree).
    """
    if blocks is None:
        blocks = analytic_blocks(potential, u, eta)
    gens = _engineering_generators()
    du = np.einsum('nij,aj->nai', gens, u).reshape(6, -1)   # (6, 3N), atom-major flat
    cd = blocks.coupling @ du.T                             # (6, 6)
    chain = cd + cd.T + du @ blocks.ifc @ du.T
    chain = 0.5 * (chain + chain.T)
    return blocks.elastic_fixed_u + chain, chain

def coupling_fixed_xcart(potential, u: np.ndarray, eta: np.ndarray,
                         blocks: Optional[HessianBlocks] = None,
                         rprimd: Optional[np.ndarray] = None) -> np.ndarray:
    """Expected FD of the code's forces w.r.t. strain at fixed xcart (6, 3N).

    With the Python displacement definition (fixed reference xred, current
    rprimd, min-image wrap), u(eta) = u(0) - x_ref @ eta^T @ R^T exactly, so
        du/deta_nu |_xcart = -x_ref @ E_nu @ R^T
    and
        dF/deta_nu |_xcart = -coupling[nu] + ifc @ flat(x_ref @ E_nu @ R^T).
    (The Fortran Eq. A4 `_compute_du_delta` linearizes a different,
    evaluator-internal displacement frame and is NOT this derivative.)
    """
    if blocks is None:
        blocks = analytic_blocks(potential, u, eta)
    if rprimd is None:
        rprimd = potential._reference_lattice
    xref = potential.supercell.crystal_sc.xred
    gens = _engineering_generators()
    du = np.einsum('ni,vij,rj->vnr', xref, gens, rprimd)    # (6, N, 3)
    return -blocks.coupling + du.reshape(6, -1) @ blocks.ifc
