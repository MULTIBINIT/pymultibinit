"""
Supercell builder for effective potential.

This module provides functions to build supercell potentials from
unitcell data, including geometry replication, IFC mapping, and
dipole-dipole computation.

References:
- abinit/src/78_effpot/m_effective_potential.F90 (supercell generation)
- abinit/src/78_effpot/m_ifc.F90 (IFC handling)
- abinit/src/44_abitools/m_dynmat.F90 (Fourier transform ftifc_q2r, bigbx9)
"""

from typing import Tuple, List, Optional
import numpy as np
from .datastructures import (
    UnitcellData, 
    SupercellPotential, 
    CrystalInfo, 
    IFCData
)
from .dipdip import build_dipole_dipole_ifcs
from .symmetry import build_atom_mapping, rotate_dynamical_matrix_full


def _find_bound_supercell(ncell: int) -> Tuple[int, int]:
    # Fortran findBound_supercell (m_supercell.F90 line 670)
    # Assumes initial min=0, max=0 (else branch always taken):
    #   min = -(ncell)/2; max = -min; if even: min = min + 1
    min_val = -(ncell // 2)
    max_val = -min_val
    if ncell % 2 == 0:
        min_val += 1
    return min_val, max_val


def _generate_supercell_rpoints(ncell: Tuple[int, int, int]) -> np.ndarray:
    """
    Generate R-point grid for supercell.
    
    Returns cell indices (i1, i2, i3) for all R-points in supercell.
    
    Parameters
    ----------
    ncell : Tuple[int, int, int]
        Supercell dimensions
        
    Returns
    -------
    cell : (3, nrpt) array
        Integer cell indices for each R-point
    """
    min1, max1 = _find_bound_supercell(ncell[0])
    min2, max2 = _find_bound_supercell(ncell[1])
    min3, max3 = _find_bound_supercell(ncell[2])
    
    rpoints = []
    for i1 in range(min1, max1 + 1):
        for i2 in range(min2, max2 + 1):
            for i3 in range(min3, max3 + 1):
                rpoints.append([i1, i2, i3])
    
    return np.array(rpoints, dtype=int).T  # Shape: (3, nrpt)


def _bigbx9_rpoints(ngqpt: np.ndarray, nqshft: int = 1) -> np.ndarray:
    """
    Generate R-point grid using ABINIT bigbx9 algorithm (brav=1, simple cubic).
    
    For brav=1: lim = (ngqpt + 1) * lqshft + buffer, buffer=1
    nrpt = (2*lim+1)^3
    
    Parameters
    ----------
    ngqpt : (3,) array
        q-point grid dimensions
    nqshft : int
        Number of q-grid shifts (1 for unshifted grid)
        
    Returns
    -------
    cell : (3, nrpt) array
        Integer cell indices for each R-point
    """
    buffer = 1
    lqshft = 1 if nqshft == 1 else 2
    
    lim = [(int(ngqpt[i]) + 1) * lqshft + buffer for i in range(3)]
    
    rpoints = []
    for r1 in range(-lim[0], lim[0] + 1):
        for r2 in range(-lim[1], lim[1] + 1):
            for r3 in range(-lim[2], lim[2] + 1):
                rpoints.append([r1, r2, r3])
    
    return np.array(rpoints, dtype=int).T  # Shape: (3, nrpt)


def _generate_full_bz_qpoints(ngqpt: np.ndarray, nqshft: int = 1,
                               q1shft: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Generate all q-points in the full Brillouin zone (smpbz for brav=1).
    
    For a Gamma-centered grid with nqshft=1, q1shft=(0,0,0):
    qbz = {(i1/N1, i2/N2, i3/N3) : 0 <= ij < Nj}
    
    Parameters
    ----------
    ngqpt : (3,) array
        q-point grid dimensions (N1, N2, N3)
    nqshft : int
        Number of q-shifts
    q1shft : (nqshft, 3) array or None
        q-shifts (default: (0,0,0))
        
    Returns
    -------
    qbz : (nqbz, 3) array
        All q-points in the full BZ (fractional coordinates)
    """
    if q1shft is None:
        q1shft = np.zeros((max(nqshft, 1), 3))
    
    n1, n2, n3 = int(ngqpt[0]), int(ngqpt[1]), int(ngqpt[2])
    qbz = []
    for ishft in range(nqshft):
        shft = q1shft[ishft]
        for i1 in range(n1):
            for i2 in range(n2):
                for i3 in range(n3):
                    q = np.array([i1/n1 + shft[0]/n1,
                                  i2/n2 + shft[1]/n2,
                                  i3/n3 + shft[2]/n3])
                    # Fold to [-0.5, 0.5)
                    q = q - np.round(q)
                    qbz.append(q)
    
    return np.array(qbz)  # (nqbz, 3)


def _expand_dynmat_to_full_bz(
    qibz: np.ndarray,
    dynmat_ibz: np.ndarray,
    qbz: np.ndarray,
    symrel: np.ndarray,
    rprimd: np.ndarray,
    xred: np.ndarray,
    tnons: np.ndarray,
    use_rotation: bool = True
) -> np.ndarray:
    nqibz = len(qibz)
    nqbz = len(qbz)
    natom = dynmat_ibz.shape[1]
    nsym = len(symrel)
    tol = 2e-8
    
    dynmat_bz = np.zeros((nqbz, natom, 3, natom, 3, 2))
    found = np.zeros(nqbz, dtype=bool)
    
    do_rotation = use_rotation and tnons is not None
    if do_rotation:
        indsym = build_atom_mapping(xred, symrel, tnons, tol=1e-6)
    else:
        indsym = None
    
    symrec = np.zeros_like(symrel, dtype=float)
    for isym in range(nsym):
        symrec[isym] = np.linalg.inv(symrel[isym]).T
    
    for iqbz in range(nqbz):
        q = qbz[iqbz]
        
        for iqibz in range(nqibz):
            if found[iqbz]:
                break
            q_irr = qibz[iqibz]
            
            for isym in range(nsym):
                S_rec = symrec[isym]
                q_sym = S_rec @ q_irr
                
                diff = q - q_sym
                diff -= np.round(diff)
                if np.max(np.abs(diff)) < tol:
                    if do_rotation and indsym is not None:
                        dynmat_bz[iqbz] = rotate_dynamical_matrix_full(
                            dynmat_ibz[iqibz], q_irr, symrel[isym], tnons[isym],
                            indsym[:, isym, :], rprimd, time_reversal=False, q_target=q
                        )
                    else:
                        dynmat_bz[iqbz] = dynmat_ibz[iqibz]
                    found[iqbz] = True
                    break
                
                q_sym_tr = -q_sym
                diff_tr = q - q_sym_tr
                diff_tr -= np.round(diff_tr)
                if np.max(np.abs(diff_tr)) < tol:
                    if do_rotation and indsym is not None:
                        dynmat_bz[iqbz] = rotate_dynamical_matrix_full(
                            dynmat_ibz[iqibz], q_irr, symrel[isym], tnons[isym],
                            indsym[:, isym, :], rprimd, time_reversal=True, q_target=q
                        )
                    else:
                        dm = dynmat_ibz[iqibz].copy()
                        dm[..., 1] = -dm[..., 1]
                        dynmat_bz[iqbz] = dm
                    found[iqbz] = True
                    break
        
        if not found[iqbz]:
            raise ValueError(
                f"Could not find irreducible q-point for q={q}. "
                f"Check symmetry operations or q-point grid."
            )
    
    return dynmat_bz


def _wrap_to_pmhalf(x: np.ndarray) -> np.ndarray:
    """Wrap coordinates to [-0.5, 0.5)."""
    return x - np.round(x)


def _canonical_coordinates(xred: np.ndarray, rprim: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform atomic positions to canonical coordinates.
    
    Wraps positions to Wigner-Seitz cell and computes translation vectors.
    
    Parameters
    ----------
    xred : (natom, 3) array
        Reduced coordinates
    rprim : (3, 3) array
        Primitive lattice vectors
        
    Returns
    -------
    rcan : (natom, 3) array
        Canonical positions (Cartesian)
    trans : (natom, 3) array
        Translation vectors from original to canonical (Cartesian)
    """
    natom = len(xred)
    rcan = np.zeros((natom, 3))
    trans = np.zeros((natom, 3))
    
    for iat in range(natom):
        # Wrap to [-0.5, 0.5)
        rok = _wrap_to_pmhalf(xred[iat])
        
        # Canonical position in Cartesian
        rcan[iat] = rok @ rprim.T
        
        # Translation from original to canonical
        trans[iat] = (xred[iat] - rok) @ rprim.T
    
    return rcan, trans


def _apply_phase_shift(
    dynmat: np.ndarray,
    qpoints: np.ndarray,
    gprim: np.ndarray,
    trans: np.ndarray
) -> np.ndarray:
    """
    Apply phase shift to dynamical matrices.
    
    D'(q, a, b) = D(q, a, b) * exp(i*2*pi*q*(tau_a - tau_b))
    
    Parameters
    ----------
    dynmat : (nqpt, natom, 3, natom, 3, 2) array
        Dynamical matrices [real, imag]
    qpoints : (nqpt, 3) array
        q-point coordinates
    gprim : (3, 3) array
        Reciprocal lattice vectors
    trans : (natom, 3) array
        Translation vectors from canonical transformation
        
    Returns
    -------
    dynmat_shifted : (nqpt, natom, 3, natom, 3, 2) array
        Phase-shifted dynamical matrices
    """
    # kk[q, 3] = gprim @ qpoints[q]  (Cartesian reciprocal coords)
    kk = qpoints @ gprim.T  # (nqpt, 3)

    # diff_trans[a, b, 3] = trans[a] - trans[b]
    diff_trans = trans[:, np.newaxis, :] - trans[np.newaxis, :, :]  # (natom, natom, 3)

    # phase_arg[q, a, b] = kk[q] . diff_trans[a, b]
    # Here kk is 2pi*q_cart, and diff_trans is R_cart. 
    # So phase_arg is already the full phase (dimensionless, 2pi-scaled).
    phase_arg = np.einsum('qi,abi->qab', kk, diff_trans)  # (nqpt, natom, natom)

    phase_re = np.cos(phase_arg)  # (nqpt, natom, natom)
    phase_im = np.sin(phase_arg)

    # dynmat: (nqpt, natom, 3, natom, 3, 2) — last dim is [re, im]
    re = dynmat[..., 0]  # (nqpt, natom, 3, natom, 3)
    im = dynmat[..., 1]

    # Broadcast phase over mu, nu dims: phase shape (nqpt, natom, 1, natom, 1)
    pr = phase_re[:, :, np.newaxis, :, np.newaxis]
    pi_ = phase_im[:, :, np.newaxis, :, np.newaxis]

    dynmat_shifted = np.stack([re * pr - im * pi_, re * pi_ + im * pr], axis=-1)

    return dynmat_shifted


def _ftifc_q2r(
    dynmat: np.ndarray,
    qpoints: np.ndarray,
    gprim: np.ndarray,
    cell: np.ndarray
) -> np.ndarray:
    """
    Fourier transform from q-space to R-space (Pure Lattice).
    
    Phi(R) = (1/N_q) * Sum_q exp(-i*2*pi*q*R) * D_shifted(q)
    """
    nqpt = len(qpoints)
    nrpt = cell.shape[1]

    # phase_arg[q, R] = q . cell[R]
    phase_arg = qpoints @ cell  # (nqpt, nrpt) 
    phase_c = np.exp(-2j * np.pi * phase_arg)

    # Use complex dynmat directly
    if dynmat.dtype == complex:
        D_c = dynmat
    else:
        D_c = dynmat[..., 0] + 1j * dynmat[..., 1] # (nqpt, natom, 3, natom, 3)
    
    # Sum over q: 
    # phase_c: (q, r), D_c: (q, ia, mu, ib, nu)
    # result: (ia, mu, ib, nu, r)
    atmfrc_c = np.einsum('qr,qijkl->ijklr', phase_c, D_c) / nqpt

    return np.real(atmfrc_c)


def _compute_rpoint_weights(
    cell: np.ndarray,
    rprimd: np.ndarray,
    rcan: np.ndarray,
    ngqpt: np.ndarray,
    nqbz: int,
    toldist: float = 1e-8,
) -> np.ndarray:
    """
    Compute Wigner-Seitz weights for R-points (wght9 for brav=1).

    Translated from ABINIT m_dynmat.F90 wght9 (brav=1 branch).

    For each (ia, ib, irpt), the weight is determined by testing whether
    the displacement vector rdiff = tau_ib - tau_ia + R lies inside the
    Wigner-Seitz cell of the q-point superlattice (ngqpt * rprimd).

    Weight assignment:
      - Inside WS cell:           weight = 1.0
      - On one boundary plane:    weight = 1/2
      - On two boundary planes:   weight = 1/3
      - On N boundary planes:     weight = 1/(N+1)
      - Outside WS cell:          weight = 0.0

    Parameters
    ----------
    cell : (3, nrpt) integer cell indices for each R-point
    rprimd : (3, 3) real-space lattice vectors
    rcan : (natom, 3) canonical Cartesian positions
    ngqpt : (3,) q-point grid dimensions
    nqbz : total number of q-points in full BZ
    toldist : tolerance for boundary detection

    Returns
    -------
    wghatm : (natom, natom, nrpt) Wigner-Seitz weights

    Raises
    ------
    RuntimeError if weight sum rule is violated (sum != nqbz)
    """
    natom = rcan.shape[0]
    nrpt = cell.shape[1]

    # Build WS boundary points from the q-point superlattice.
    # Each boundary point is pp = n1*ngqpt[0]*a1 + n2*ngqpt[1]*a2 + n3*ngqpt[2]*a3
    # in Cartesian coordinates. Origin (0,0,0) is excluded (normsq=0).
    ptws_list = []
    ptws_normsq_list = []
    for ii in range(-2, 3):
        for jj in range(-2, 3):
            for kk in range(-2, 3):
                coeff = np.array([ii * ngqpt[0], jj * ngqpt[1], kk * ngqpt[2]],
                                 dtype=float)
                pp = rprimd @ coeff  # Cartesian boundary point
                normsq = np.dot(pp, pp)
                if normsq > 1e-12:
                    ptws_list.append(pp)
                    ptws_normsq_list.append(0.5 * normsq)

    ptws = np.array(ptws_list)           # (nptws, 3)
    ptws_normsq = np.array(ptws_normsq_list)  # (nptws,)

    # Precompute Cartesian R-vectors from integer cell indices
    # R_cart = rprimd @ cell_int for each R-point
    R_cart = (rprimd @ cell).T  # (nrpt, 3)

    wghatm = np.zeros((natom, natom, nrpt))

    for ia in range(natom):
        for ib in range(natom):
            # rdiff[irpt] = rcan[ib] - rcan[ia] + R_cart[irpt]
            rdiff = rcan[ib] - rcan[ia] + R_cart  # (nrpt, 3)

            # Project onto WS boundary points: proj[irpt, ipt] = rdiff[irpt] · ptws[ipt]
            proj = rdiff @ ptws.T  # (nrpt, nptws)

            # diff_from_boundary[irpt, ipt] = proj - 0.5*|pp|^2
            diff_from_boundary = proj - ptws_normsq[np.newaxis, :]  # (nrpt, nptws)

            # Outside WS cell: any boundary test fails (point is closer to pp than origin)
            outside = np.any(diff_from_boundary > toldist, axis=1)  # (nrpt,)

            # Count equidistant boundaries (point lies on the perpendicular bisector)
            equidistant = np.sum(np.abs(diff_from_boundary) <= toldist,
                                 axis=1)  # (nrpt,)

            # Weight = 1 / (1 + n_equidistant) if inside, 0 if outside
            nreq = 1 + equidistant
            weight = np.where(outside, 0.0, 1.0 / nreq.astype(float))

            wghatm[ia, ib, :] = weight

    # Verify sum rule: sum_{irpt} wghatm[ia, ib, irpt] == nqbz
    for ia in range(natom):
        for ib in range(natom):
            total = np.sum(wghatm[ia, ib, :])
            if abs(total - nqbz) > 1e-6:
                raise RuntimeError(
                    f"Weight sum rule violated for ({ia},{ib}): "
                    f"sum={total:.6f}, expected={nqbz}"
                )

    return wghatm


def _filter_rpoints_by_weights(
    wghatm: np.ndarray,
    cell: np.ndarray,
    atmfrc: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Filter out R-points where all weights are zero (cutmode=1).

    Matches ABINIT get_bigbox_and_weights filtering: an R-point is removed
    if sum(|wghatm[:,:,irpt]|) < tol20 for all atom pairs.

    Parameters
    ----------
    wghatm : (natom, natom, nrpt) Wigner-Seitz weights
    cell : (3, nrpt) integer cell indices
    atmfrc : (natom, 3, natom, 3, nrpt) real-space IFCs

    Returns
    -------
    cell_filtered : (3, nrpt_filtered)
    atmfrc_filtered : (natom, 3, natom, 3, nrpt_filtered)
    wghatm_filtered : (natom, natom, nrpt_filtered)
    """
    tol20 = 1e-20

    # Keep R-points where at least one weight is non-zero
    has_weight = np.sum(np.abs(wghatm), axis=(0, 1)) > tol20  # (nrpt,)

    return (cell[:, has_weight],
            atmfrc[:, :, :, :, has_weight],
            wghatm[:, :, has_weight])


def _apply_asr_weighted(
    atmfrc: np.ndarray,
    cell: np.ndarray,
    wghatm: np.ndarray,
) -> np.ndarray:
    """
    Apply weighted acoustic sum rule (asrif9 with asr=1).

    Translated from ABINIT m_dynmat.F90 asrif9.

    For each (mu, nu, ia):
      sumifc = sum_{ib, irpt} wghatm(ia, ib, irpt) * atmfrc(ia, mu, ib, nu, irpt)
      atmfrc(ia, mu, ia, nu, izero) -= sumifc

    where izero is the index of R=(0,0,0).

    Parameters
    ----------
    atmfrc : (natom, 3, natom, 3, nrpt) real-space IFCs
    cell : (3, nrpt) integer cell indices
    wghatm : (natom, natom, nrpt) Wigner-Seitz weights

    Returns
    -------
    atmfrc : ASR-corrected IFCs (modified in-place)
    """
    natom = atmfrc.shape[0]
    nrpt = atmfrc.shape[4]

    # Find R=(0,0,0) index
    izero = None
    for irpt in range(nrpt):
        if np.all(np.abs(cell[:, irpt]) < 1e-10):
            izero = irpt
            break

    if izero is None:
        raise ValueError("R=(0,0,0) not found in R-point grid")

    # Compute weighted sum: sumifc[ia, mu, nu] = sum_{ib,irpt} wghatm * atmfrc
    # wghatm: (natom, natom, nrpt) → expand to (natom, 1, natom, 1, nrpt)
    w = wghatm[:, np.newaxis, :, np.newaxis, :]
    sumifc = np.sum(w * atmfrc, axis=(2, 4))  # (natom, 3, 3)

    # Correct self-interaction at R=(0,0,0)
    for ia in range(natom):
        atmfrc[ia, :, ia, :, izero] -= sumifc[ia]

    return atmfrc


def _compute_dipdip_per_rpoint(
    unitcell: UnitcellData,
    cell: np.ndarray,
    rprimd_sc: np.ndarray
) -> np.ndarray:
    """
    Compute dipole-dipole IFCs for each R-point (unitcell basis).
    
    Matches ABINIT's effective_potential_generateDipDip:
    - Uses SUPERCELL geometry for Ewald summation
    - For R=(0,0,0): dipole-dipole within reference cell
    - For R≠(0,0,0): dipole-dipole between reference cell and shifted cell
    
    Parameters
    ----------
    unitcell : UnitcellData
        Unit cell with dielectric tensor, Born charges, positions
    cell : (3, nrpt) array
        Cell indices for each R-point
    rprimd_sc : (3, 3) array
        Supercell lattice vectors (for Ewald summation)
        
    Returns
    -------
    ewald_atmfrc : (natom_uc, 3, natom_uc, 3, nrpt) array
        Dipole-dipole IFCs for each R-point (unitcell basis)
    """
    from .dipdip import ewald_dipole_dipole_for_rpoint
    
    natom_uc = unitcell.natom
    nrpt = cell.shape[1]
    
    epsilon_inf = unitcell.epsilon_inf
    zeff = unitcell.zeff
    rprimd_uc = unitcell.rprimd
    xcart_uc = unitcell.xcart
    
    if epsilon_inf is None or zeff is None:
        return np.zeros((natom_uc, 3, natom_uc, 3, nrpt))
    
    ewald_atmfrc = np.zeros((natom_uc, 3, natom_uc, 3, nrpt))
    
    volume_sc = np.abs(np.linalg.det(rprimd_sc))
    
    for irpt in range(nrpt):
        cell_shift = cell[:, irpt]
        
        if cell_shift[0] == 0 and cell_shift[1] == 0 and cell_shift[2] == 0:
            ewald_atmfrc[:, :, :, :, irpt] = ewald_dipole_dipole_for_rpoint(
                xcart_uc, xcart_uc,
                epsilon_inf, zeff, zeff,
                rprimd_sc, volume_sc
            )
        else:
            R_cart = (cell_shift[0] * rprimd_uc[0] + cell_shift[1] * rprimd_uc[1]
                      + cell_shift[2] * rprimd_uc[2])
            xcart_shifted = xcart_uc + R_cart
            
            ewald_atmfrc[:, :, :, :, irpt] = ewald_dipole_dipole_for_rpoint(
                xcart_uc, xcart_shifted,
                epsilon_inf, zeff, zeff,
                rprimd_sc, volume_sc
            )
    
    return ewald_atmfrc


def _build_supercell_ifcs_fourier(
    unitcell: UnitcellData,
    crystal_sc: CrystalInfo,
    ncell: Tuple[int, int, int]
) -> IFCData:
    """
    Build supercell IFCs using Fourier transform from q-points.
    
    Implements the full ABINIT algorithm:
    1. Expand irreducible q-points to full BZ using symmetry (symdm9)
    2. Generate R-points using bigbx9 (based on ngqpt, NOT ncell)
    3. Apply canonical coordinate phase shift
    4. Fourier transform: full BZ q → R (total IFCs from DDB)
    5. Compute dipole-dipole for each R-point (using SUPERCELL geometry with ncell)
    6. Extract short-range: short = total - dipdip
    7. Apply ASR to short-range
    8. Total = short + dipdip
    9. Replicate to supercell using ncell
    
    Parameters
    ----------
    unitcell : UnitcellData
        Primitive cell with q-points and dynamical matrices
    crystal_sc : CrystalInfo
        Supercell crystal structure
    ncell : Tuple[int, int, int]
        Supercell dimensions
        
    Returns
    -------
    IFCData
        Supercell IFCs with proper R-point grid
    """
    from .datastructures import IFCData
    
    natom_uc = unitcell.natom
    natom_sc = crystal_sc.natom
    
    # Get ngqpt for bigbx9 R-point generation
    ngqpt = unitcell.ngqpt
    if ngqpt is None:
        # Fallback: infer from q-points or use ncell
        ngqpt = np.array(ncell, dtype=int)
    nqshft = unitcell.nqshft
    q1shft = unitcell.q1shft
    
    # Step 1: Generate full BZ q-points (smpbz equivalent)
    qbz = _generate_full_bz_qpoints(ngqpt, nqshft, q1shft)
    nqbz = len(qbz)
    
    # Step 2: Expand irreducible dynmat to full BZ (symdm9 equivalent)
    symrel = unitcell.symrel
    if symrel is None:
        # No symmetry: assume qibz == qbz (only works if DDB has full grid)
        dynmat_bz = unitcell.dynmat
        if dynmat_bz is None:
            raise ValueError("No dynamical matrices in unitcell")
    else:
        if unitcell.dynmat is None:
            raise ValueError("No dynamical matrices in unitcell")
        if unitcell.qpoints is None:
            raise ValueError("No q-points in unitcell")
        tnons = getattr(unitcell, 'tnons', None)
        if tnons is None:
            tnons = np.zeros((len(symrel), 3))
        dynmat_bz = _expand_dynmat_to_full_bz(
            unitcell.qpoints, unitcell.dynmat, qbz, symrel,
            unitcell.rprimd, unitcell.xred, tnons, use_rotation=True
        )
    
    assert dynmat_bz is not None  # guaranteed by checks above

    # Step 3: Generate R-points using bigbx9 (based on ngqpt, NOT ncell)
    cell_rpt = _bigbx9_rpoints(ngqpt, nqshft)
    nrpt = cell_rpt.shape[1]
    
    rprimd = unitcell.rprimd
    gprim = 2 * np.pi * np.linalg.inv(rprimd).T
    
    # Step 4: Subtract ewald from total dynmat in q-space (Fortran ifc_init)
    # This must happen BEFORE phase shift and FT, matching Fortran exactly.
    ewald_atmfrc_uc = None
    if unitcell.epsilon_inf is not None and unitcell.zeff is not None:
        if np.linalg.norm(unitcell.zeff) > 1e-10:
            from .dipdip import compute_dipdip_dynmat
            for iq in range(nqbz):
                ewald_q = compute_dipdip_dynmat(qbz[iq], unitcell, sumg0=0)
                dynmat_bz[iq, ..., 0] -= ewald_q.real
                dynmat_bz[iq, ..., 1] -= ewald_q.imag

    # Step 5: Canonical coordinate transform + phase shift
    rcan, trans = _canonical_coordinates(unitcell.xred, unitcell.rprimd)
    
    dynmat_shifted = _apply_phase_shift(
        dynmat_bz,
        qbz,
        gprim,
        trans
    )
    
    # Step 6: FT (output is short-range since ewald subtracted at Step 4)
    total_atmfrc_uc = _ftifc_q2r(
        dynmat_shifted,
        qbz,
        gprim,
        cell_rpt,
    )

    # Step 7: Compute Wigner-Seitz weights (wght9 for brav=1)
    wghatm = _compute_rpoint_weights(
        cell_rpt, rprimd, rcan, ngqpt, nqbz
    )

    # Step 8: Filter zero-weight R-points (cutmode=1)
    tol20 = 1e-20
    has_weight = np.sum(np.abs(wghatm), axis=(0, 1)) > tol20
    cell_rpt = cell_rpt[:, has_weight]
    total_atmfrc_uc = total_atmfrc_uc[:, :, :, :, has_weight]
    wghatm = wghatm[:, :, has_weight]
    nrpt = cell_rpt.shape[1]

    # Step 9: total_atmfrc_uc is already short-range (ewald subtracted in q-space at Step 4)
    short_atmfrc_uc = total_atmfrc_uc

    # Step 10: Apply weighted ASR to short-range (asrif9 with asr=1)
    short_atmfrc_uc = _apply_asr_weighted(short_atmfrc_uc, cell_rpt, wghatm)

    # --- Phase 2: generateDipDip (m_effective_potential.F90 lines 634-1176) ---
    # Map short-range to union R-point grid, combine with ewald, apply
    # second unweighted ASR (harmonics_terms_applySumRule), then replicate.

    # Step 11: Union of bigbx9 and supercell R-point ranges (generateDipDip lines 1001-1031)
    uc_min = cell_rpt.min(axis=1)
    uc_max = cell_rpt.max(axis=1)
    sc_min1, sc_max1 = _find_bound_supercell(ncell[0])
    sc_min2, sc_max2 = _find_bound_supercell(ncell[1])
    sc_min3, sc_max3 = _find_bound_supercell(ncell[2])
    sc_min = np.array([sc_min1, sc_min2, sc_min3])
    sc_max = np.array([sc_max1, sc_max2, sc_max3])
    union_min = np.minimum(uc_min, sc_min)
    union_max = np.maximum(uc_max, sc_max)

    union_rpts = []
    for i1 in range(int(union_min[0]), int(union_max[0]) + 1):
        for i2 in range(int(union_min[1]), int(union_max[1]) + 1):
            for i3 in range(int(union_min[2]), int(union_max[2]) + 1):
                union_rpts.append([i1, i2, i3])
    cell_union = np.array(union_rpts, dtype=int).T
    nrpt_union = cell_union.shape[1]

    # Step 12: Map short-range to union R-points (coordinate matching)
    uc_rpt_lookup = {}
    for irpt in range(nrpt):
        key = (int(cell_rpt[0, irpt]), int(cell_rpt[1, irpt]), int(cell_rpt[2, irpt]))
        uc_rpt_lookup[key] = irpt

    short_atmfrc_union = np.zeros((natom_uc, 3, natom_uc, 3, nrpt_union))
    for irpt_u in range(nrpt_union):
        r = cell_union[:, irpt_u]
        key = (int(r[0]), int(r[1]), int(r[2]))
        if key in uc_rpt_lookup:
            irpt_uc = uc_rpt_lookup[key]
            short_atmfrc_union[:, :, :, :, irpt_u] = short_atmfrc_uc[:, :, :, :, irpt_uc]

    # Step 13: Ewald for union R-points (supercell geometry)
    ewald_atmfrc_union = np.zeros((natom_uc, 3, natom_uc, 3, nrpt_union))
    if unitcell.epsilon_inf is not None and unitcell.zeff is not None:
        if np.linalg.norm(unitcell.zeff) > 1e-10:
            ewald_atmfrc_union = _compute_dipdip_per_rpoint(
                unitcell, cell_union, crystal_sc.rprimd
            )

    # Step 14: Combine short-range + ewald at union R-points
    atmfrc_union = short_atmfrc_union + ewald_atmfrc_union

    # Step 15: Unweighted ASR (m_harmonics_terms.F90 lines 724-820)
    # sum_{ib, irpt} atmfrc[ia, mu, ib, nu, irpt]; subtract from R=0 diagonal
    izero_union = None
    for irpt_u in range(nrpt_union):
        if np.all(np.abs(cell_union[:, irpt_u]) < 1e-10):
            izero_union = irpt_u
            break

    if izero_union is not None:
        for mu in range(3):
            for nu in range(3):
                for ia in range(natom_uc):
                    s = 0.0
                    for ib in range(natom_uc):
                        for irpt_u in range(nrpt_union):
                            s += atmfrc_union[ia, mu, ib, nu, irpt_u]
                    atmfrc_union[ia, mu, ia, nu, izero_union] -= s

    # Step 16: Replicate to supercell (fold union R-points into supercell grid)
    nx, ny, nz = ncell
    cell_sc = _generate_supercell_rpoints(ncell)
    nrpt_sc = cell_sc.shape[1]

    atmfrc_sc = np.zeros((natom_sc, 3, natom_sc, 3, nrpt_sc))
    short_atmfrc_sc = np.zeros((natom_sc, 3, natom_sc, 3, nrpt_sc))
    ewald_atmfrc_sc = np.zeros((natom_sc, 3, natom_sc, 3, nrpt_sc))

    sc_rpt_lookup = {}
    sc_bounds = []
    for dim in range(3):
        mn, mx = _find_bound_supercell(ncell[dim])
        sc_bounds.append((mn, mx))
    for irpt_sc in range(nrpt_sc):
        key = (int(cell_sc[0, irpt_sc]), int(cell_sc[1, irpt_sc]), int(cell_sc[2, irpt_sc]))
        sc_rpt_lookup[key] = irpt_sc

    def _fold_to_sc(r, dim):
        mn, _ = sc_bounds[dim]
        n = ncell[dim]
        return ((r - mn) % n) + mn

    for irpt_u in range(nrpt_union):
        r1, r2, r3 = int(cell_union[0, irpt_u]), int(cell_union[1, irpt_u]), int(cell_union[2, irpt_u])

        r1_sc = _fold_to_sc(r1, 0)
        r2_sc = _fold_to_sc(r2, 1)
        r3_sc = _fold_to_sc(r3, 2)
        irpt_sc = sc_rpt_lookup[(r1_sc, r2_sc, r3_sc)]

        for i_uc in range(natom_uc):
            for j_uc in range(natom_uc):
                ifc_u = atmfrc_union[i_uc, :, j_uc, :, irpt_u]
                short_u = short_atmfrc_union[i_uc, :, j_uc, :, irpt_u]
                ewald_u = ewald_atmfrc_union[i_uc, :, j_uc, :, irpt_u]

                for ix in range(nx):
                    for iy in range(ny):
                        for iz in range(nz):
                            i_sc = i_uc + natom_uc * (ix + nx * (iy + ny * iz))
                            jx = (ix + r1) % nx
                            jy = (iy + r2) % ny
                            jz = (iz + r3) % nz
                            j_sc = j_uc + natom_uc * (jx + nx * (jy + ny * jz))

                            atmfrc_sc[i_sc, :, j_sc, :, irpt_sc] += ifc_u
                            short_atmfrc_sc[i_sc, :, j_sc, :, irpt_sc] += short_u
                            ewald_atmfrc_sc[i_sc, :, j_sc, :, irpt_sc] += ewald_u

    return IFCData(
        nrpt=nrpt_sc,
        cell=cell_sc,
        atmfrc=atmfrc_sc,
        short_atmfrc=short_atmfrc_sc,
        ewald_atmfrc=ewald_atmfrc_sc,
        wghatm=wghatm,
    )


def build_supercell(unitcell: UnitcellData, ncell: Tuple[int, int, int]) -> SupercellPotential:
    """
    Build supercell potential from unitcell data.
    
    Parameters
    ----------
    unitcell : UnitcellData
        Primitive cell data (from DDB/XML)
    ncell : Tuple[int, int, int]
        Supercell dimensions (nx, ny, nz)
        
    Returns
    -------
    SupercellPotential
        Supercell ready for evaluation
        
    Examples
    --------
    >>> from pymultibinit.pyeffpot import read_ddb
    >>> unitcell = read_ddb("system.DDB")
    >>> supercell = build_supercell(unitcell, (4, 4, 4))
    >>> print(f"Supercell has {supercell.natom_sc} atoms")
    """
    # Step 1: Build supercell geometry
    crystal_sc = _build_supercell_geometry(unitcell.crystal, ncell)
    
    # Step 2: Build supercell IFCs
    if unitcell.qpoints is not None and unitcell.dynmat is not None:
        # Use full q→R Fourier transform (ABINIT-style)
        ifcs_sc = _build_supercell_ifcs_fourier(unitcell, crystal_sc, ncell)
    else:
        if unitcell.ifcs is None:
            raise ValueError("Unitcell IFCs not available")
        ifcs_sc = _replicate_ifcs(unitcell.ifcs, unitcell.crystal, crystal_sc, ncell)
        
        if unitcell.epsilon_inf is not None and unitcell.zeff is not None:
            _compute_dipole_dipole(ifcs_sc, unitcell, crystal_sc, ncell)
        
        _apply_asr(ifcs_sc)
    
    supercell = SupercellPotential(
        unitcell=unitcell,
        ncell=ncell,
        crystal_sc=crystal_sc,
        ifcs_sc=ifcs_sc,
        anharmonic_coeffs=None
    )
    
    return supercell


def _build_supercell_geometry(crystal_uc: CrystalInfo, ncell: Tuple[int, int, int]) -> CrystalInfo:
    """
    Build supercell crystal structure from unitcell.
    
    Parameters
    ----------
    crystal_uc : CrystalInfo
        Unitcell crystal structure
    ncell : Tuple[int, int, int]
        Supercell dimensions (nx, ny, nz)
        
    Returns
    -------
    CrystalInfo
        Supercell crystal structure
    """
    nx, ny, nz = ncell
    natom_uc = crystal_uc.natom
    natom_sc = natom_uc * nx * ny * nz
    
    rprimd_sc = np.diag(ncell) @ crystal_uc.rprimd
    
    xred_sc = np.zeros((natom_sc, 3))
    typat_sc = np.zeros(natom_sc, dtype=int)
    
    idx = 0
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                for iat in range(natom_uc):
                    xred_sc[idx, :] = crystal_uc.xred[iat, :] + np.array([ix, iy, iz])
                    typat_sc[idx] = crystal_uc.typat[iat]
                    idx += 1
    
    xred_sc[:, 0] /= nx
    xred_sc[:, 1] /= ny
    xred_sc[:, 2] /= nz
    
    xcart_sc = xred_sc @ rprimd_sc.T
    
    return CrystalInfo(
        natom=natom_sc,
        ntypat=crystal_uc.ntypat,
        rprimd=rprimd_sc,
        xred=xred_sc,
        xcart=xcart_sc,
        typat=typat_sc,
        amu=crystal_uc.amu.copy(),
        znucl=crystal_uc.znucl.copy()
    )


def _replicate_ifcs(ifcs_uc: IFCData, crystal_uc: CrystalInfo, 
                    crystal_sc: CrystalInfo, ncell: Tuple[int, int, int]) -> IFCData:
    """
    Replicate short-range IFCs from unitcell to supercell.
    """
    if ifcs_uc is None:
        raise ValueError("Unitcell IFCs not available")
    
    nx, ny, nz = ncell
    natom_uc = crystal_uc.natom
    natom_sc = crystal_sc.natom
    nrpt_uc = ifcs_uc.nrpt
    
    nrpt_sc = nrpt_uc
    atmfrc_sc = np.zeros((natom_sc, 3, natom_sc, 3, nrpt_uc))
    short_atmfrc_sc = np.zeros((natom_sc, 3, natom_sc, 3, nrpt_uc))
    cell_sc = ifcs_uc.cell.copy()
    
    for irpt in range(nrpt_uc):
        cell_shift = ifcs_uc.cell[:, irpt] if ifcs_uc.cell.ndim == 2 else np.zeros(3, dtype=int)
        for i_uc in range(natom_uc):
            for j_uc in range(natom_uc):
                ifc_uc = ifcs_uc.short_atmfrc[i_uc, :, j_uc, :, irpt]
                
                for ix in range(nx):
                    for iy in range(ny):
                        for iz in range(nz):
                            i_sc = i_uc + natom_uc * (ix + nx * (iy + ny * iz))
                            jx = (ix + int(cell_shift[0])) % nx
                            jy = (iy + int(cell_shift[1])) % ny
                            jz = (iz + int(cell_shift[2])) % nz
                            j_sc = j_uc + natom_uc * (jx + nx * (jy + ny * jz))
                            
                            short_atmfrc_sc[i_sc, :, j_sc, :, irpt] = ifc_uc
                            atmfrc_sc[i_sc, :, j_sc, :, irpt] = ifc_uc
    
    return IFCData(
        nrpt=nrpt_sc,
        cell=cell_sc,
        atmfrc=atmfrc_sc,
        short_atmfrc=short_atmfrc_sc,
        ewald_atmfrc=None
    )


def _compute_dipole_dipole(ifcs_sc: IFCData, unitcell: UnitcellData, 
                           crystal_sc: CrystalInfo, ncell: Tuple[int, int, int]):
    """
    Compute dipole-dipole (Ewald) contribution to IFCs at Gamma point.
    """
    natom_sc = crystal_sc.natom
    natom_uc = unitcell.crystal.natom
    nrpt_sc = ifcs_sc.nrpt
    
    ifcs_sc.ewald_atmfrc = np.zeros((natom_sc, 3, natom_sc, 3, nrpt_sc))
    
    if unitcell.epsilon_inf is None or unitcell.zeff is None:
        return
    
    if np.linalg.norm(unitcell.zeff) < 1e-10:
        return
    
    ncell_prod = ncell[0] * ncell[1] * ncell[2]
    zeff_sc = np.zeros((natom_sc, 3, 3))
    for i_sc in range(natom_sc):
        i_uc = i_sc % natom_uc
        zeff_sc[i_sc, :, :] = unitcell.zeff[i_uc, :, :]
    
    dd_ifcs = build_dipole_dipole_ifcs(
        positions_cart=crystal_sc.xcart,
        epsilon_inf=unitcell.epsilon_inf,
        zeff=zeff_sc,
        lattice_vectors=crystal_sc.rprimd,
        use_ewald=True,
    )
    
    ifcs_sc.ewald_atmfrc[:, :, :, :, 0] = dd_ifcs
    ifcs_sc.atmfrc[:, :, :, :, 0] += dd_ifcs


def _apply_asr(ifcs_sc: IFCData):
    """Apply Acoustic Sum Rule correction to IFCs."""
    natom = ifcs_sc.atmfrc.shape[1]
    
    for irpt in range(ifcs_sc.nrpt):
        for i in range(natom):
            for mu in range(3):
                sum_ifc = np.sum(ifcs_sc.atmfrc[i, mu, :, :, irpt])
                correction = sum_ifc / (3 * natom)
                ifcs_sc.atmfrc[i, mu, :, :, irpt] -= correction
                ifcs_sc.short_atmfrc[mu, i, :, :, irpt] -= correction


def set_anharmonic_coeffs(supercell: SupercellPotential, coeffs: List):
    """
    Set anharmonic coefficients for supercell.
    """
    supercell.anharmonic_coeffs = coeffs


