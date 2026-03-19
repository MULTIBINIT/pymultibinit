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


def _find_bound_supercell(ncell: int) -> Tuple[int, int]:
    """
    Find R-point bounds for supercell.
    
    For even ncell (e.g., 2): min = -1, max = 2
    For odd ncell (e.g., 3):  min = -1, max = 1
    
    Matches ABINIT's findBound_supercell function.
    """
    if ncell % 2 == 0:
        min_val = -ncell // 2
        max_val = ncell // 2
    else:
        min_val = -(ncell - 1) // 2
        max_val = (ncell - 1) // 2
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
    symrel: np.ndarray
) -> np.ndarray:
    """
    Expand dynamical matrices from irreducible BZ to full BZ using symmetry.
    
    Implements ABINIT's symdm9 algorithm: for each q in the full BZ, find
    an irreducible q_ibz and symmetry S such that q = S * q_ibz (or -S*q_ibz
    for time reversal), then use D(q) from D(q_ibz).
    
    NOTE: This function does NOT rotate the dynamical matrix; it only finds
    which irreducible q-point maps to each full BZ q-point. For the Fourier
    transform, we only need the correct assignment of D(q) values.
    
    For a simple cubic lattice the symmetry operations are integer matrices
    acting on reciprocal coordinates: q_sym = symrec @ q_ibz where
    symrec = symrel^{-T} (for orthogonal symrel, symrec = symrel).
    
    Parameters
    ----------
    qibz : (nqibz, 3) array
        Irreducible q-points (from DDB)
    dynmat_ibz : (nqibz, natom, 3, natom, 3, 2) array
        Dynamical matrices at irreducible q-points
    qbz : (nqbz, 3) array
        Full BZ q-points
    symrel : (nsym, 3, 3) int array
        Symmetry operations (in reduced coordinates)
        
    Returns
    -------
    dynmat_bz : (nqbz, natom, 3, natom, 3, 2) array
        Dynamical matrices at all full BZ q-points
    """
    nqibz = len(qibz)
    nqbz = len(qbz)
    natom = dynmat_ibz.shape[1]
    nsym = len(symrel)
    tol = 2e-8
    
    dynmat_bz = np.zeros((nqbz, natom, 3, natom, 3, 2))
    found = np.zeros(nqbz, dtype=bool)
    
    for iqbz in range(nqbz):
        q = qbz[iqbz]
        
        for iqibz in range(nqibz):
            if found[iqbz]:
                break
            q_irr = qibz[iqibz]
            
            for isym in range(nsym):
                S = symrel[isym].astype(float)
                # q_sym = S @ q_irr (symrec for cubic = symrel since symrel is orthogonal)
                q_sym = S @ q_irr
                
                # Check direct match
                diff = q - q_sym
                diff -= np.round(diff)
                if np.max(np.abs(diff)) < tol:
                    dynmat_bz[iqbz] = dynmat_ibz[iqibz]
                    found[iqbz] = True
                    break
                
                # Check time-reversal: q = -S @ q_irr => D(q) = D(-q) = D*(q_irr)
                q_sym_tr = -q_sym
                diff_tr = q - q_sym_tr
                diff_tr -= np.round(diff_tr)
                if np.max(np.abs(diff_tr)) < tol:
                    # D(q) = D(-q) = conj(D(q_irr)) for time reversal
                    dm = dynmat_ibz[iqibz].copy()
                    dm[..., 1] = -dm[..., 1]  # conjugate: flip imaginary part
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
    phase_arg = np.einsum('qi,abi->qab', kk, diff_trans)  # (nqpt, natom, natom)

    phase_re = np.cos(2 * np.pi * phase_arg)  # (nqpt, natom, natom)
    phase_im = np.sin(2 * np.pi * phase_arg)

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
    Fourier transform from q-space to R-space (ABINIT ftifc_q2r).
    
    Phi(R) = (1/N_q) * Sum_q exp(i*2*pi*q*R) * D(q)
    
    Parameters
    ----------
    dynmat : (nqpt, natom, 3, natom, 3, 2) array
        Dynamical matrices [real, imag] for FULL BZ
    qpoints : (nqpt, 3) array
        Full BZ q-point coordinates
    gprim : (3, 3) array
        Reciprocal lattice vectors
    cell : (3, nrpt) array
        R-point cell indices (from bigbx9)
        
    Returns
    -------
    atmfrc : (3, natom, 3, natom, nrpt) array
        Real-space IFCs (real-valued)
    """
    nqpt = len(qpoints)
    nrpt = cell.shape[1]

    # kk[q, 3] = gprim @ q  (Cartesian reciprocal)
    kk = qpoints @ gprim.T  # (nqpt, 3)

    # rpt[3, nrpt] -> rpt.T = (nrpt, 3)
    rpt = cell.T.astype(float)  # (nrpt, 3)

    # phase_arg[q, R] = kk[q] . rpt[R]
    phase_arg = kk @ rpt.T  # (nqpt, nrpt)

    phase_re = np.cos(2 * np.pi * phase_arg)  # (nqpt, nrpt)
    phase_im = np.sin(2 * np.pi * phase_arg)

    # dynmat: (nqpt, natom, 3, natom, 3, 2)
    # Convert to complex: D_c[q, ia, mu, ib, nu]
    D_re = dynmat[..., 0]  # (nqpt, natom, 3, natom, 3)
    D_im = dynmat[..., 1]
    D_c = D_re + 1j * D_im  # (nqpt, natom, 3, natom, 3)

    # phase_c[q, R] = phase_re + i*phase_im
    phase_c = phase_re + 1j * phase_im  # (nqpt, nrpt)

    # Sum over q: atmfrc_c[ia, mu, ib, nu, R] = sum_q phase_c[q,R] * D_c[q, ia, mu, ib, nu]
    # Use einsum: 'qr,qabcd->abcdr' where D_c has shape (nqpt, natom, 3, natom, 3)
    atmfrc_c = np.einsum('qr,qijkl->ijklr', phase_c, D_c) / nqpt  # (natom, 3, natom, 3, nrpt)

    # atmfrc_c is real (imaginary part should be ~0 for physical IFCs)
    # Transpose from (ia, mu, ib, nu, R) to (mu, ia, nu, ib, R)
    atmfrc = np.real(atmfrc_c).transpose(1, 0, 3, 2, 4)  # (3, natom, 3, natom, nrpt)

    return atmfrc


def _apply_asr_to_array(atmfrc: np.ndarray) -> np.ndarray:
    """
    Apply acoustic sum rule.
    
    Ensures translational invariance: Sum_{j,R} Phi(mu, i, nu, j, R) = 0
    
    Parameters
    ----------
    atmfrc : (3, natom, 3, natom, nrpt) array
        Real-space IFCs
        
    Returns
    -------
    atmfrc : (3, natom, 3, natom, nrpt) array
        ASR-corrected IFCs
    """
    natom = atmfrc.shape[1]
    nrpt = atmfrc.shape[4]
    
    for mu in range(3):
        for ia in range(natom):
            total = 0.0
            for nu in range(3):
                for ib in range(natom):
                    total += np.sum(atmfrc[mu, ia, nu, ib, :])
            
            correction = total / (3 * natom * nrpt)
            for nu in range(3):
                for ib in range(natom):
                    atmfrc[mu, ia, nu, ib, :] -= correction
    
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
    ewald_atmfrc : (3, natom_uc, 3, natom_uc, nrpt) array
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
        return np.zeros((3, natom_uc, 3, natom_uc, nrpt))
    
    ewald_atmfrc = np.zeros((3, natom_uc, 3, natom_uc, nrpt))
    
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
        dynmat_bz = _expand_dynmat_to_full_bz(
            unitcell.qpoints, unitcell.dynmat, qbz, symrel
        )
    
    assert dynmat_bz is not None  # guaranteed by checks above

    # Step 3: Generate R-points using bigbx9 (based on ngqpt, NOT ncell)
    cell_rpt = _bigbx9_rpoints(ngqpt, nqshft)
    nrpt = cell_rpt.shape[1]
    
    rprimd = unitcell.rprimd
    gprim = 2 * np.pi * np.linalg.inv(rprimd).T
    
    # Step 4: Canonical coordinate transform + phase shift
    rcan, trans = _canonical_coordinates(unitcell.xred, unitcell.rprimd)
    
    dynmat_shifted = _apply_phase_shift(
        dynmat_bz,
        qbz,
        gprim,
        trans
    )
    
    # Step 5: Fourier transform full BZ → R-points
    total_atmfrc_uc = _ftifc_q2r(
        dynmat_shifted,
        qbz,
        gprim,
        cell_rpt
    )
    
    # Step 6: Compute dipole-dipole for each R-point using SUPERCELL geometry
    # The dipole-dipole is computed at Gamma of the supercell BZ
    ewald_atmfrc_uc = np.zeros_like(total_atmfrc_uc)
    if unitcell.epsilon_inf is not None and unitcell.zeff is not None:
        if np.linalg.norm(unitcell.zeff) > 1e-10:
            ewald_atmfrc_uc = _compute_dipdip_per_rpoint(
                unitcell, cell_rpt, crystal_sc.rprimd
            )
    
    # Step 7: Extract short-range IFCs
    short_atmfrc_uc = total_atmfrc_uc - ewald_atmfrc_uc
    
    # Step 8: Apply ASR to short-range
    short_atmfrc_uc = _apply_asr_to_array(short_atmfrc_uc)
    
    # Step 9: Combine
    atmfrc_uc = short_atmfrc_uc + ewald_atmfrc_uc
    
    # Step 10: Replicate to supercell using ncell
    nx, ny, nz = ncell
    
    # Generate supercell R-points (same grid as bigbx9 R-points, used for indexing)
    cell_sc = _generate_supercell_rpoints(ncell)
    nrpt_sc = cell_sc.shape[1]
    
    atmfrc_sc = np.zeros((3, natom_sc, 3, natom_sc, nrpt_sc))
    short_atmfrc_sc = np.zeros((3, natom_sc, 3, natom_sc, nrpt_sc))
    ewald_atmfrc_sc = np.zeros((3, natom_sc, 3, natom_sc, nrpt_sc))
    
    # Build lookup: cell index -> R-point index in bigbx9 grid
    cell_dict = {}
    for irpt in range(nrpt):
        key = (int(cell_rpt[0, irpt]), int(cell_rpt[1, irpt]), int(cell_rpt[2, irpt]))
        cell_dict[key] = irpt
    
    for irpt_sc in range(nrpt_sc):
        cell_shift = cell_sc[:, irpt_sc]
        key = (int(cell_shift[0]), int(cell_shift[1]), int(cell_shift[2]))
        
        if key not in cell_dict:
            # R-point not in bigbx9 grid — skip (IFC is effectively zero beyond cutoff)
            continue
        
        irpt_uc = cell_dict[key]
        
        for i_uc in range(natom_uc):
            for j_uc in range(natom_uc):
                ifc_uc = atmfrc_uc[:, i_uc, :, j_uc, irpt_uc]
                short_ifc_uc = short_atmfrc_uc[:, i_uc, :, j_uc, irpt_uc]
                ewald_ifc_uc = ewald_atmfrc_uc[:, i_uc, :, j_uc, irpt_uc]
                
                for ix in range(nx):
                    for iy in range(ny):
                        for iz in range(nz):
                            i_sc = i_uc + natom_uc * (ix + nx * (iy + ny * iz))
                            jx = (ix + int(cell_shift[0])) % nx
                            jy = (iy + int(cell_shift[1])) % ny
                            jz = (iz + int(cell_shift[2])) % nz
                            j_sc = j_uc + natom_uc * (jx + nx * (jy + ny * jz))
                            
                            atmfrc_sc[:, i_sc, :, j_sc, irpt_sc] = ifc_uc
                            short_atmfrc_sc[:, i_sc, :, j_sc, irpt_sc] = short_ifc_uc
                            ewald_atmfrc_sc[:, i_sc, :, j_sc, irpt_sc] = ewald_ifc_uc
    
    return IFCData(
        nrpt=nrpt_sc,
        cell=cell_sc,
        atmfrc=atmfrc_sc,
        short_atmfrc=short_atmfrc_sc,
        ewald_atmfrc=ewald_atmfrc_sc
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
    atmfrc_sc = np.zeros((3, natom_sc, 3, natom_sc, nrpt_sc))
    short_atmfrc_sc = np.zeros((3, natom_sc, 3, natom_sc, nrpt_sc))
    cell_sc = ifcs_uc.cell.copy()
    
    for irpt in range(nrpt_uc):
        cell_shift = ifcs_uc.cell[:, irpt] if ifcs_uc.cell.ndim == 2 else np.zeros(3, dtype=int)
        for i_uc in range(natom_uc):
            for j_uc in range(natom_uc):
                ifc_uc = ifcs_uc.short_atmfrc[:, i_uc, :, j_uc, irpt]
                
                for ix in range(nx):
                    for iy in range(ny):
                        for iz in range(nz):
                            i_sc = i_uc + natom_uc * (ix + nx * (iy + ny * iz))
                            jx = (ix + int(cell_shift[0])) % nx
                            jy = (iy + int(cell_shift[1])) % ny
                            jz = (iz + int(cell_shift[2])) % nz
                            j_sc = j_uc + natom_uc * (jx + nx * (jy + ny * jz))
                            
                            short_atmfrc_sc[:, i_sc, :, j_sc, irpt] = ifc_uc
                            atmfrc_sc[:, i_sc, :, j_sc, irpt] = ifc_uc
    
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
    
    ifcs_sc.ewald_atmfrc = np.zeros((3, natom_sc, 3, natom_sc, nrpt_sc))
    
    if unitcell.epsilon_inf is None or unitcell.zeff is None:
        return
    
    if np.linalg.norm(unitcell.zeff) < 1e-10:
        return
    
    ncell_prod = ncell[0] * ncell[1] * ncell[2]
    zeff_sc = np.zeros((3, 3, natom_sc))
    for i_sc in range(natom_sc):
        i_uc = i_sc % natom_uc
        zeff_sc[:, :, i_sc] = unitcell.zeff[:, :, i_uc]
    
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
                sum_ifc = np.sum(ifcs_sc.atmfrc[mu, i, :, :, irpt])
                correction = sum_ifc / (3 * natom)
                ifcs_sc.atmfrc[mu, i, :, :, irpt] -= correction
                ifcs_sc.short_atmfrc[mu, i, :, :, irpt] -= correction


def set_anharmonic_coeffs(supercell: SupercellPotential, coeffs: List):
    """
    Set anharmonic coefficients for supercell.
    """
    supercell.anharmonic_coeffs = coeffs


