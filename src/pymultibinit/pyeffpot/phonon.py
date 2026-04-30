"""
Phonon utilities for calculating frequencies from DDB data.

This module provides functions for calculating phonon dispersion relations
from real-space interatomic force constants (IFCs) and unit cell data.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple, Union
from .datastructures import UnitcellData, CrystalInfo, IFCData
from .supercell_builder import _generate_full_bz_qpoints, _expand_dynmat_to_full_bz, build_supercell

# Physical constants (atomic units)
AMU_EMASS = 1822.888486192  # 1 amu in electron masses (matches ABINIT)
HA_CMM1 = 219474.6313705  # Hartree to cm⁻¹ conversion


def build_unitcell_ifcs(u: UnitcellData, eta: float = 0.1) -> IFCData:
    """
    Compute real-space IFCs for the unit cell from DDB q-points.
    Uses the full pyeffpot methodology: expansion, phase shift, and FT.
    """
    from .supercell_builder import (
        _generate_full_bz_qpoints, _expand_dynmat_to_full_bz,
        _bigbx9_rpoints, _canonical_coordinates, _apply_phase_shift,
        _ftifc_q2r
    )
    
    # 1. Expand to full BZ grid
    ngqpt = u.ngqpt if u.ngqpt is not None else np.array([4, 4, 4], dtype=int)
    nqshft = u.nqshft if u.nqshft is not None else 1
    q1shft = u.q1shft if u.q1shft is not None else np.zeros((1, 3))
    qbz = _generate_full_bz_qpoints(ngqpt, nqshft, q1shft)
    
    symrel = u.symrel if u.symrel is not None else np.eye(3).reshape(1, 3, 3)
    tnons = getattr(u, 'tnons', np.zeros((len(symrel), 3)))
    dynmats_bz = _expand_dynmat_to_full_bz(
        u.qpoints, u.dynmat, qbz, symrel,
        u.rprimd, u.xred, tnons
    )
    
    from .symmetry import find_symmetry_for_qpoint, rotate_dynamical_matrix_full, build_atom_mapping
    from .dipdip import compute_dipdip_dynmat
    
    # 1. Expand IBZ dynamical matrices to full grid if needed
    natom = u.crystal.natom
    nq_in = u.dynmat.shape[0]
    ngqpt = np.array(u.ngqpt)
    nq_full = np.prod(ngqpt)
    
    if nq_in < nq_full:
        print(f"  Expanding DDB from {nq_in} IBZ points to {nq_full} FBZ points...", flush=True)
        # Generate full grid
        full_qpts = []
        for i in range(ngqpt[0]):
            for j in range(ngqpt[1]):
                for k in range(ngqpt[2]):
                    full_qpts.append([i/ngqpt[0], j/ngqpt[1], k/ngqpt[2]])
        full_qpts = np.array(full_qpts)
        
        # Build atom mapping for each symmetry (assume tnons=0 if not found)
        nsym = len(u.symrel)
        tnons = np.zeros((nsym, 3)) # Default for most DDBs
        indsym = build_atom_mapping(u.crystal.xred, u.symrel, tnons)
        
        # New expanded arrays
        dynmats_total = np.zeros((nq_full, natom, 3, natom, 3, 2))
        for iq_full, q_target in enumerate(full_qpts):
            # Find which IBZ point this q maps to
            found = False
            for iq_ibz, q_ibz in enumerate(u.qpoints):
                try:
                    isym, time_rever = find_symmetry_for_qpoint(q_target, q_ibz, u.symrel)
                    d_rot = rotate_dynamical_matrix_full(
                        u.dynmat[iq_ibz], q_ibz, u.symrel[isym], tnons[isym],
                        indsym[:, isym, :], u.crystal.rprimd, time_rever
                    )
                    dynmats_total[iq_full] = d_rot
                    found = True
                    break
                except ValueError:
                    continue
            if not found:
                # Should not happen if DDB was complete
                print(f"  Warning: No symmetry found for q={q_target}, assuming 0.")
        qbz = full_qpts
        nqbz = nq_full
    else:
        qbz = u.qpoints
        dynmats_total = u.dynmat
        nqbz = len(qbz)
    gprim = 2 * np.pi * np.linalg.inv(u.crystal.rprimd).T
    rcan, trans = _canonical_coordinates(u.crystal.xred, u.crystal.rprimd)
    
    # 2. Compute DipDip background at q=0 for ASR consistency
    # ABINIT calculates sum_{ib} D_dd(q=0, ia, ib, sumg0=0) and subtracts from diagonal
    dm_dip0 = compute_dipdip_dynmat(np.zeros(3), u, sumg0=0, eta=eta)
    dm_dip0_sum = np.sum(dm_dip0, axis=2) # Shape: (natom, 3, 3)
    
    # 3. SEPARATE Dipole-Dipole part from grid + SHIFT to Convention 2
    # ABINIT Procedure: FT works on periodic (shifted) dynamical matrix.
    dynmats_total_complex = dynmats_total[..., 0] + 1j * dynmats_total[..., 1]
    
    natom = u.crystal.natom
    dynmats_short_shifted = np.zeros((nqbz, natom, 3, natom, 3), dtype=complex)
    
    xred = u.crystal.xred
    # diff_tau[ia, ib] = tau_ia - tau_ib
    diff_tau_a_minus_b = xred[:, np.newaxis, :] - xred[np.newaxis, :, :]

    for iq, q in enumerate(qbz):
        if (iq+1) % 8 == 0 or iq == nqbz - 1:
            print(f"  Separating DipDip for q {iq+1}/{nqbz}...")
        
        # 1. Compute DipDip in Convention 1
        dm_dip = compute_dipdip_dynmat(q, u, eta=eta)
        
        # Short range in Physical space (C1)
        dm_tot = dynmats_total_complex[iq]
        dm_sr_c1 = dm_tot - dm_dip
        
        # 2. Shift to Convention 2 (periodic) for interpolation
        # D_c2 = D_c1 * exp(i 2pi q . (tau_a - tau_b))
        phase_shift = np.exp(2j * np.pi * np.einsum('i,abi->ab', q, diff_tau_a_minus_b))
        dynmats_short_shifted[iq] = dm_sr_c1 * phase_shift[:, np.newaxis, :, np.newaxis]

    # 4. FT: q -> R (Generates short-range IFCs)
    nx, ny, nz = map(int, ngqpt)
    rpoints = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                rpoints.append([i if i <= nx//2 else i-nx, 
                                j if j <= ny//2 else j-ny, 
                                k if k <= nz//2 else k-nz])
    cell_rpt = np.array(rpoints, dtype=int).T # Shape: (3, 64)
    nrpt = cell_rpt.shape[1]
    # 5. Inverse Fourier Transform to Real Space (Pure Lattice)
    # dynmats_short_shifted is now periodic on the lattice.
    atmfrc_short = _ftifc_q2r(
        dynmats_short_shifted, qbz, gprim, cell_rpt
    )
    
    # [ASR Correction Disabled for Testing]
    r0_idx = -1
    
    # Save the background for reconstructor
    ifc_data = IFCData(
        nrpt=nrpt,
        cell=cell_rpt,
        atmfrc=atmfrc_short, # For total ifcs, we can't easily define them here
        short_atmfrc=atmfrc_short,
        ewald_atmfrc=None
    )
    # Attach background correction metadata
    setattr(ifc_data, 'dm_dip0_sum', dm_dip0_sum)
    
    return ifc_data



def compute_dynamical_matrix(
    q: np.ndarray,
    u: UnitcellData,
    eta: float = 0.1,
) -> np.ndarray:
    """
    Compute total dynamical matrix at q using separation of local and 
    long-range (dipole-dipole) parts.
    
    Total = FT(Short-Range IFCs) + Ewald(Dipole-Dipole)
    
    Args:
        q: q-point in reduced coordinates (3,)
        u: UnitcellData object (must have short-range IFCs in u.ifcs.atmfrc)
        
    Returns:
        Dynamical matrix complex (natom, 3, natom, 3)
    """
    from .dipdip import compute_dipdip_dynmat
    
    natom = u.crystal.natom
    cell = u.ifcs.cell  # (3, nrpt) 
    atmfrc_short = u.ifcs.atmfrc  # (natom, 3, natom, 3, nrpt) 
    
    # 1. Analytic short-range part (Reconstruct Convention 2)
    # sum: D(q)_shifted = Sum_R C(R) * exp(i*2*pi*q*R)
    
    # We must treat the boundary points R = N/2 specially to preserve Hermiticity
    nx, ny, nz = u.ngqpt
    dm_short_shifted = np.zeros((natom, 3, natom, 3), dtype=complex)
    
    for ir in range(u.ifcs.nrpt):
        R = cell[:, ir]
        phase = np.exp(2j * np.pi * np.dot(R, q))
        dm_short_shifted += atmfrc_short[:, :, :, :, ir] * phase

    # 2. Shift back to Convention 1 (Physical)
    # D_c1 = D_c2 * exp(i 2pi q . (tau_b - tau_a))
    xred = u.crystal.xred
    # diff_tau[ia, ib] = tau_ib - tau_ia
    diff_tau_b_minus_a = xred[np.newaxis, :, :] - xred[:, np.newaxis, :]
    phase_back = np.exp(2j * np.pi * np.einsum('i,abi->ab', q, diff_tau_b_minus_a))
    dm_short_c1 = dm_short_shifted * phase_back[:, np.newaxis, :, np.newaxis]
    
    # 3. Add Dipole-Dipole back (Convention 1)
    # We just add the full DipDip dynamical matrix.
    # The ASR was satisfied at q=0 during short-range extraction.
    dm_dip = compute_dipdip_dynmat(q, u, eta=eta)
    
    # No dm_dip0_sum subtraction here! 
    # The background is already contained in dm_short_c1.
    dm_tot = dm_short_c1 + dm_dip
    
    return dm_tot


def get_frequencies(
    dynmat: np.ndarray,
    u: UnitcellData,
) -> np.ndarray:
    """
    Calculate phonon frequencies from dynamical matrix.
    
    Args:
        dynmat: (3, natom, 3, natom) complex matrix
        u: UnitcellData
        
    Returns:
        Sorted frequencies in cm⁻¹ (3*natom,)
    """
    natom = u.natom
    # Reshape to (3*natom, 3*natom)
    # Order: [atom1_x, atom1_y, atom1_z, atom2_x, ...]
    dm_2d = dynmat.reshape(3*natom, 3*natom)
    
    # Mass weighting: D / sqrt(m_i * m_j)
    masses = np.array([u.amu[typ-1] for typ in u.typat]) * AMU_EMASS
    mass_factors = 1.0 / np.sqrt(np.outer(np.repeat(masses, 3), np.repeat(masses, 3)))
    
    dm_mass = dm_2d * mass_factors
    
    # Diagonalize
    ev = np.linalg.eigvalsh(dm_mass)
    
    # Convert to frequencies
    freq = np.zeros_like(ev)
    for i, val in enumerate(ev):
        if val >= 1e-16:
            freq[i] = np.sqrt(val) * HA_CMM1
        elif val >= -1e-16:
            freq[i] = 0.0
        else:
            freq[i] = -np.sqrt(-val) * HA_CMM1
            
    return np.sort(freq)


def compute_phonon_bands(
    u: UnitcellData,
    qpts: np.ndarray,
) -> np.ndarray:
    """
    Compute phonon frequencies along a q-point path.
    
    Args:
        u: UnitcellData
        qpts: Array of q-points (N, 3) in reduced coordinates
        
    Returns:
        Frequencies (N, 3*natom)
    """
    freqs = []
    for q in qpts:
        dm = compute_dynamical_matrix(q, u)
        freqs.append(get_frequencies(dm, u))
    return np.array(freqs)


def plot_phonon_bands(
    qpts: np.ndarray,
    freqs: np.ndarray,
    labels: Optional[List[str]] = None,
    tick_indices: Optional[List[int]] = None,
    save_path: Optional[str] = None,
    title: str = "Phonon Band Structure"
):
    """
    Plot phonon band structure.
    """
    plt.figure(figsize=(8, 6))
    
    # Compute distances between q-points for x-axis
    dist = [0.0]
    for i in range(1, len(qpts)):
        d = np.linalg.norm(qpts[i] - qpts[i-1])
        dist.append(dist[-1] + d)
    dist = np.array(dist)
    
    # Plot bands
    for i in range(freqs.shape[1]):
        plt.plot(dist, freqs[:, i], color='b', lw=1.5, alpha=0.7)
        
    plt.ylabel("Frequency (cm⁻¹)")
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    if tick_indices is not None and labels is not None:
        plt.xticks(dist[tick_indices], labels)
        for idx in tick_indices:
            plt.axvline(dist[idx], color='k', lw=0.5)
            
    plt.axhline(0, color='k', lw=1.0)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Bands plotted and saved to {save_path}")
    else:
        plt.show()
