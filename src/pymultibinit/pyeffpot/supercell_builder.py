"""
Supercell builder for effective potential.

This module provides functions to build supercell potentials from
unitcell data, including geometry replication, IFC mapping, and
dipole-dipole computation.

References:
- abinit/src/78_effpot/m_effective_potential.F90 (supercell generation)
- abinit/src/78_effpot/m_ifc.F90 (IFC handling)
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
    
    # Step 2: Replicate IFCs
    if unitcell.ifcs is None:
        raise ValueError("Unitcell IFCs not available")
    ifcs_sc = _replicate_ifcs(unitcell.ifcs, unitcell.crystal, crystal_sc, ncell)
    
    # Step 3: Compute dipole-dipole (if we have dielectric data)
    if unitcell.epsilon_inf is not None and unitcell.zeff is not None:
        _compute_dipole_dipole(ifcs_sc, unitcell, crystal_sc, ncell)
    
    # Step 4: Apply ASR
    _apply_asr(ifcs_sc)
    
    # Create supercell potential
    supercell = SupercellPotential(
        unitcell=unitcell,
        ncell=ncell,
        crystal_sc=crystal_sc,
        ifcs_sc=ifcs_sc,
        anharmonic_coeffs=None  # Set separately if needed
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
    
    # Supercell lattice vectors
    # rprimd_sc = diag(ncell) @ rprimd_uc
    rprimd_sc = np.diag(ncell) @ crystal_uc.rprimd
    
    # Generate atomic positions
    # For each unitcell in supercell, replicate atoms
    xred_sc = np.zeros((natom_sc, 3))
    typat_sc = np.zeros(natom_sc, dtype=int)
    
    idx = 0
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                for iat in range(natom_uc):
                    # Reduced coordinates: xred_uc + (ix, iy, iz) / ncell
                    xred_sc[idx, :] = crystal_uc.xred[iat, :] + np.array([ix, iy, iz])
                    typat_sc[idx] = crystal_uc.typat[iat]
                    idx += 1
    
    # Normalize to [0, 1)
    xred_sc[:, 0] /= nx
    xred_sc[:, 1] /= ny
    xred_sc[:, 2] /= nz
    
    # Convert to Cartesian
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
    
    Parameters
    ----------
    ifcs_uc : IFCData
        Unitcell IFCs
    crystal_uc : CrystalInfo
        Unitcell crystal structure
    crystal_sc : CrystalInfo
        Supercell crystal structure
    ncell : Tuple[int, int, int]
        Supercell dimensions
        
    Returns
    -------
    IFCData
        Supercell IFCs
    """
    if ifcs_uc is None:
        raise ValueError("Unitcell IFCs not available")
    
    nx, ny, nz = ncell
    natom_uc = crystal_uc.natom
    natom_sc = crystal_sc.natom
    nrpt_uc = ifcs_uc.nrpt
    
    # For now, keep same number of range points
    # In full implementation, we'd expand the range
    nrpt_sc = nrpt_uc
    
    # Initialize supercell IFCs
    atmfrc_sc = np.zeros((3, natom_sc, 3, natom_sc, nrpt_sc))
    short_atmfrc_sc = np.zeros((3, natom_sc, 3, natom_sc, nrpt_sc))
    cell_sc = ifcs_uc.cell.copy()
    
    # Map unitcell atoms to supercell atoms
    # Atom i_uc in cell (ix, iy, iz) -> atom i_sc = i_uc + natom_uc*(ix + nx*iy + nx*ny*iz)
    for irpt in range(nrpt_uc):
        cell_shift = ifcs_uc.cell[:, irpt] if ifcs_uc.cell.ndim == 2 else np.zeros(3, dtype=int)
        for i_uc in range(natom_uc):
            for j_uc in range(natom_uc):
                # Get IFC in unitcell
                ifc_uc = ifcs_uc.short_atmfrc[:, i_uc, :, j_uc, irpt]
                
                # Replicate to all cells in supercell
                for ix in range(nx):
                    for iy in range(ny):
                        for iz in range(nz):
                            # Supercell atom indices
                            i_sc = i_uc + natom_uc * (ix + nx * (iy + ny * iz))
                            jx = (ix + int(cell_shift[0])) % nx
                            jy = (iy + int(cell_shift[1])) % ny
                            jz = (iz + int(cell_shift[2])) % nz
                            j_sc = j_uc + natom_uc * (jx + nx * (jy + ny * jz))
                            
                            # Copy IFC (same relative position)
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
    Compute dipole-dipole (Ewald) contribution to IFCs.
    
    Computes dipole-dipole interactions directly on the supercell at Gamma point
    using Ewald summation. This is NOT a replication of unit-cell dipole-dipole.
    
    Parameters
    ----------
    ifcs_sc : IFCData
        Supercell IFCs (modified in place)
    unitcell : UnitcellData
        Unitcell with dielectric and Born charge data
    crystal_sc : CrystalInfo
        Supercell crystal structure
    ncell : Tuple[int, int, int]
        Supercell dimensions
    """
    natom_sc = crystal_sc.natom
    nrpt_sc = ifcs_sc.nrpt
    
    ifcs_sc.ewald_atmfrc = np.zeros((3, natom_sc, 3, natom_sc, nrpt_sc))
    
    # Only compute if we have dielectric and Born charge data
    if unitcell.epsilon_inf is None or unitcell.zeff is None:
        return
    
    # Check if Born charges are non-zero
    if np.linalg.norm(unitcell.zeff) < 1e-10:
        return
    
    # Build dipole-dipole IFCs for this supercell at Gamma
    # Use Ewald summation for proper long-range treatment
    try:
        dd_ifcs = build_dipole_dipole_ifcs(
            positions_cart=crystal_sc.xcart,
            epsilon_inf=unitcell.epsilon_inf,
            zeff=np.repeat(unitcell.zeff, np.prod(ncell), axis=2),
            lattice_vectors=crystal_sc.rprimd,
            use_ewald=True,
        )
    except Exception:
        # If Ewald fails (e.g., scipy not available), fall back to simple
        dd_ifcs = build_dipole_dipole_ifcs(
            positions_cart=crystal_sc.xcart,
            epsilon_inf=unitcell.epsilon_inf,
            zeff=np.repeat(unitcell.zeff, np.prod(ncell), axis=2),
            lattice_vectors=crystal_sc.rprimd,
            use_ewald=False,
        )
    
    ifcs_sc.ewald_atmfrc[:, :, :, :, 0] = dd_ifcs
    ifcs_sc.atmfrc += ifcs_sc.ewald_atmfrc


def _apply_asr(ifcs_sc: IFCData):
    """
    Apply Acoustic Sum Rule correction to IFCs.
    
    Ensures translational invariance: Σ_j Φ_ij = 0
    
    Parameters
    ----------
    ifcs_sc : IFCData
        Supercell IFCs (modified in place)
    """
    # ASR: for each atom i and direction mu, sum over j,nu,cell of Phi(mu,i,nu,j,cell) = 0
    # This ensures that translating all atoms together costs no energy
    
    natom = ifcs_sc.atmfrc.shape[1]
    
    for irpt in range(ifcs_sc.nrpt):
        for i in range(natom):
            for mu in range(3):
                # Sum over all j and nu
                sum_ifc = np.sum(ifcs_sc.atmfrc[mu, i, :, :, irpt])
                
                # Subtract average to enforce ASR
                # Distribute correction evenly
                correction = sum_ifc / (3 * natom)
                ifcs_sc.atmfrc[mu, i, :, :, irpt] -= correction
                ifcs_sc.short_atmfrc[mu, i, :, :, irpt] -= correction


def set_anharmonic_coeffs(supercell: SupercellPotential, coeffs: List):
    """
    Set anharmonic coefficients for supercell.
    
    Coefficients remain in unitcell basis and are applied during
    evaluation to supercell displacements.
    
    Parameters
    ----------
    supercell : SupercellPotential
        Supercell potential
    coeffs : List
        List of polynomial coefficients (from XML)
    """
    supercell.anharmonic_coeffs = coeffs
