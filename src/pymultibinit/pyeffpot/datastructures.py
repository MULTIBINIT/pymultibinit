"""
Data structures for pure Python effective potential implementation.

This module defines the core data structures for unitcell and supercell
potential data, independent of file I/O.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class CrystalInfo:
    """Crystal structure information."""
    natom: int
    ntypat: int
    rprimd: np.ndarray  # (3, 3) lattice vectors in Bohr
    xred: np.ndarray    # (natom, 3) reduced coordinates
    xcart: np.ndarray   # (natom, 3) Cartesian coordinates in Bohr
    typat: np.ndarray   # (natom,) atom types
    amu: np.ndarray     # (ntypat,) atomic masses in atomic mass units
    znucl: np.ndarray   # (ntypat,) atomic numbers
    

@dataclass
class IFCData:
    """Interatomic force constants data."""
    nrpt: int                        # Number of range points
    cell: np.ndarray                 # (3, nrpt) cell indices for each range point
    atmfrc: np.ndarray               # (3, natom, 3, natom, nrpt) total IFCs
    short_atmfrc: np.ndarray         # (3, natom, 3, natom, nrpt) short-range IFCs
    ewald_atmfrc: Optional[np.ndarray] = None  # (3, natom, 3, natom, nrpt) Ewald IFCs


@dataclass  
class UnitcellData:
    """Primitive cell data from files (DDB/XML)."""
    crystal: CrystalInfo
    energy: float                    # Reference energy in Hartree
    
    # Harmonic terms
    ifcs: Optional[IFCData] = None  # Interatomic force constants
    
    # Dielectric and Born charges
    epsilon_inf: Optional[np.ndarray] = None  # (3, 3) dielectric tensor
    zeff: Optional[np.ndarray] = None         # (3, 3, natom) Born effective charges
    
    # Elastic constants
    elastic_constants: Optional[np.ndarray] = None  # (6, 6) elastic tensor
    
    # Strain-phonon coupling (optional)
    strain_coupling: Optional[np.ndarray] = None  # (6, 3, natom) if present
    
    @property
    def natom(self) -> int:
        return self.crystal.natom
    
    @property
    def ntypat(self) -> int:
        return self.crystal.ntypat


@dataclass
class SupercellPotential:
    """Supercell potential ready for evaluation."""
    unitcell: UnitcellData           # Reference to unitcell data
    ncell: Tuple[int, int, int]      # Supercell dimensions (nx, ny, nz)
    
    # Supercell crystal info
    crystal_sc: CrystalInfo          # Supercell crystal structure
    
    # IFCs in supercell (including dipole-dipole)
    ifcs_sc: IFCData                 # Supercell IFCs
    
    # Anharmonic coefficients (from XML, in unitcell basis)
    # These will be applied during evaluation
    anharmonic_coeffs: Optional[List] = None
    
    @property
    def natom_sc(self) -> int:
        """Number of atoms in supercell."""
        return self.crystal_sc.natom
    
    @property
    def ncells(self) -> int:
        """Total number of unit cells in supercell."""
        return self.ncell[0] * self.ncell[1] * self.ncell[2]
    
    @classmethod
    def from_unitcell(cls, unitcell: UnitcellData, ncell: Tuple[int, int, int]):
        """
        Build supercell potential from unitcell.
        
        Parameters
        ----------
        unitcell : UnitcellData
            Primitive cell data
        ncell : Tuple[int, int, int]
            Supercell dimensions (nx, ny, nz)
            
        Returns
        -------
        SupercellPotential
            Supercell ready for evaluation
        """
        # This will be implemented in supercell_builder.py
        raise NotImplementedError("Use build_supercell() function")
