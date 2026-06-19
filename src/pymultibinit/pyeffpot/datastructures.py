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
    atmfrc: np.ndarray               # (natom, 3, natom, 3, nrpt) total IFCs
    short_atmfrc: np.ndarray         # (natom, 3, natom, 3, nrpt) short-range IFCs
    ewald_atmfrc: Optional[np.ndarray] = None  # (natom, 3, natom, 3, nrpt) Ewald IFCs
    wghatm: Optional[np.ndarray] = None        # (natom, natom, nrpt) Wigner-Seitz weights


@dataclass  
class UnitcellData:
    """Primitive cell data from files (DDB/XML)."""
    crystal: CrystalInfo
    energy: float                    # Reference energy in Hartree
    
    # Harmonic terms
    ifcs: Optional[IFCData] = None  # Interatomic force constants
    
    # Dielectric and Born charges
    epsilon_inf: Optional[np.ndarray] = None  # (3, 3) dielectric tensor
    zeff: Optional[np.ndarray] = None         # (natom, 3, 3) Born effective charges
    
    # Elastic constants
    elastic_constants: Optional[np.ndarray] = None  # (6, 6) elastic tensor
    
    # Strain-phonon coupling (3rd order: ∂³E/∂u∂u∂η)
    # List of 6 IFCData objects, one for each strain direction
    phonon_strain: Optional[List[IFCData]] = None
    
    # Elastic-displacement coupling (3rd order: ∂³E/∂η∂η∂u)
    elastic_displacement: Optional[np.ndarray] = None  # (6, 6, 3, natom)
    
    # Higher order elastic (3rd and 4th)
    elastic3rd: Optional[np.ndarray] = None  # (6, 6, 6)
    elastic4th: Optional[np.ndarray] = None  # (6, 6, 6, 6)
    
    # Strain-displacement coupling (3rd order: ∂³E/∂η∂u∂u)
    # (Same as phonon_strain)
    
    # Internal strain coupling (2nd order: ∂²E/∂η∂u)
    strain_coupling: Optional[np.ndarray] = None  # (6, 3, natom) if present

    # Optional DDB-specific metadata
    acell: Optional[np.ndarray] = None            # (3,) lattice parameters in Bohr
    qpoints: Optional[np.ndarray] = None          # (nqpt, 3) irreducible q-points from DDB
    dynmat: Optional[np.ndarray] = None           # (nqpt, natom, 3, natom, 3, 2)
    blocks: Optional[List] = None                 # Raw parsed DDB blocks
    ngqpt: Optional[np.ndarray] = None            # (3,) q-point grid dimensions
    symrel: Optional[np.ndarray] = None           # (nsym, 3, 3) symmetry operations (integer)
    tnons: Optional[np.ndarray] = None            # (nsym, 3) fractional symmetry translations
    nqshft: int = 1                               # Number of q-grid shifts
    q1shft: Optional[np.ndarray] = None           # (nqshft, 3) q-grid shifts
    atom_mapping: Optional[np.ndarray] = None      # (natom, nsym, 4) if computed
    
    @property
    def natom(self) -> int:
        return self.crystal.natom
    
    @property
    def ntypat(self) -> int:
        return self.crystal.ntypat

    @property
    def rprimd(self) -> np.ndarray:
        return self.crystal.rprimd

    @property
    def xred(self) -> np.ndarray:
        return self.crystal.xred

    @property
    def xcart(self) -> np.ndarray:
        return self.crystal.xcart

    @property
    def typat(self) -> np.ndarray:
        return self.crystal.typat

    @property
    def amu(self) -> np.ndarray:
        return self.crystal.amu

    @property
    def znucl(self) -> np.ndarray:
        return self.crystal.znucl

    @property
    def nqpt(self) -> int:
        if self.qpoints is None:
            return 0
        return len(self.qpoints)


@dataclass
class SupercellPotential:
    """Supercell potential ready for evaluation."""
    unitcell: UnitcellData           # Reference to unitcell data
    ncell: Tuple[int, int, int]      # Supercell dimensions (nx, ny, nz)
    
    # Supercell crystal info
    crystal_sc: CrystalInfo          # Supercell crystal structure
    
    # IFCs in supercell (including dipole-dipole)
    ifcs_sc: IFCData                 # Supercell IFCs
    
    # Strain-phonon coupling in supercell (List of 6 IFCData)
    phonon_strain_sc: Optional[List[IFCData]] = None
    
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
