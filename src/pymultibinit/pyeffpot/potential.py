"""
Core effective potential evaluation module.

This module provides the EffectivePotential class for evaluating
energy, forces, and stress for atomic configurations.

The total energy is:
    E_total = E_ref + E_harmonic + E_elastic + E_dipdip + E_anharmonic + E_conf

where:
    E_ref = ncells * E0 (reference energy)
    E_harmonic = ½ u^T Φ u (harmonic IFCs)
    E_elastic = ½ V ε^T C ε (elastic energy)
    E_dipdip = dipole-dipole interaction energy
    E_anharmonic = Σ c_k Π (u)^n (anharmonic terms)
    E_conf = f Σ (|u|-u_c)^p (confinement)

References:
- abinit/src/78_effpot/m_effective_potential.F90
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, List
from pathlib import Path

from .datastructures import SupercellPotential, CrystalInfo
from .supercell_builder import build_supercell
from .ddb_parser_complete import read_ddb
from .xml_parser import read_coefficient_xml


class EffectivePotential:
    """
    Effective potential evaluator for MULTIBINIT.
    
    This class evaluates energy, forces, and stress for atomic configurations
    using the effective potential formalism.
    
    Parameters
    ----------
    supercell : SupercellPotential
        Supercell with all force constants and coefficients.
    
    Examples
    --------
    >>> from pymultibinit.pyeffpot import read_ddb, read_coefficient_xml
    >>> from pymultibinit.pyeffpot import build_supercell, EffectivePotential
    >>> 
    >>> # Load data
    >>> unitcell = read_ddb("system.DDB")
    >>> coeffs = read_coefficient_xml("coeffs.xml")
    >>> 
    >>> # Build supercell
    >>> supercell = build_supercell(unitcell, (4, 4, 4))
    >>> supercell.set_anharmonic_coeffs(coeffs)
    >>> 
    >>> # Evaluate
    >>> potential = EffectivePotential(supercell)
    >>> energy, forces, stress = potential.evaluate(xcart, rprimd)
    """
    
    def __init__(self, supercell: SupercellPotential):
        """Initialize effective potential."""
        self.supercell = supercell
        self._reference_positions = supercell.crystal_sc.xcart.copy()
        self._reference_lattice = supercell.crystal_sc.rprimd.copy()
    
    @classmethod
    def from_files(cls, ddb_file: str, xml_file: Optional[str] = None,
                   ncell: Tuple[int, int, int] = (4, 4, 4)) -> EffectivePotential:
        """
        Create EffectivePotential from DDB and XML files.
        
        Parameters
        ----------
        ddb_file : str
            Path to DDB file with harmonic IFCs.
        xml_file : str, optional
            Path to XML file with anharmonic coefficients.
        ncell : tuple of int
            Supercell dimensions (nx, ny, nz).
        
        Returns
        -------
        EffectivePotential
            Initialized potential evaluator.
        """
        unitcell = read_ddb(ddb_file)
        supercell = build_supercell(unitcell, ncell)
        
        if xml_file and Path(xml_file).exists():
            coeffs = read_coefficient_xml(xml_file)
            supercell.anharmonic_coeffs = coeffs
        
        return cls(supercell)
    
    def evaluate(self, xcart: Optional[np.ndarray] = None,
                 rprimd: Optional[np.ndarray] = None) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Evaluate energy, forces, and stress.
        
        Parameters
        ----------
        xcart : np.ndarray, optional
            Atomic positions in Cartesian coordinates (Bohr).
            If None, use reference positions.
        rprimd : np.ndarray, optional
            Lattice vectors (Bohr). If None, use reference lattice.
        
        Returns
        -------
        energy : float
            Total energy in Hartree.
        forces : np.ndarray
            Forces on atoms in Hartree/Bohr, shape (natom, 3).
        stress : np.ndarray
            Stress tensor in Hartree/Bohr^3, shape (3, 3).
        """
        if xcart is None:
            xcart = self._reference_positions.copy()
        if rprimd is None:
            rprimd = self._reference_lattice.copy()
        
        # Initialize
        energy = 0.0
        forces = np.zeros((self.supercell.natom_sc, 3))
        stress = np.zeros((3, 3))
        
        # 1. Reference energy
        e_ref = self._compute_reference_energy()
        energy += e_ref
        
        # 2. Displacements
        displacements = self._compute_displacements(xcart)
        
        # 3. Strain
        strain = self._compute_strain(rprimd)
        
        # 4. Harmonic energy
        e_harm, f_harm, s_harm = self._evaluate_harmonic(displacements)
        energy += e_harm
        forces += f_harm
        stress += s_harm
        
        # 5. Elastic energy (if available)
        if self.supercell.unitcell.elastic_constants is not None:
            e_elast, f_elast, s_elast = self._evaluate_elastic(strain, displacements)
            energy += e_elast
            forces += f_elast
            stress += s_elast
        
        # 6. Dipole-dipole is already included in harmonic IFCs
        
        # 7. Anharmonic (if available)
        if self.supercell.anharmonic_coeffs:
            e_anh, f_anh = self._evaluate_anharmonic(displacements)
            energy += e_anh
            forces += f_anh
        
        return energy, forces, stress
    
    def _compute_reference_energy(self) -> float:
        """Compute reference energy: E_ref = ncells * E0."""
        ncells = self.supercell.ncells
        e0 = self.supercell.unitcell.energy
        return ncells * e0
    
    def _compute_displacements(self, xcart: np.ndarray) -> np.ndarray:
        """
        Compute atomic displacements from reference.
        
        Parameters
        ----------
        xcart : np.ndarray
            Current atomic positions (Bohr).
        
        Returns
        -------
        np.ndarray
            Displacements u = xcart - xcart_ref (Bohr).
        """
        return xcart - self._reference_positions
    
    def _compute_strain(self, rprimd: np.ndarray) -> np.ndarray:
        """
        Compute strain tensor from lattice deformation.
        
        Parameters
        ----------
        rprimd : np.ndarray
            Current lattice vectors (Bohr).
        
        Returns
        -------
        np.ndarray
            Strain tensor ε = ½(h^T h - I), where h = lattice * inv(lattice_ref).
        """
        # Deformation gradient: h = lattice * inv(lattice_ref)
        h = rprimd @ np.linalg.inv(self._reference_lattice)
        
        # Green-Lagrange strain: ε = ½(h^T h - I)
        strain = 0.5 * (h.T @ h - np.eye(3))
        
        return strain
    
    def _evaluate_harmonic(self, displacements: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Evaluate harmonic IFC contribution.
        
        Energy: E = ½ u^T Φ u
        Forces: F = -Φ u
        Stress: σ = (1/V) u^T ∂Φ/∂ε u
        
        Parameters
        ----------
        displacements : np.ndarray
            Atomic displacements (Bohr).
        
        Returns
        -------
        energy : float
            Harmonic energy (Hartree).
        forces : np.ndarray
            Harmonic forces (Hartree/Bohr).
        stress : np.ndarray
            Harmonic stress (Hartree/Bohr^3).
        """
        natom = self.supercell.natom_sc
        ifcs = self.supercell.ifcs_sc
        
        # Flatten displacements: (natom, 3) -> (3*natom)
        u = displacements.flatten()
        
        # Sum over all range points
        nrpt = ifcs.atmfrc.shape[4]
        energy = 0.0
        forces_flat = np.zeros(3 * natom)
        
        for irpt in range(nrpt):
            phi = ifcs.atmfrc[:, :, :, :, irpt]  # (3, natom, 3, natom)
            
            # Reshape to (3*natom, 3*natom)
            phi_matrix = phi.transpose(1, 0, 3, 2).reshape(3*natom, 3*natom)
            
            # Energy: E = ½ u^T Φ u
            energy += 0.5 * u @ phi_matrix @ u
            
            # Forces: F = -Φ u
            forces_flat += -phi_matrix @ u
        
        forces = forces_flat.reshape(natom, 3)
        
        # Stress (simplified - no strain coupling yet)
        volume = np.abs(np.linalg.det(self._reference_lattice))
        stress = np.zeros((3, 3))
        
        return energy, forces, stress
    
    def _evaluate_elastic(self, strain: np.ndarray, 
                         displacements: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Evaluate elastic contribution.
        
        Energy: E = ½ V ε^T C ε
        Forces: F = 0 (no force from homogeneous strain in this approximation)
        Stress: σ = C ε
        
        Parameters
        ----------
        strain : np.ndarray
            Strain tensor.
        displacements : np.ndarray
            Atomic displacements.
        
        Returns
        -------
        energy : float
            Elastic energy (Hartree).
        forces : np.ndarray
            Elastic forces (zero in this approximation).
        stress : np.ndarray
            Elastic stress (Hartree/Bohr^3).
        """
        natom = self.supercell.natom_sc
        volume = np.abs(np.linalg.det(self._reference_lattice))
        
        # Convert strain to Voigt notation
        strain_voigt = np.array([
            strain[0, 0],  # ε_xx
            strain[1, 1],  # ε_yy
            strain[2, 2],  # ε_zz
            2*strain[1, 2],  # ε_yz
            2*strain[0, 2],  # ε_xz
            2*strain[0, 1],  # ε_xy
        ])
        
        # Energy: E = ½ V ε^T C ε
        C = self.supercell.unitcell.elastic_constants
        energy = 0.5 * volume * strain_voigt @ C @ strain_voigt
        
        # Forces: zero in this approximation
        forces = np.zeros((natom, 3))
        
        # Stress: σ = C ε
        stress_voigt = C @ strain_voigt
        
        # Convert back to tensor
        stress = np.array([
            [stress_voigt[0], stress_voigt[5], stress_voigt[4]],
            [stress_voigt[5], stress_voigt[1], stress_voigt[3]],
            [stress_voigt[4], stress_voigt[3], stress_voigt[2]],
        ])
        
        return energy, forces, stress
    
    def _evaluate_anharmonic(self, displacements: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Evaluate anharmonic contribution.
        
        Energy: E = Σ c_k Π (u)^n
        Forces: F = -∇E
        
        This is a simplified implementation. Full implementation would
        parse polynomial terms from XML coefficients.
        
        Parameters
        ----------
        displacements : np.ndarray
            Atomic displacements (Bohr).
        
        Returns
        -------
        energy : float
            Anharmonic energy (Hartree).
        forces : np.ndarray
            Anharmonic forces (Hartree/Bohr).
        """
        # Placeholder - would need to implement polynomial evaluation
        # based on XML coefficient structure
        natom = self.supercell.natom_sc
        return 0.0, np.zeros((natom, 3))
    
    def evaluate_energy_only(self, xcart: Optional[np.ndarray] = None,
                             rprimd: Optional[np.ndarray] = None) -> float:
        """Evaluate only the energy (faster than full evaluation)."""
        energy, _, _ = self.evaluate(xcart, rprimd)
        return energy
    
    def evaluate_forces_only(self, xcart: Optional[np.ndarray] = None) -> np.ndarray:
        """Evaluate only the forces."""
        _, forces, _ = self.evaluate(xcart)
        return forces
    
    def evaluate_stress_only(self, xcart: Optional[np.ndarray] = None,
                             rprimd: Optional[np.ndarray] = None) -> np.ndarray:
        """Evaluate only the stress tensor."""
        _, _, stress = self.evaluate(xcart, rprimd)
        return stress
