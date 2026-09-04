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
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path

from .datastructures import SupercellPotential
from .supercell_builder import build_supercell, _supercell_atom_index
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

    def __init__(self, supercell: SupercellPotential, standalone_compat: bool = False,
                 standalone_ifc_file: Optional[str] = None):
        """Initialize effective potential."""
        self.supercell = supercell
        self._reference_positions = supercell.crystal_sc.xcart.copy()
        self._reference_lattice = supercell.crystal_sc.rprimd.copy()
        self._standalone_compat = standalone_compat
        self._standalone_frame_shifts: Optional[np.ndarray] = None

        self._phi_matrix: Optional[np.ndarray] = None
        self._phi_local: Optional[np.ndarray] = None
        self._phi_dipdip: Optional[np.ndarray] = None
        self._phonon_strain_matrices: Optional[List[Optional[np.ndarray]]] = None
        self._reference_stress: np.ndarray = np.zeros((3, 3), dtype=float)

        # Precompute harmonic force constant matrix (3*natom_sc, 3*natom_sc)
        natom = supercell.natom_sc
        if standalone_compat:
            if standalone_ifc_file is None:
                raise ValueError("standalone_ifc_file is required when standalone_compat=True")
            with np.load(standalone_ifc_file) as data:
                self._phi_matrix = np.array(data["phi_matrix"], dtype=float, copy=True)
            expected_shape = (3 * natom, 3 * natom)
            if self._phi_matrix.shape != expected_shape:
                raise ValueError(
                    f"standalone phi_matrix has shape {self._phi_matrix.shape}, expected {expected_shape}"
                )
        elif supercell.ifcs_sc is not None:
            ifcs = supercell.ifcs_sc
            # Local (short-range) harmonic IFC matrix.
            phi_local = ifcs.atmfrc.sum(axis=4).reshape(3 * natom, 3 * natom)
            # Dipole-dipole (Ewald long-range) harmonic IFC matrix.
            # When dipdip is disabled the builder still returns a zero-filled
            # ewald_atmfrc; treat that as "no dipole-dipole term" so the
            # decomposition does not report a vacuous zero contribution.
            phi_dipdip = None
            if ifcs.ewald_atmfrc is not None and np.any(ifcs.ewald_atmfrc):
                phi_dipdip = ifcs.ewald_atmfrc.sum(axis=4).reshape(3 * natom, 3 * natom)
                row_sums = phi_dipdip.sum(axis=1)
                for i in range(3 * natom):
                    phi_dipdip[i, i] -= row_sums[i]
            # Apply the acoustic-sum-rule diagonal correction to the local
            # part. When a dipdip part exists its correction was applied above;
            # the two corrections are linear and sum to the combined ASR matrix.
            row_sums = phi_local.sum(axis=1)
            for i in range(3 * natom):
                phi_local[i, i] -= row_sums[i]
            self._phi_local = phi_local
            self._phi_dipdip = phi_dipdip
            self._phi_matrix = phi_local + (phi_dipdip if phi_dipdip is not None else 0.0)
        else:
            self._phi_matrix = None

        # Precompute strain coupling matrices
        if hasattr(supercell, 'phonon_strain_sc') and supercell.phonon_strain_sc is not None:
            smats: List[Optional[np.ndarray]] = []
            for alpha in range(6):
                ifcs = supercell.phonon_strain_sc[alpha]
                if ifcs is None or ifcs.nrpt == 0:
                    smats.append(None)
                else:
                    p_sum = ifcs.atmfrc.sum(axis=4)
                    mat = p_sum.reshape(3*natom, 3*natom)
                    smats.append(mat)
            self._phonon_strain_matrices = smats

        self._anharmonic_compiled: Optional[List[Dict[str, Any]]] = None
        self._compile_anharmonic_terms()
        self._jax_compiled = None
        self._use_jax = False
        self.enable_jax()  # auto-enable GPU if available

    def enable_jax(self) -> bool:
        """Try to enable JAX GPU evaluation. Returns True if JAX is available."""
        if not self._anharmonic_compiled:
            return False
        try:
            from .jax_eval import compile_terms, _detect_backend
            if _detect_backend() == "none":
                return False
            n1, n2, n3 = self.supercell.ncell
            natom_uc = self.supercell.unitcell.crystal.natom
            self._jax_compiled = compile_terms(
                self.supercell.anharmonic_coeffs, (n1, n2, n3), natom_uc)
            self._use_jax = True
            return True
        except Exception:
            return False

    @classmethod
    def from_files(cls, ddb_file: str, xml_file: Optional[str] = None,
                   ncell: Tuple[int, int, int] = (4, 4, 4),
                    dipdip: bool = True,
                    asr: bool = True,
                    reference_stress: Optional[np.ndarray] = None,
                    standalone_compat: bool = False,
                    standalone_ifc_file: Optional[str] = None) -> EffectivePotential:
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
        reference_stress : np.ndarray, optional
            Equilibrium stress tensor (3, 3) in Ha/Bohr^3. Added to all
            stress predictions so the model reproduces absolute stress.

        Returns
        -------
        EffectivePotential
            Initialized potential evaluator.
        """
        unitcell = read_ddb(ddb_file)
        supercell = build_supercell(unitcell, ncell, dipdip=dipdip, asr=asr)

        if xml_file and Path(xml_file).exists():
            coeffs = read_coefficient_xml(xml_file)
            supercell.anharmonic_coeffs = coeffs

        pot = cls(supercell, standalone_compat=standalone_compat,
                  standalone_ifc_file=standalone_ifc_file)
        if reference_stress is not None:
            pot.set_reference_stress(reference_stress)
        return pot

    def set_reference_stress(self, stress: np.ndarray) -> None:
        """Set the equilibrium stress tensor (3,3) in Ha/Bohr^3."""
        s = np.asarray(stress, dtype=float)
        if s.shape == (6,):
            s = self._voigt_to_tensor(s)
        if s.shape != (3, 3):
            raise ValueError(f"reference_stress must be (3,3) or (6,), got {s.shape}")
        self._reference_stress = s.copy()

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
        displacements = self._compute_displacements(xcart, rprimd)

        # 3. Strain
        strain = self._compute_strain(rprimd)

        # 4. Harmonic energy
        if self._phi_matrix is not None:
            e_harm, f_harm, s_harm = self._evaluate_harmonic(displacements, strain, rprimd)
            energy += e_harm
            forces += f_harm
            stress += s_harm

        # 5. Elastic energy (if available)
        if self.supercell.unitcell.elastic_constants is not None:
            e_elast, f_elast, s_elast = self._evaluate_elastic(strain, displacements, rprimd)
            energy += e_elast
            forces += f_elast
            stress += s_elast

        # 6. Dipole-dipole is already included in harmonic IFCs

        # 7. Strain coupling (phonon-strain)
        if hasattr(self.supercell, 'phonon_strain_sc') and self.supercell.phonon_strain_sc is not None:
            e_sc, f_sc, s_sc = self._evaluate_strain_coupling(strain, displacements, rprimd)
            energy += e_sc
            forces += f_sc
            stress += s_sc

        # 8. Anharmonic (if available)
        if self.supercell.anharmonic_coeffs:
            e_anh, f_anh, s_anh = self._evaluate_anharmonic(displacements, strain, rprimd)
            energy += e_anh
            forces += f_anh
            stress += s_anh

        # 9. Reference stress (equilibrium pressure of parent structure)
        stress += self._reference_stress

        # 10. ABINIT effective_potential_distributeResidualForces
        # Mass-weighted projection ensuring total force is exactly zero.
        typat = self.supercell.crystal_sc.typat.astype(int)
        amu = self.supercell.unitcell.crystal.amu
        masses = amu[typat - 1]
        total_mass = masses.sum()
        total_force = forces.sum(axis=0)
        forces -= masses[:, np.newaxis] / total_mass * total_force[np.newaxis, :]

        return energy, forces, stress

    def evaluate_contributions(self, xcart: Optional[np.ndarray] = None,
                               rprimd: Optional[np.ndarray] = None) -> Dict[str, Tuple[float, np.ndarray, np.ndarray]]:
        """
        Evaluate energy, forces, and stress decomposed by physical term.

        The returned dict maps each active contribution name to a
        ``(energy, forces, stress)`` tuple in atomic units (Hartree,
        Hartree/Bohr, Hartree/Bohr^3; stress is a full (3, 3) tensor).

        Contributions
        --------------
        reference        : E_ref = ncells * E0 plus the equilibrium stress.
        harmonic_local   : local (short-range) harmonic IFCs.  [the "(b)" term]
        dipdip           : dipole-dipole (Ewald long-range) harmonic IFCs.  [the "(a)" term]
        anharmonic       : anharmonic XML coefficients.  [the "(c)" term]
        elastic          : homogeneous elastic constants (when present).
        strain_coupling  : phonon-strain coupling (when present).

        Only the contributions that are active for this potential appear as
        keys. The arrays summed over all keys reproduce ``evaluate()`` exactly
        (the mass-weighted residual-force projection is applied to each term,
        which is a linear operation, so the per-term results sum to the total).

        Parameters
        ----------
        xcart : np.ndarray, optional
            Atomic positions in Cartesian coordinates (Bohr).
            Defaults to the reference positions.
        rprimd : np.ndarray, optional
            Lattice vectors (Bohr). Defaults to the reference lattice.

        Returns
        -------
        dict
            ``{term_name: (energy, forces, stress)}``.
        """
        if xcart is None:
            xcart = self._reference_positions.copy()
        if rprimd is None:
            rprimd = self._reference_lattice.copy()

        natom = self.supercell.natom_sc
        displacements = self._compute_displacements(xcart, rprimd)
        strain = self._compute_strain(rprimd)

        contributions: Dict[str, Tuple[float, np.ndarray, np.ndarray]] = {}

        # Reference energy + equilibrium stress.
        e_ref = self._compute_reference_energy()
        contributions['reference'] = (
            e_ref, np.zeros((natom, 3)), self._reference_stress.copy()
        )

        # Harmonic: local (short-range) and dipole-dipole (Ewald), split.
        if self._phi_local is not None:
            contributions['harmonic_local'] = self._evaluate_harmonic_with_phi(
                self._phi_local, displacements, strain, rprimd)
        if self._phi_dipdip is not None:
            contributions['dipdip'] = self._evaluate_harmonic_with_phi(
                self._phi_dipdip, displacements, strain, rprimd)
        # Fallback when only a combined matrix is known (e.g. standalone mode).
        if 'harmonic_local' not in contributions and 'dipdip' not in contributions \
                and self._phi_matrix is not None:
            contributions['harmonic'] = self._evaluate_harmonic_with_phi(
                self._phi_matrix, displacements, strain, rprimd)

        # Elastic constants.
        if self.supercell.unitcell.elastic_constants is not None:
            contributions['elastic'] = self._evaluate_elastic(
                strain, displacements, rprimd)

        # Phonon-strain coupling.
        if getattr(self.supercell, 'phonon_strain_sc', None) is not None:
            contributions['strain_coupling'] = self._evaluate_strain_coupling(
                strain, displacements, rprimd)

        # Anharmonic coefficients.
        if self.supercell.anharmonic_coeffs:
            contributions['anharmonic'] = self._evaluate_anharmonic(
                displacements, strain, rprimd)

        # Apply the mass-weighted residual-force projection to every term.
        # Linear in the forces, so the terms still sum to evaluate().
        typat = self.supercell.crystal_sc.typat.astype(int)
        amu = self.supercell.unitcell.crystal.amu
        masses = amu[typat - 1]
        total_mass = masses.sum()
        corrected: Dict[str, Tuple[float, np.ndarray, np.ndarray]] = {}
        for name, (e, f, s) in contributions.items():
            term_total_force = f.sum(axis=0)
            f_corr = f - masses[:, np.newaxis] / total_mass * term_total_force[np.newaxis, :]
            corrected[name] = (e, f_corr, s)
        return corrected

    def _evaluate_strain_coupling(self, strain: np.ndarray,
                                  displacements: np.ndarray,
                                  rprimd: Optional[np.ndarray] = None) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Evaluate phonon-strain coupling contribution.

        Energy: E = (1/6) Σ_α ε_α Σ_{ia,μ,jb,ν,R} Φ_α(μ,ia,ν,jb,R) u_μ,ia u_ν,jb
        Forces: F_μ,ia = -(1/2) Σ_α ε_α Σ_{jb,ν,R} Φ_α(μ,ia,ν,jb,R) u_ν,jb
        Stress: σ_α = (1/2) Σ_{ia,μ,jb,ν,R} Φ_α(μ,ia,ν,jb,R) u_μ,ia u_ν,jb

        Parameters
        ----------
        strain : np.ndarray
            Strain tensor (3, 3).
        displacements : np.ndarray
            Atomic displacements (natom_sc, 3).

        Returns
        -------
        energy : float
        forces : np.ndarray
        stress : np.ndarray
        """
        energy = 0.0
        natom = self.supercell.natom_sc
        forces = np.zeros((natom, 3))
        stress = np.zeros((3, 3))

        if rprimd is None:
            rprimd = self._reference_lattice
        strain_voigt = self._strain_to_voigt(strain)

        u = displacements.flatten()
        stress_voigt = np.zeros(6)

        mats = self._phonon_strain_matrices
        if mats is None:
            return energy, forces, stress

        for alpha in range(6):
            mat = mats[alpha]
            if mat is None:
                continue

            # val = u^T @ Φ_α @ u
            val = u @ mat @ u

            # F_α = Φ_α @ u
            forces_flat = mat @ u

            energy += (1.0 / 6.0) * strain_voigt[alpha] * val
            forces -= (1.0 / 2.0) * strain_voigt[alpha] * forces_flat.reshape(natom, 3)
            stress_voigt[alpha] += (1.0 / 2.0) * val

        stress = self._finalize_stress(stress_voigt, forces, displacements, strain, rprimd)

        return energy, forces, stress

    def _compute_reference_energy(self) -> float:
        """Compute reference energy: E_ref = ncells * E0."""
        ncells = self.supercell.ncells
        e0 = self.supercell.unitcell.energy
        return ncells * e0

    def _compute_displacements(self, xcart: np.ndarray, rprimd: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute atomic displacements from reference, wrapped to the minimum image.

        Parameters
        ----------
        xcart : np.ndarray
            Current atomic positions (Bohr).
        rprimd : np.ndarray, optional
            Current lattice vectors (Bohr). Defaults to the reference lattice.

        Returns
        -------
        np.ndarray
            Minimum-image displacements ``u`` in the current cell, shape (natom, 3), Bohr.
            Atoms that have drifted across a cell boundary (fractional delta
            near ±1) wrap to the equivalent small displacement (near 0).
        """
        if rprimd is None:
            rprimd = self._reference_lattice
        if self._standalone_compat:
            current_xred = xcart @ np.linalg.inv(rprimd).T
            reference_xred = self.supercell.crystal_sc.xred
            if self._standalone_frame_shifts is None:
                delta = current_xred - reference_xred
                self._standalone_frame_shifts = np.where(
                    delta > 0.5, -1.0, np.where(delta < -0.5, 1.0, 0.0)
                )
            return (current_xred + self._standalone_frame_shifts - reference_xred) @ rprimd.T
        reference_xred = self.supercell.crystal_sc.xred
        reference_positions_in_current_cell = reference_xred @ rprimd.T
        raw_disp = xcart - reference_positions_in_current_cell
        lat_inv = np.linalg.inv(rprimd)
        dr_frac = raw_disp @ lat_inv.T
        dr_frac_wrapped = dr_frac - np.round(dr_frac)
        return dr_frac_wrapped @ rprimd.T

    def _compute_strain(self, rprimd: np.ndarray) -> np.ndarray:
        """
        Compute engineering (Biot-like) strain matching the MULTIBINIT/Fortran convention.

        Fortran reference: strain_get() in m_strain.F90 via fit_data_compute in m_fit_data.F90.
        Formula: eta = h_def @ h_ref^{-T} - I
        where h_def = rprimd (deformed), h_ref = reference lattice.

        Parameters
        ----------
        rprimd : np.ndarray
            Current lattice vectors (Bohr).

        Returns
        -------
        np.ndarray
            Engineering strain tensor eta = h_def @ h_ref^{-T} - I.
        """
        # Deformation gradient relative to reference: h = rprimd @ inv(rprimd_ref)^T
        # Fortran: mat_delta = matmul(rprim_def, transpose(rprim_inv)) - identity
        # where rprim_inv = inv(rprimd_ref)
        h = rprimd @ np.linalg.inv(self._reference_lattice).T
        strain = h - np.eye(3)
        return strain

    def _compute_du_delta(self, displacements: np.ndarray, strain: np.ndarray) -> np.ndarray:
        """Compute MULTIBINIT displacement derivatives with respect to Voigt strain."""
        strain_inv = np.linalg.inv(np.eye(3) + strain)
        # strain_inv @ disp is the same transform for every atom -> one matmul
        # over all atoms, then a fixed voigt gather (no per-atom loop needed).
        strain_inv_u = displacements @ strain_inv.T                # (natom, 3)
        pairs = ((0, 0), (1, 1), (2, 2), (2, 1), (2, 0), (1, 0))
        du_delta = np.zeros((6, displacements.shape[0], 3), dtype=float)
        for ivoigt, (alpha, beta) in enumerate(pairs):
            for mu in range(3):
                if alpha == mu:
                    du_delta[ivoigt, :, mu] += 0.5 * strain_inv_u[:, beta]
                if beta == mu:
                    du_delta[ivoigt, :, mu] += 0.5 * strain_inv_u[:, alpha]
        return du_delta

    @staticmethod
    def _strain_to_voigt(strain: np.ndarray) -> np.ndarray:
        return np.array([
            strain[0, 0],
            strain[1, 1],
            strain[2, 2],
            strain[1, 2] + strain[2, 1],
            strain[2, 0] + strain[0, 2],
            strain[0, 1] + strain[1, 0],
        ], dtype=float)

    @staticmethod
    def _voigt_to_tensor(voigt: np.ndarray) -> np.ndarray:
        return np.array([
            [voigt[0], voigt[5], voigt[4]],
            [voigt[5], voigt[1], voigt[3]],
            [voigt[4], voigt[3], voigt[2]],
        ], dtype=float)

    def _finalize_stress(self, stress_voigt: np.ndarray, forces: np.ndarray, displacements: np.ndarray, strain: np.ndarray, rprimd: np.ndarray) -> np.ndarray:
        """Apply MULTIBINIT's du/deta correction and strain-volume scaling."""
        du_delta = self._compute_du_delta(displacements, strain)
        corrected = np.array(stress_voigt, dtype=float, copy=True)
        corrected -= np.einsum("vna,na->v", du_delta, forces)
        strain_voigt = self._strain_to_voigt(strain)
        ucvol = abs(np.linalg.det(rprimd))
        corrected[:3] *= (1.0 + strain_voigt[:3]) / ucvol
        corrected[3:] *= (1.0 - strain_voigt[3:] ** 2) / ucvol
        return self._voigt_to_tensor(corrected)

    def _evaluate_harmonic_with_phi(self, phi_matrix: Optional[np.ndarray],
                                    displacements: np.ndarray,
                                    strain: Optional[np.ndarray] = None,
                                    rprimd: Optional[np.ndarray] = None) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Evaluate the harmonic IFC contribution for an explicit force-constant
        matrix (3*natom, 3*natom).

        Used to split the harmonic term into its local (short-range) and
        dipole-dipole (Ewald) parts by passing the corresponding phi matrix.

        Energy: E = ½ u^T Φ u, Forces: F = -Φ u.
        """
        natom = self.supercell.natom_sc
        if strain is None:
            strain = np.zeros((3, 3), dtype=float)
        if rprimd is None:
            rprimd = self._reference_lattice
        if phi_matrix is None:
            return 0.0, np.zeros((natom, 3)), np.zeros((3, 3))

        u = displacements.flatten()
        energy = 0.5 * u @ phi_matrix @ u
        forces = (-phi_matrix @ u).reshape(natom, 3)
        stress = self._finalize_stress(np.zeros(6, dtype=float), forces, displacements, strain, rprimd)
        return energy, forces, stress

    def _evaluate_harmonic(self, displacements: np.ndarray,
                           strain: Optional[np.ndarray] = None,
                           rprimd: Optional[np.ndarray] = None) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Evaluate the combined harmonic IFC contribution (local + dipole-dipole).

        Energy: E = ½ u^T Φ u, Forces: F = -Φ u,
        Stress: σ = (1/V) u^T ∂Φ/∂ε u.
        """
        return self._evaluate_harmonic_with_phi(self._phi_matrix, displacements, strain, rprimd)

    def _evaluate_elastic(self, strain: np.ndarray,
                          displacements: np.ndarray,
                          rprimd: Optional[np.ndarray] = None) -> Tuple[float, np.ndarray, np.ndarray]:
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
            Elastic and internal-strain forces.
        stress : np.ndarray
            Elastic stress (Hartree/Bohr^3).
        """
        natom = self.supercell.natom_sc
        if rprimd is None:
            rprimd = self._reference_lattice
        strain_voigt = self._strain_to_voigt(strain)

        # Fortran stores elastic constants as unit-cell energy derivatives.
        C = self.supercell.unitcell.elastic_constants
        energy = 0.5 * self.supercell.ncells * strain_voigt @ C @ strain_voigt
        forces = np.zeros((natom, 3))
        stress_voigt = self.supercell.ncells * (C @ strain_voigt)

        coupling = getattr(self.supercell.unitcell, "strain_coupling", None)
        if coupling is not None:
            coupling = np.asarray(coupling, dtype=float)
            if coupling.shape == (6, 3, self.supercell.unitcell.natom):
                # Vectorized replacement for the per-atom triple loop:
                #   for n in range(natom):
                #       iuc = n % natom_uc
                #       for alpha in range(6):
                #           for mu in range(3):
                #               value = coupling[alpha, mu, iuc]
                #               energy   += 0.5 * value * sv[alpha] * disp[n, mu]
                #               forces[n, mu] -= 0.5 * value * sv[alpha]
                #               stress[alpha]  += 0.5 * value * disp[n, mu]
                # c_sc[n, alpha, mu] = coupling[alpha, mu, n % natom_uc] broadcasts the
                # unit-cell coupling to every supercell atom.
                iuc_all = np.arange(natom) % self.supercell.unitcell.natom
                c_sc = coupling[:, :, iuc_all].transpose(2, 0, 1)
                energy += 0.5 * np.einsum("nam,nm,a->", c_sc, displacements, strain_voigt)
                forces -= 0.5 * np.einsum("nam,a->nm", c_sc, strain_voigt)
                stress_voigt += 0.5 * np.einsum("nam,nm->a", c_sc, displacements)

        stress = self._finalize_stress(stress_voigt, forces, displacements, strain, rprimd)
        return energy, forces, stress

    def _compile_anharmonic_terms(self):
        """Precompute indices and shapes for fast vectorized anharmonic evaluation."""
        if not hasattr(self.supercell, "anharmonic_coeffs") or not self.supercell.anharmonic_coeffs:
            return

        n1, n2, n3 = self.supercell.ncell
        natom_uc = self.supercell.unitcell.crystal.natom
        nx, ny, nz = int(n1), int(n2), int(n3)

        # Pre-compute the (atom_a, cell_a) mappings to supercell indices
        def get_sc_indices(atom_uc, cell):
            # cell is [ix, iy, iz] relative shifts
            idx = np.zeros(nx * ny * nz, dtype=int)
            count = 0
            for ix0 in range(nx):
                for iy0 in range(ny):
                    for iz0 in range(nz):
                        ix = (ix0 + int(cell[0])) % nx
                        iy = (iy0 + int(cell[1])) % ny
                        iz = (iz0 + int(cell[2])) % nz
                        i_sc = _supercell_atom_index(atom_uc, ix, iy, iz, self.supercell.ncell, natom_uc)
                        idx[count] = i_sc
                        count += 1
            return idx

        compiled_terms = []
        for coeff in self.supercell.anharmonic_coeffs:
            for term in coeff.terms:
                compiled_disp = []
                for disp in term.displacements:
                    idx_a = get_sc_indices(disp['atom_a'], disp['cell_a'])
                    idx_b = get_sc_indices(disp['atom_b'], disp['cell_b'])

                    direction_map = {'x': 0, 'y': 1, 'z': 2}
                    dir_idx = direction_map[disp['direction']]

                    compiled_disp.append({
                        'idx_a': idx_a,
                        'idx_b': idx_b,
                        'dir': dir_idx,
                        'power': disp['power']
                    })

                compiled_terms.append({
                    'value': coeff.value,
                    'weight': term.weight,
                    'displacements': compiled_disp,
                    'strains': term.strains
                })

        self._anharmonic_compiled = compiled_terms

    def _evaluate_anharmonic(self, displacements: np.ndarray, strain: np.ndarray, rprimd: Optional[np.ndarray] = None) -> Tuple[float, np.ndarray, np.ndarray]:
        natom = self.supercell.natom_sc
        compiled_terms = self._anharmonic_compiled
        if not compiled_terms:
            return 0.0, np.zeros((natom, 3)), np.zeros((3, 3))

        if rprimd is None:
            rprimd = self._reference_lattice
        strain_voigt = self._strain_to_voigt(strain)

        if self._use_jax and self._jax_compiled is not None:
            from .jax_eval import evaluate_jax
            energy, forces, stress_voigt = evaluate_jax(
                self._jax_compiled, displacements, strain_voigt)
            stress = self._finalize_stress(stress_voigt, forces, displacements, strain, rprimd)
            return energy, forces, stress

        forces = np.zeros((natom, 3), dtype=float)
        stress_voigt = np.zeros(6, dtype=float)
        energy = 0.0
        ncells = np.prod(self.supercell.ncell)

        for term_info in compiled_terms:
            base_coeff = term_info['value'] * term_info['weight']

            # Strain multiplier for the whole term
            strain_val = 1.0
            for st in term_info['strains']:
                strain_val *= strain_voigt[st['voigt']-1] ** st['power']

            prod_disp = np.ones(ncells, dtype=float)
            for disp in term_info['displacements']:
                if disp['power'] == 0:
                    continue
                diff = displacements[disp['idx_a'], disp['dir']] - displacements[disp['idx_b'], disp['dir']]
                prod_disp *= diff ** disp['power']

            term_energy_array = base_coeff * strain_val * prod_disp
            energy += float(term_energy_array.sum())

            # Compute derivative for each displacement factor
            for k, disp_k in enumerate(term_info['displacements']):
                p_k = disp_k['power']
                if p_k == 0:
                    continue

                deriv = np.ones(ncells, dtype=float) * (base_coeff * strain_val * p_k)

                for j, disp_j in enumerate(term_info['displacements']):
                    p_j = disp_j['power']
                    if p_j == 0:
                        continue
                    diff_j = displacements[disp_j['idx_a'], disp_j['dir']] - displacements[disp_j['idx_b'], disp_j['dir']]
                    if j == k:
                        deriv *= diff_j ** (p_j - 1)
                    else:
                        deriv *= diff_j ** p_j

                np.add.at(forces[:, disp_k['dir']], disp_k['idx_a'], -deriv)
                np.add.at(forces[:, disp_k['dir']], disp_k['idx_b'], deriv)

            # Compute derivative for each strain factor
            for k, st_k in enumerate(term_info['strains']):
                p_k = st_k['power']
                if p_k == 0:
                    continue

                # Use precomputed prod_disp and strain_val
                # dE / ds_alpha_k = base_coeff * p_k * s_alpha_k^(p_k-1) * other_strains * prod_disp
                # This is equal to (energy_of_term / strain_voigt_k) * p_k

                v_idx = st_k['voigt'] - 1
                s_val = strain_voigt[v_idx]

                if abs(s_val) > 1e-12:
                    deriv_s = term_energy_array * p_k / s_val
                else:
                    # Handle s_val = 0: only non-zero if p_k = 1
                    if p_k == 1:
                        # factor is base_coeff * other_strains * prod_disp
                        deriv_s = base_coeff * prod_disp
                        for j, st_j in enumerate(term_info['strains']):
                            if j != k:
                                deriv_s *= strain_voigt[st_j['voigt']-1] ** st_j['power']
                    else:
                        deriv_s = np.zeros_like(term_energy_array)

                stress_voigt[v_idx] += float(deriv_s.sum())

        stress = self._finalize_stress(stress_voigt, forces, displacements, strain, rprimd)

        return energy, forces, stress

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
