"""
High-level Python API for MULTIBINIT effective potential.

This module provides user-friendly classes that wrap the C API and handle
unit conversions automatically.

Supports two backends:
- 'cffi': CFFI wrapper for Fortran library (default, optimal performance)
- 'pyeffpot': Pure Python implementation (no Fortran dependency)
"""
import numpy as np
from typing import Optional, Tuple, Literal, List, Protocol
import warnings
from .atom_matching import (
    find_atom_mapping_pbc,
    apply_mapping_to_positions,
    apply_inverse_mapping_to_forces,
)

# Unit conversion constants
BOHR_TO_ANGSTROM = 0.529177210903
ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM
HARTREE_TO_EV = 27.211386245988
EV_TO_HARTREE = 1.0 / HARTREE_TO_EV


class _EffectivePotentialProtocol(Protocol):
    _reference_positions: np.ndarray
    _reference_lattice: np.ndarray

    def set_reference_stress(self, stress: np.ndarray) -> None:
        ...

    def evaluate(self, xcart: np.ndarray, rprimd: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        ...


class MultibinitPotential:
    """
    High-level interface to MULTIBINIT effective potential.
    
    Supports two initialization modes:
    1. from_abi(): Load from a .abi input file (standard MULTIBINIT format)
    2. from_params(): Direct initialization without .abi file
    
    Always uses Angstrom and eV for I/O (matching ASE conventions).
    Internal conversions to/from atomic units (Bohr/Hartree) are handled automatically.
    
    Example:
        >>> # Using .abi file
        >>> pot = MultibinitPotential.from_abi("input.abi")
        >>> energy, forces, stress = pot.evaluate(positions_ang, lattice_ang)
        
        >>> # Using parameters
        >>> pot = MultibinitPotential.from_params(
        ...     ddb_file="system_DDB",
        ...     sys_file="system.xml",
        ...     ncell=(2, 2, 2)
        ... )
        >>> energy, forces, stress = pot.evaluate(positions_ang, lattice_ang)
    """
    def __init__(self, lib_path: Optional[str] = None, 
                 use_atomic_units: bool = False,  # DEPRECATED: Ignored, always uses Angstrom/eV
                 auto_match_atoms: bool = True,
                 match_tolerance: float = 0.1,
                 backend: Literal['cffi', 'pyeffpot'] = 'cffi'):
        """
        Initialize the potential object.
        
        Args:
            lib_path: Path to libabinit.so/dylib (optional, only for cffi backend)
            use_atomic_units: DEPRECATED - Ignored. Always uses Angstrom/eV for ASE compatibility.
            auto_match_atoms: If True, automatically match and reorder atoms on first evaluate()
            match_tolerance: Maximum distance for atom matching (in Angstrom)
            backend: 'cffi' for Fortran wrapper, 'pyeffpot' for pure Python
        """
        self.backend = backend
        self._pyeffpot_potential: Optional[_EffectivePotentialProtocol] = None  # For pyeffpot backend
        self._lib_path = lib_path
        self.wrapper = None
        self._use_atomic_units = bool(use_atomic_units)
        
        if use_atomic_units:
            warnings.warn(
                "use_atomic_units is deprecated. Prefer Angstrom/eV inputs for ASE compatibility.",
                DeprecationWarning, stacklevel=2
            )
        self._initialized = False
        
        # Atom matching configuration
        self.auto_match_atoms = auto_match_atoms
        self.match_tolerance = match_tolerance
        
        # Persistent mapping storage
        self._atom_mapping: Optional[np.ndarray] = None  # Forward mapping: ref[i] = input[mapping[i]]
        self._inverse_mapping: Optional[np.ndarray] = None  # Inverse: input[j] -> ref atom index
        self._reference_positions: Optional[np.ndarray] = None  # Reference structure positions
        self._reference_lattice: Optional[np.ndarray] = None  # Reference lattice
        self._mapping_validated = False
        self._is_identity_mapping = False  # Flag to skip force remapping when no reordering needed
        # Chemical identity metadata for ASE export (Bug fix: was hardcoded to 'X')
        self._znucl: Optional[np.ndarray] = None        # (ntypat,) atomic number per species
        self._typat_super: Optional[np.ndarray] = None  # (natom_sc,) species index per supercell atom
        self._typat_ref: Optional[np.ndarray] = None    # (natom_uc,) species index per unit-cell atom
        self._ncell: Optional[Tuple[int, int, int]] = None  # Supercell dimensions

    def _ensure_cffi_wrapper(self):
        """Load libabinit only when a CFFI-backed operation needs it."""
        if self.backend != 'cffi':
            raise RuntimeError("CFFI wrapper requested for non-CFFI backend")
        if self.wrapper is None:
            from .wrapper_cffi import MultibinitWrapperCFFI
            self.wrapper = MultibinitWrapperCFFI(lib_path=self._lib_path)
        return self.wrapper
    
    @property
    def natoms(self) -> int:
        """
        Number of atoms in the supercell.
        
        Returns:
            int: Number of atoms, or 0 if unknown (reference structure not set).
        """
        if self._reference_positions is not None:
            return len(self._reference_positions)
        return 0
    
    @property
    def expected_natoms(self) -> Optional[int]:
        """
        Expected number of atoms based on reference structure.
        
        Returns None until the first evaluate() call, after which it returns
        the number of atoms from the reference structure.
        
        Returns:
            int or None: Expected number of atoms, or None if not set yet.
        """
        if self._reference_positions is not None:
            return len(self._reference_positions)
        return None
        
    @property
    def supercell(self) -> Optional[Tuple[int, int, int]]:
        """
        Get supercell dimensions if known (from DDB or user input).
        """
        return self._ncell

    @classmethod
    def from_abi(cls, abi_file: str, lib_path: Optional[str] = None, 
                 use_atomic_units: bool = False) -> 'MultibinitPotential':
        """
        Create potential from a .abi input file.
        
        Args:
            abi_file: Path to the .abi input file
            lib_path: Path to libabinit.so/dylib (optional)
            use_atomic_units: DEPRECATED - Ignored. Always uses Angstrom/eV.
            
        Returns:
            Initialized MultibinitPotential instance
        """
        pot = cls(lib_path=lib_path, use_atomic_units=use_atomic_units)
        wrapper = pot._ensure_cffi_wrapper()
        wrapper.init_from_abi_file(abi_file)
        pot._initialized = True
        
        # Parse znucl/typat from the DDB referenced in the .abi file
        try:
            from .wrapper_cffi import MultibinitWrapperCFFI
            ddb_path, _, _ = MultibinitWrapperCFFI._parse_abi_file(abi_file)
            pot._load_znucl_from_ddb(ddb_path)
        except Exception as e:
            warnings.warn(f"Could not load znucl metadata from .abi: {e}")
        
        # Automatically get internal supercell reference from C API
        try:
            pot._fetch_internal_supercell_reference()
        except Exception as e:
            warnings.warn(f"Failed to fetch internal supercell reference: {e}")
            
        return pot
    
    @classmethod
    def from_params(cls, ddb_file: str, sys_file: str = "", coeff_file: str = "",
                   ncell: Tuple[int, int, int] = (1, 1, 1),
                   ngqpt: Tuple[int, int, int] = (1, 1, 1),
                   dipdip: int = 1,
                   lib_path: Optional[str] = None,
                   use_atomic_units: bool = False) -> 'MultibinitPotential':
        """
        Create potential from direct parameters (no .abi file).
        
        Args:
            ddb_file: Path to DDB file
            sys_file: Path to system XML file (optional)
            coeff_file: Path to coefficient XML file (optional)
            ncell: Supercell dimensions [nx, ny, nz]
            ngqpt: q-point grid [nqx, nqy, nqz]
            dipdip: Dipole-dipole interactions (0=off, 1=on)
            lib_path: Path to libabinit.so/dylib (optional)
            use_atomic_units: DEPRECATED - Ignored. Always uses Angstrom/eV.
            
        Returns:
            Initialized MultibinitPotential instance
        """
        pot = cls(lib_path=lib_path, use_atomic_units=use_atomic_units)
        wrapper = pot._ensure_cffi_wrapper()
        wrapper.init_from_params(
            ddb_file=ddb_file,
            sys_file=sys_file,
            coeff_file=coeff_file,
            ncell=ncell,
            ngqpt=ngqpt,
            dipdip=dipdip
        )
        pot._initialized = True
        pot._load_znucl_from_ddb(ddb_file)
        
        # Automatically get internal supercell reference from C API
        try:
            pot._fetch_internal_supercell_reference()
        except Exception as e:
            warnings.warn(f"Failed to fetch internal supercell reference: {e}")
            
        return pot
    
    @classmethod
    def from_pyeffpot(cls, ddb_file: str, xml_file: Optional[str] = None,
                      ncell: Tuple[int, int, int] = (4, 4, 4),
                      dipdip: bool = True,
                      asr: bool = True,
                      auto_match_atoms: bool = True,
                      match_tolerance: float = 0.1,
                      reference_stress_ha_bohr3: Optional[np.ndarray] = None) -> 'MultibinitPotential':
        """
        Create potential using pure Python backend (no Fortran dependency).
        
        Args:
            ddb_file: Path to DDB file
            xml_file: Path to coefficient XML file (optional)
            ncell: Supercell dimensions [nx, ny, nz]
            auto_match_atoms: If True, automatically match atoms on first evaluate()
            match_tolerance: Maximum distance for atom matching (in Angstrom)
            
        Returns:
            Initialized MultibinitPotential instance with pyeffpot backend
            
        Example:
            >>> pot = MultibinitPotential.from_pyeffpot(
            ...     ddb_file="system.DDB",
            ...     xml_file="coeffs.xml",
            ...     ncell=(4, 4, 4)
            ... )
            >>> energy, forces, stress = pot.evaluate(positions_ang, lattice_ang)
        """
        from .pyeffpot import EffectivePotential, read_ddb, build_supercell
        from .pyeffpot.xml_parser import read_coefficient_xml
        
        pot = cls(backend='pyeffpot', auto_match_atoms=auto_match_atoms, 
                  match_tolerance=match_tolerance)
        
        unitcell = read_ddb(ddb_file)
        supercell = build_supercell(unitcell, ncell, dipdip=dipdip, asr=asr)
        
        if xml_file:
            from pathlib import Path
            if Path(xml_file).exists():
                coeffs = read_coefficient_xml(xml_file)
                supercell.anharmonic_coeffs = coeffs
        
        pot._pyeffpot_potential = EffectivePotential(supercell)
        pot._znucl = np.asarray(unitcell.znucl, dtype=int).copy()
        pot._typat_ref = np.asarray(unitcell.typat, dtype=int).copy()
        pot._typat_super = np.asarray(supercell.crystal_sc.typat, dtype=int).copy()
        if reference_stress_ha_bohr3 is not None:
            pot._pyeffpot_potential.set_reference_stress(reference_stress_ha_bohr3)
        pot._initialized = True
        
        pot._ncell = ncell
        
        ref_pos_bohr = pot._pyeffpot_potential._reference_positions
        ref_lat_bohr = pot._pyeffpot_potential._reference_lattice
        pot.set_reference_structure(
            ref_pos_bohr * BOHR_TO_ANGSTROM,
            ref_lat_bohr * BOHR_TO_ANGSTROM
        )
        
        return pot
    
    def _load_znucl_from_ddb(self, ddb_file: str) -> None:
        """
        Parse znucl (atomic numbers per species) and typat from a DDB file
        using the pure-Python pyeffpot parser. Stored for ASE symbol resolution.

        Failures are non-fatal: a warning is emitted and the fields stay None,
        causing ASE exports to fall back to 'X' symbols.
        """
        try:
            from .pyeffpot import read_ddb
            unitcell = read_ddb(ddb_file)
            self._znucl = np.asarray(unitcell.znucl, dtype=int).copy()
            self._typat_ref = np.asarray(unitcell.typat, dtype=int).copy()
        except Exception as e:
            warnings.warn(
                f"Could not parse znucl/typat from DDB '{ddb_file}': {e}. "
                f"ASE exports will use placeholder 'X' symbols.",
                RuntimeWarning,
                stacklevel=2,
            )

    def _fetch_internal_supercell_reference(self):
        """
        Fetch the expected supercell structure directly from the C library.
        Sets _reference_positions and _reference_lattice.
        """
        if not self._initialized:
            return
            
        # Get supercell info (in Bohr) from C API
        try:
            wrapper = self._ensure_cffi_wrapper()
            natom_super, species, pos_super_bohr, lat_super_bohr = wrapper.get_supercell_structure()
        except RuntimeError:
            # Might happen if initialization failed or data not ready
            return
        except AttributeError:
            # Fallback if C wrapper hasn't been updated with get_supercell_structure yet
            warnings.warn("get_supercell_structure not found in C wrapper. Cannot set reference structure.")
            return

        # Convert to Angstrom
        pos_super = pos_super_bohr * BOHR_TO_ANGSTROM
        lat_super = lat_super_bohr * BOHR_TO_ANGSTROM
        self._typat_super = np.asarray(species, dtype=int).copy()
        
        self.set_reference_structure(pos_super, lat_super)
    
    @classmethod
    def from_config_file(cls, config_file: str) -> 'MultibinitPotential':
        """
        Create potential from a configuration file.
        
        The configuration file can specify either:
        1. abi_file: Path to .abi input file (standard MULTIBINIT format)
        2. ddb_file + sys_file: Direct initialization without .abi file
        
        Simple format example:
            ```
            ddb_file: system_DDB
            sys_file: system.xml
            ncell: 2 2 2
            ```
        
        INI-like format example:
            ```
            [files]
            ddb_file = system_DDB
            sys_file = system.xml
            
            [parameters]
            ncell = 2 2 2
            ngqpt = 4 4 4
            dipdip = 1
            ```
        
        Args:
            config_file: Path to the configuration file
            
        Returns:
            Initialized MultibinitPotential instance
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If required parameters are missing or invalid
        """
        from .config import MultibinitConfig
        
        config = MultibinitConfig.from_file(config_file)
        
        # Determine initialization mode
        if config.is_abi_mode():
            if config.abi_file is None:
                raise ValueError("abi_file is required for ABI mode")
            # Use .abi file
            pot = cls.from_abi(
                abi_file=config.abi_file,
                lib_path=config.lib_path,
                use_atomic_units=config.use_atomic_units
            )
        else:
            if config.ddb_file is None:
                raise ValueError("ddb_file is required for parameter mode")
            # Use parameters
            pot = cls.from_params(
                ddb_file=config.ddb_file,
                sys_file=config.sys_file or "",
                coeff_file=config.coeff_file,
                ncell=config.ncell,
                ngqpt=config.ngqpt,
                dipdip=config.dipdip,
                lib_path=config.lib_path,
                use_atomic_units=config.use_atomic_units
            )
        
        # Apply atom matching configuration
        pot.auto_match_atoms = config.auto_match_atoms
        pot.match_tolerance = config.match_tolerance
        
        return pot
    
    def set_reference_structure(self, positions: np.ndarray, lattice: np.ndarray):
        """
        Set the reference structure for atom matching.
        
        This should be called with the MULTIBINIT internal supercell structure.
        If not called, the first structure passed to evaluate() will be used as reference.
        
        Args:
            positions: Reference atomic positions, shape (natom, 3)
            lattice: Reference lattice vectors, shape (3, 3)
        """
        positions_array = np.array(positions, dtype=np.float64)
        lattice_array = np.array(lattice, dtype=np.float64)
        if self._use_atomic_units:
            positions_array = positions_array * BOHR_TO_ANGSTROM
            lattice_array = lattice_array * BOHR_TO_ANGSTROM
        self._reference_positions = positions_array
        self._reference_lattice = lattice_array
        self._mapping_validated = False
        
        # Clear any existing mapping
        self._atom_mapping = None
        self._inverse_mapping = None
        self._is_identity_mapping = False
    
    def compute_atom_mapping(self, positions: np.ndarray, lattice: np.ndarray, 
                            tolerance: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute atom mapping between input structure and reference.
        
        The mapping is stored internally and reused for subsequent evaluations.
        
        Args:
            positions: Input atomic positions, shape (natom, 3)
            lattice: Input lattice vectors, shape (3, 3)
            tolerance: Maximum distance for matching (uses self.match_tolerance if None)
            
        Returns:
            tuple: (mapping, inverse_mapping)
                - mapping[i] = j means reference atom i matches input atom j
                - inverse_mapping[j] = i means input atom j corresponds to reference atom i
                
        Raises:
            RuntimeError: If reference structure not set
        """
        if self._reference_positions is None:
            raise RuntimeError("Reference structure not set. Call set_reference_structure() first.")
        
        if tolerance is None:
            tolerance = self.match_tolerance
        
        mapping, inverse_mapping = find_atom_mapping_pbc(
            positions_input=positions,
            positions_ref=self._reference_positions,
            lattice=lattice,
            tolerance=tolerance
        )
        
        # Store mapping for reuse
        self._atom_mapping = mapping
        self._inverse_mapping = inverse_mapping
        self._mapping_validated = True
        
        # Check if this is identity mapping with no PBC shifts
        from .atom_matching import is_identity_mapping_no_pbc_shift
        self._is_identity_mapping = is_identity_mapping_no_pbc_shift(
            positions, self._reference_positions, mapping, lattice, tolerance=1e-6
        )
        
        return mapping, inverse_mapping
    
    def clear_atom_mapping(self):
        """Clear stored atom mapping. Next evaluate() will recompute if auto_match_atoms=True."""
        self._atom_mapping = None
        self._inverse_mapping = None
        self._mapping_validated = False
        self._is_identity_mapping = False
    
    def get_atom_mapping(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Get the currently stored atom mapping.
        
        Returns:
            tuple or None: (mapping, inverse_mapping) if mapping exists, None otherwise
        """
        if self._atom_mapping is not None and self._inverse_mapping is not None:
            return self._atom_mapping.copy(), self._inverse_mapping.copy()
        return None
    
    def evaluate(self, positions: np.ndarray, lattice: np.ndarray, 
                skip_atom_matching: bool = False) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Evaluate energy, forces, and stresses.
        
        If auto_match_atoms=True (default), automatically matches and reorders atoms
        on the first call, then reuses the mapping for subsequent calls.
        
        Args:
            positions: Atomic positions, shape (natom, 3)
                      In Angstrom if use_atomic_units=False, Bohr if True
            lattice: Lattice vectors (row vectors), shape (3, 3)
                    In Angstrom if use_atomic_units=False, Bohr if True
            skip_atom_matching: If True, skip atom matching even if auto_match_atoms=True.
                               Use this if you're certain atoms are already in correct order.
            
        Returns:
            tuple: (energy, forces, stress)
                   - energy: float (eV or Hartree)
                   - forces: array shape (natom, 3) in INPUT atom order (eV/Angstrom or Hartree/Bohr)
                   - stress: array shape (6,) Voigt order (eV/Angstrom^3 or Hartree/Bohr^3)
                   
        Raises:
            RuntimeError: If not initialized or if atom matching fails
            
        Notes:
            - Forces are automatically mapped back to the input atom order
            - The atom mapping is computed once and reused for efficiency
            - Call clear_atom_mapping() to reset if structure changes significantly
        """
        if not self._initialized:
            raise RuntimeError("Potential not initialized. Use from_abi() or from_params().")
        
        # Validate number of atoms
        expected_n = self.expected_natoms
        if expected_n is not None and len(positions) != expected_n:
            raise ValueError(
                f"Structure has {len(positions)} atoms, but the potential expects {expected_n} atoms. "
                f"This mismatch usually means the supercell size is incompatible with the ncell parameter "
                f"used during initialization. Make sure the input structure has the correct supercell size."
            )
        
        # Handle atom matching
        positions_for_eval = positions
        need_force_remapping = False
        
        if self.auto_match_atoms and not skip_atom_matching:
            # Set reference on first call
            if self._reference_positions is None:
                self._reference_positions = np.array(positions, dtype=np.float64)
                self._reference_lattice = np.array(lattice, dtype=np.float64)
                self._mapping_validated = True
                self._is_identity_mapping = True  # First call is always identity
                warnings.warn(
                    "Using first input structure as reference for atom matching. "
                    "If this is not the MULTIBINIT internal supercell, call "
                    "set_reference_structure() with the correct reference.",
                    RuntimeWarning
                )
            
            # Compute or reuse mapping
            if self._atom_mapping is None:
                try:
                    self.compute_atom_mapping(positions, lattice)
                    
                    # Check if reordering or PBC shifting is needed
                    if not self._is_identity_mapping:
                        if self._atom_mapping is not None and not np.array_equal(self._atom_mapping, np.arange(len(positions))):
                            warnings.warn(
                                f"Atom ordering differs from reference. Automatically reordering "
                                f"{len(positions)} atoms. Forces will be mapped back to input order.",
                                RuntimeWarning
                            )
                        need_force_remapping = True
                    
                except Exception as e:
                    raise RuntimeError(
                        f"Atom matching failed: {e}\n"
                        f"Set auto_match_atoms=False if you're certain atoms are in correct order, "
                        f"or call set_reference_structure() with the MULTIBINIT internal supercell."
                    ) from e
            else:
                # Reuse stored mapping - use cached identity flag
                need_force_remapping = not self._is_identity_mapping
            
            # Reorder positions if needed (skip if identity mapping with no PBC shift)
            if need_force_remapping and self._atom_mapping is not None:
                positions_for_eval = apply_mapping_to_positions(positions, self._atom_mapping)
        
        # Convert input from Angstrom to Bohr (both backends use atomic units internally)
        pos_bohr = positions_for_eval * ANGSTROM_TO_BOHR
        lat_bohr = lattice * ANGSTROM_TO_BOHR
        
        # Call backend (both return atomic units: Hartree, Hartree/Bohr, Hartree/Bohr^3)
        if self.backend == 'pyeffpot':
            assert self._pyeffpot_potential is not None
            energy_ha, forces_ha_bohr, stress_ha_bohr3 = self._pyeffpot_potential.evaluate(
                pos_bohr, lat_bohr
            )
            if stress_ha_bohr3.shape == (3, 3):
                stress_ha_bohr3 = np.array([
                    stress_ha_bohr3[0, 0],
                    stress_ha_bohr3[1, 1],
                    stress_ha_bohr3[2, 2],
                    stress_ha_bohr3[1, 2],
                    stress_ha_bohr3[0, 2],
                    stress_ha_bohr3[0, 1],
                ])
        else:
            wrapper = self._ensure_cffi_wrapper()
            energy_ha, forces_ha_bohr, stress_ha_bohr3 = wrapper.evaluate(pos_bohr, lat_bohr)
        
        # Map forces back to input order if needed
        # Skip if identity mapping (optimization)
        if need_force_remapping and self._inverse_mapping is not None:
            forces_ha_bohr = apply_inverse_mapping_to_forces(forces_ha_bohr, self._inverse_mapping)
        
        # Convert output from atomic units to Angstrom/eV
        energy_ev = energy_ha * HARTREE_TO_EV
        forces_ev_ang = forces_ha_bohr * (HARTREE_TO_EV / BOHR_TO_ANGSTROM)
        stress_ev_ang3 = stress_ha_bohr3 * (HARTREE_TO_EV / (BOHR_TO_ANGSTROM ** 3))
        return energy_ev, forces_ev_ang, stress_ev_ang3
    
    def get_supercell_structure(self) -> Optional[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]:
        """
        Get the MULTIBINIT internal supercell structure (reference structure).
        
        This returns the reference structure that was either:
        1. Set explicitly via set_reference_structure()
        2. Automatically captured from the first evaluate() call
        
        Returns:
            tuple or None: (positions, lattice, atomic_numbers) if reference is set, None otherwise
                - positions: array shape (natom, 3) in current units (Angstrom or Bohr)
                - lattice: array shape (3, 3) in current units
                - atomic_numbers: array shape (natom,) or None if not available
                
        Notes:
            - If auto_match_atoms=True, the first evaluate() call sets the reference
            - Atomic numbers are resolved from DDB metadata when available, otherwise None
            - Call this after at least one evaluate() if using automatic reference
        """
        if self._reference_positions is None or self._reference_lattice is None:
            return None
        
        atomic_numbers: Optional[np.ndarray] = None
        if self._typat_super is not None and self._znucl is not None:
            try:
                # typat is 1-indexed species number; znucl[typat-1] is the atomic number
                typat = np.asarray(self._typat_super, dtype=int)
                atomic_numbers = np.asarray(self._znucl, dtype=int)[typat - 1]
            except (IndexError, ValueError) as e:
                warnings.warn(f"Could not resolve atomic numbers from typat/znucl: {e}")
        
        return (
            self._reference_positions.copy(),
            self._reference_lattice.copy(),
            None if atomic_numbers is None else atomic_numbers.copy(),
        )
    
    def export_supercell_to_ase(self):
        """
        Export the MULTIBINIT internal supercell structure to ASE Atoms object.
        
        Returns:
            ase.Atoms: Atoms object with supercell structure
            
        Raises:
            RuntimeError: If reference structure not available
            ImportError: If ASE is not installed
            
        Notes:
            - Positions and cell in Angstrom (ASE convention)
            - Atomic symbols are resolved from DDB metadata (typat→znucl→symbol)
            - If no DDB metadata is available, symbols default to 'X'
            
        Example:
            >>> pot = MultibinitPotential.from_params(...)
            >>> pot.evaluate(positions, lattice)  # Sets reference
            >>> atoms = pot.export_supercell_to_ase()
            >>> atoms.write('supercell.cif')
        """
        try:
            from ase import Atoms
        except ImportError:
            raise ImportError("ASE is required for this function. Install with: pip install ase")
        
        structure = self.get_supercell_structure()
        if structure is None:
            raise RuntimeError(
                "Reference structure not available. Call evaluate() at least once, "
                "or use set_reference_structure()."
            )
        
        positions, lattice, atomic_numbers = structure
        
        # Positions and lattice are already in Angstrom (from wrapper or already converted)
        natom = len(positions)
        if atomic_numbers is not None:
            from ase.data import chemical_symbols
            symbols = [chemical_symbols[int(z)] for z in atomic_numbers]
        else:
            symbols = ['X'] * natom  # Fallback: no DDB metadata available

        atoms = Atoms(
            symbols=symbols,
            positions=positions,
            cell=lattice,
            pbc=True
        )
        
        return atoms
    
    def export_supercell_to_file(self, filename: str, format: Optional[str] = None):
        """
        Export the MULTIBINIT internal supercell structure to a file.
        
        Supported formats: cif, vasp, poscar, xyz, json, extxyz, etc. (any ASE format)
        
        Args:
            filename: Output filename
            format: File format (auto-detected from extension if None)
            
        Raises:
            RuntimeError: If reference structure not available
            ImportError: If ASE is not installed
            
        Notes:
            - Atomic symbols are resolved from DDB metadata (typat→znucl→symbol)
            - If no DDB metadata is available, symbols default to 'X'
            
        Example:
            >>> pot = MultibinitPotential.from_params(...)
            >>> pot.evaluate(positions, lattice)  # Sets reference
            >>> pot.export_supercell_to_file('supercell.cif')
        """
        atoms = self.export_supercell_to_ase()
        atoms.write(filename, format=format)
    
    def export_reference_to_ase(self):
        """
        Export the reference unit cell structure to ASE Atoms object.
        
        This uses the new C API to get the unit cell structure directly
        without needing to call evaluate() first.
        
        Returns:
            ase.Atoms: Atoms object with unit cell structure
            
        Raises:
            RuntimeError: If potential not initialized
            ImportError: If ASE is not installed
            
        Notes:
            - Positions and cell in Angstrom (ASE convention)
            - Atomic symbols are resolved from DDB metadata (typat→znucl→symbol)
            - If no DDB metadata is available, symbols default to 'X'
            
        Example:
            >>> pot = MultibinitPotential.from_config_file('config.conf')
            >>> atoms = pot.export_reference_to_ase()
            >>> atoms.write('unit_cell.cif')
        """
        try:
            from ase import Atoms
            from ase.data import chemical_symbols
        except ImportError:
            raise ImportError("ASE is required for this function. Install with: pip install ase")
        
        if not self._initialized:
            raise RuntimeError("Potential not initialized. Call from_params() or from_config_file() first.")
        
        # Get reference structure from C API
        wrapper = self._ensure_cffi_wrapper()
        natom, species, positions, lattice = wrapper.get_reference_structure()
        
        # Convert to Angstrom if needed (C API returns Bohr)
        positions_ang = positions * BOHR_TO_ANGSTROM
        lattice_ang = lattice * BOHR_TO_ANGSTROM
        
        # Convert typat (1-indexed species) → atomic number via znucl → ASE symbol
        symbols: List[str] = []
        if self._znucl is not None:
            znucl_arr = np.asarray(self._znucl, dtype=int)
            for typat in species:
                typat_int = int(typat)
                if 1 <= typat_int <= len(znucl_arr):
                    z = int(znucl_arr[typat_int - 1])
                    symbols.append(chemical_symbols[z] if 0 < z < len(chemical_symbols) else 'X')
                else:
                    symbols.append('X')
        else:
            # No znucl metadata — fall back to 'X' for all atoms
            symbols = ['X'] * len(species)
        
        # Create Atoms object
        atoms = Atoms(
            symbols=symbols,
            positions=positions_ang,
            cell=lattice_ang,
            pbc=True
        )
        
        return atoms
    
    def export_reference_to_file(self, filename: str, format: Optional[str] = None):
        """
        Export the reference unit cell structure to a file.
        
        Supported formats: cif, vasp, poscar, xyz, json, extxyz, etc. (any ASE format)
        
        Args:
            filename: Output filename
            format: File format (auto-detected from extension if None)
            
        Raises:
            RuntimeError: If potential not initialized
            ImportError: If ASE is not installed
            
        Example:
            >>> pot = MultibinitPotential.from_config_file('config.conf')
            >>> pot.export_reference_to_file('unit_cell.cif')
        """
        atoms = self.export_reference_to_ase()
        atoms.write(filename, format=format)  # type: ignore
    
    def free(self):
        """Free resources."""
        if self._initialized:
            if self.backend == 'cffi' and self.wrapper is not None:
                self.wrapper.free()
            self._initialized = False
    
    def __enter__(self):
        """Context manager support."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup."""
        self.free()
        return False
    
    def __del__(self):
        """Destructor."""
        #self.free()
        pass
