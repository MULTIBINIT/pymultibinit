"""
ASE Calculator interface for MULTIBINIT effective potential.

Allows seamless integration with the Atomic Simulation Environment (ASE)
for structure optimization, molecular dynamics, phonon calculations, etc.
"""
from ase.calculators.calculator import Calculator, all_changes
import numpy as np
from typing import Optional, Tuple, Literal
from .potential import MultibinitPotential


class MultibinitCalculator(Calculator):
    """
    ASE calculator for ABINIT's MULTIBINIT effective potential.
    
    This calculator wraps the MULTIBINIT C API and provides a standard ASE
    interface for energy, force, and stress calculations.
    
    Properties implemented: energy, forces, stress
    
    Two backends are available:
        - 'cffi'   : Requires libabinit.so/dylib (Fortran). Use from_abi(),
                     from_params(), or from_config_file().
        - 'pyeffpot': Pure Python, no Fortran dependency. Use from_pyeffpot().
                       Reads a DDB file, an optional XML coefficient file, and
                       activates dipole-dipole / supercell via keyword args.

    Example:
        >>> from ase import Atoms
        >>> from pymultibinit import MultibinitCalculator
        >>>
        >>> # --- CFFI (Fortran) backend ---
        >>> # Using .abi file
        >>> calc = MultibinitCalculator.from_abi("input.abi")
        >>> atoms.calc = calc
        >>> energy = atoms.get_potential_energy()
        >>>
        >>> # Using parameters
        >>> calc = MultibinitCalculator.from_params(
        ...     ddb_file="system_DDB",
        ...     sys_file="system.xml",
        ...     ncell=(2, 2, 2)
        ... )
        >>> atoms.calc = calc
        >>>
        >>> # --- Pure Python (pyeffpot) backend, no Fortran needed ---
        >>> calc = MultibinitCalculator.from_pyeffpot(
        ...     ddb_file="system.DDB",
        ...     xml_file="coeffs.xml",   # optional, None to skip
        ...     ncell=(4, 4, 4),         # supercell size
        ...     dipdip=True,             # dipole-dipole (long-range Coulomb)
        ... )
        >>> atoms.calc = calc
    """
    
    implemented_properties = ['energy', 'forces', 'stress']
    
    def __init__(self, potential: MultibinitPotential, **kwargs):
        """
        Initialize the calculator with an existing potential.
        
        Args:
            potential: Initialized MultibinitPotential instance
            **kwargs: Additional arguments for ASE Calculator
        """
        super().__init__(**kwargs)
        self.potential = potential
    
    @classmethod
    def from_abi(cls, abi_file: str, lib_path: Optional[str] = None,
                **kwargs) -> 'MultibinitCalculator':
        """
        Create calculator from a .abi input file.
        
        Args:
            abi_file: Path to the .abi input file
            lib_path: Path to libabinit.so/dylib (optional)
            **kwargs: Additional arguments for ASE Calculator
            
        Returns:
            Initialized MultibinitCalculator instance
        """
        potential = MultibinitPotential.from_abi(
            abi_file=abi_file,
            lib_path=lib_path
        )
        return cls(potential=potential, **kwargs)
    
    @classmethod
    def from_params(cls, ddb_file: str, sys_file: str = "", coeff_file: str = "",
                   ncell: Tuple[int, int, int] = (1, 1, 1),
                   ngqpt: Tuple[int, int, int] = (1, 1, 1),
                   dipdip: int = 1,
                   lib_path: Optional[str] = None,
                   **kwargs) -> 'MultibinitCalculator':
        """
        Create calculator from direct parameters (no .abi file).
        
        Args:
            ddb_file: Path to DDB file
            sys_file: Path to system XML file (optional)
            coeff_file: Path to coefficient XML file (optional)
            ncell: Supercell dimensions [nx, ny, nz]
            ngqpt: q-point grid [nqx, nqy, nqz]
            dipdip: Dipole-dipole interactions (0=off, 1=on)
            lib_path: Path to libabinit.so/dylib (optional)
            **kwargs: Additional arguments for ASE Calculator
            
        Returns:
            Initialized MultibinitCalculator instance
        """
        potential = MultibinitPotential.from_params(
            ddb_file=ddb_file,
            sys_file=sys_file,
            coeff_file=coeff_file,
            ncell=ncell,
            ngqpt=ngqpt,
            dipdip=dipdip,
            lib_path=lib_path
        )
        return cls(potential=potential, **kwargs)
    
    @classmethod
    def from_pyeffpot(cls, ddb_file: str, xml_file: Optional[str] = None,
                      ncell: Tuple[int, int, int] = (4, 4, 4),
                      dipdip: bool = True,
                      asr: bool = True,
                      auto_match_atoms: bool = True,
                      match_tolerance: float = 0.1,
                      reference_stress_ha_bohr3: Optional[np.ndarray] = None,
                      **kwargs) -> 'MultibinitCalculator':
        """
        Create a calculator using the **pure Python** backend (no Fortran/libabinit required).

        This is the recommended path when libabinit.so is not available.
        It parses the DDB file directly in Python, optionally overlays anharmonic
        coefficients from an XML file, builds the supercell, applies the
        dipole-dipole long-range correction and the acoustic sum rule (ASR),
        and returns a fully functional ASE calculator.

        Args:
            ddb_file: Path to the ABINIT DDB (derivative database) file.
            xml_file: Path to an XML coefficient file with anharmonic terms.
                Optional - pass None or omit to use the harmonic DDB-only model.
            ncell: Supercell dimensions (nx, ny, nz). Defaults to (4, 4, 4).
            dipdip: If True (default), activate dipole-dipole (long-range
                Coulomb) correction using Born effective charges and the
                dielectric tensor read from the DDB. Set False to disable.
            asr: If True (default), enforce the acoustic sum rule on the
                interatomic force constants.
            auto_match_atoms: If True, automatically reorder input atoms to
                match the MULTIBINIT reference on the first evaluate() call.
            match_tolerance: Tolerance (Angstrom) for atom matching.
            reference_stress_ha_bohr3: Optional (3,3) reference stress tensor
                in Hartree/Bohr^3 to subtract (used for parity tests).
            **kwargs: Additional arguments forwarded to the ASE Calculator base.

        Returns:
            Initialized MultibinitCalculator instance backed by pyeffpot.

        Example:
            >>> from ase import Atoms
            >>> from pymultibinit import MultibinitCalculator
            >>>
            >>> # Pure-Python ASE calculator, no libabinit needed
            >>> calc = MultibinitCalculator.from_pyeffpot(
            ...     ddb_file="BTO.DDB",
            ...     xml_file="model.xml",   # optional
            ...     ncell=(2, 2, 2),
            ...     dipdip=True,
            ... )
            >>> atoms = Atoms(...)
            >>> atoms.calc = calc
            >>> energy = atoms.get_potential_energy()   # eV
            >>> forces = atoms.get_forces()             # eV/Angstrom
            >>> stress = atoms.get_stress()             # eV/Angstrom^3
        """
        potential = MultibinitPotential.from_pyeffpot(
            ddb_file=ddb_file,
            xml_file=xml_file,
            ncell=ncell,
            dipdip=dipdip,
            asr=asr,
            auto_match_atoms=auto_match_atoms,
            match_tolerance=match_tolerance,
            reference_stress_ha_bohr3=reference_stress_ha_bohr3,
        )
        return cls(potential=potential, **kwargs)
    
    @classmethod
    def from_config_file(cls, config_file: str, **kwargs) -> 'MultibinitCalculator':
        """
        Create calculator from a configuration file.
        
        The configuration file can specify either:
        1. abi_file: Path to .abi input file
        2. ddb_file + sys_file: Direct initialization
        
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
            ```
        
        Args:
            config_file: Path to the configuration file
            **kwargs: Additional arguments for ASE Calculator
            
        Returns:
            Initialized MultibinitCalculator instance
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If required parameters are missing or invalid
        """
        potential = MultibinitPotential.from_config_file(config_file)
        return cls(potential=potential, **kwargs)
    
    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        """
        Calculate properties for the given atoms object.
        
        Args:
            atoms: ASE Atoms object (if None, uses self.atoms)
            properties: List of properties to calculate
            system_changes: List of changed properties since last calculation
        """
        super().calculate(atoms, properties, system_changes)
        
        # Get positions and cell from atoms (in Angstrom)
        positions = self.atoms.get_positions()  # (natom, 3) in Angstrom
        cell = self.atoms.get_cell().array      # (3, 3) in Angstrom
        
        # Evaluate using potential (handles unit conversion internally)
        energy, forces, stress = self.potential.evaluate(positions, cell)
        
        # Store results in ASE format
        self.results['energy'] = energy  # eV
        self.results['forces'] = forces  # eV/Angstrom
        
        # ASE stress convention:
        # ASE Calculator.get_stress() returns: [xx, yy, zz, yz, xz, xy]
        # Sign convention:
        # ASE standard: positive = tension (expanding the cell lowers energy? No)
        # ASE optimization algorithms (UnitCellFilter) move in direction of -gradient.
        # Gradient w.r.t strain is V * stress.
        # If stress > 0 (tension), dE/d\epsilon > 0. Increasing epsilon (expanding) increases energy.
        # To minimize energy, we should decrease epsilon (contract).
        # So UnitCellFilter should contract the cell if stress is positive.
        #
        # However, some DFT codes output stress with pressure convention (P = -sigma).
        # If ABINIT returns stress where positive = tension, then ASE expects it as is.
        #
        # Let's check ABINIT convention in potential.py:
        # "stress tensor (Hartree/Bohr^3) - Voigt notation"
        # If ABINIT uses thermodynamic stress (dE/de), then it matches ASE definition.
        #
        # Experimentally, if relaxation diverges (explodes), try flipping the sign.
        # Original code had: self.results['stress'] = -stress
        # This implies ABINIT stress was treated as Pressure (positive = compression).
        # If ABINIT stress is actually Tensile (positive = tension), then -stress would mean
        # negative tension (compression), so optimizer would expand to relieve it.
        # If it was already tensile, expanding makes it worse -> divergence!
        #
        # So if divergence occurs with -stress, it means we should probably use +stress.
        
        self.results['stress'] = stress  # eV/Angstrom^3
    
    def __del__(self):
        """Destructor - ensure cleanup."""
        if hasattr(self, 'potential'):
            #self.potential.free()
            pass
