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
    
    Example:
        >>> from ase import Atoms
        >>> from pymultibinit import MultibinitCalculator
        >>> 
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
    """
    
    implemented_properties = ['energy', 'forces', 'stress']
    
    def __init__(self, potential: MultibinitPotential, **kwargs):
        """
        Initialize the calculator with an existing potential.
        
        Args:
            potential: Initialized MultibinitPotential instance
            **kwargs: Additional arguments for ASE Calculator
        """
        Calculator.__init__(self, **kwargs)
        self.potential = potential
    
    @classmethod
    def from_abi(cls, abi_file: str, lib_path: Optional[str] = None,
                backend: Literal["ctypes", "cffi"] = "ctypes",
                **kwargs) -> 'MultibinitCalculator':
        """
        Create calculator from a .abi input file.
        
        Args:
            abi_file: Path to the .abi input file
            lib_path: Path to libabinit.so/dylib (optional)
            backend: Which wrapper backend to use ("ctypes" or "cffi")
            **kwargs: Additional arguments for ASE Calculator
            
        Returns:
            Initialized MultibinitCalculator instance
        """
        potential = MultibinitPotential.from_abi(
            abi_file=abi_file,
            lib_path=lib_path,
            use_atomic_units=False,  # ASE uses eV/Angstrom
            backend=backend
        )
        return cls(potential=potential, **kwargs)
    
    @classmethod
    def from_params(cls, ddb_file: str, sys_file: str = "", coeff_file: str = "",
                   ncell: Tuple[int, int, int] = (1, 1, 1),
                   ngqpt: Tuple[int, int, int] = (1, 1, 1),
                   dipdip: int = 1,
                   lib_path: Optional[str] = None,
                   backend: Literal["ctypes", "cffi"] = "ctypes",
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
            backend: Which wrapper backend to use ("ctypes" or "cffi")
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
            lib_path=lib_path,
            use_atomic_units=False,  # ASE uses eV/Angstrom
            backend=backend
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
        Calculator.calculate(self, atoms, properties, system_changes)
        
        # Get positions and cell from atoms (in Angstrom)
        positions = self.atoms.get_positions()  # (natom, 3) in Angstrom
        cell = self.atoms.get_cell().array      # (3, 3) in Angstrom
        
        # Evaluate using potential (handles unit conversion internally)
        energy, forces, stress = self.potential.evaluate(positions, cell)
        
        # Store results in ASE format
        self.results['energy'] = energy  # eV
        self.results['forces'] = forces  # eV/Angstrom
        
        # ASE stress convention: negative of standard Voigt stress
        # (ASE uses pressure-like sign convention)
        # Standard Voigt: [xx, yy, zz, yz, xz, xy]
        self.results['stress'] = -stress  # eV/Angstrom^3
    
    def __del__(self):
        """Destructor - ensure cleanup."""
        if hasattr(self, 'potential'):
            self.potential.free()
