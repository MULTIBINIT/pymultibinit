"""
pymultibinit: Python interface to MULTIBINIT effective potential.

This package provides Python bindings to ABINIT's MULTIBINIT library for
computing energies, forces, and stresses using effective potentials derived
from DFT calculations.

Main classes:
    - MultibinitPotential: High-level potential interface
    - MultibinitCalculator: ASE calculator interface
    - MultibinitWrapperCFFI: Low-level CFFI wrapper (advanced users)

Example:
    >>> from pymultibinit import MultibinitCalculator
    >>> from ase import Atoms
    >>> 
    >>> # Create calculator from .abi file
    >>> calc = MultibinitCalculator.from_abi("input.abi")
    >>> atoms.calc = calc
    >>> energy = atoms.get_potential_energy()
    >>> forces = atoms.get_forces()
    
    >>> # Or from parameters
    >>> calc = MultibinitCalculator.from_params(
    ...     ddb_file="system_DDB",
    ...     sys_file="system.xml",
    ...     ncell=(2, 2, 2)
    ... )
"""

from .potential import MultibinitPotential
from .calculator import MultibinitCalculator
from .wrapper_cffi import MultibinitWrapperCFFI

# Atom matching utilities
from . import atom_matching

# Configuration file support
from .config import MultibinitConfig

__version__ = "0.2.0"

__all__ = [
    "MultibinitPotential",
    "MultibinitCalculator", 
    "MultibinitWrapperCFFI",
    "MultibinitConfig",
    "atom_matching",
]



