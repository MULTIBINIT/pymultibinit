"""
Pure Python implementation of effective potential evaluation.

This module provides a pure Python alternative to the Fortran-based
effective potential evaluation. It reads unitcell data from DDB/XML files,
builds supercell potentials, and evaluates energy/forces/stress.

Modules:
- ddb_parser: Parse ABINIT DDB files (simple)
- ddb_parser_complete: Complete DDB parser with all q-points
- phonon: Phonon frequency calculation from DDB data
- xml_parser: Parse MULTIBINIT XML coefficient files  
- supercell_builder: Build supercell from unitcell
- datastructures: Core data structures (UnitcellData, SupercellPotential)

Example:
    >>> from pymultibinit.pyeffpot import read_ddb
    >>> from pymultibinit.pyeffpot.phonon import calculate_phonon_frequencies
    >>> from pymultibinit.pyeffpot.xml_parser import read_coefficient_xml
    >>> from pymultibinit.pyeffpot.supercell_builder import build_supercell
    >>> 
    >>> # Load unitcell from DDB
    >>> unitcell = read_ddb("BaHfO3.DDB")
    >>> 
    >>> # Load anharmonic coefficients from XML
    >>> coeffs = read_coefficient_xml("coeffs.xml")
    >>> 
    >>> # Build supercell
    >>> supercell = build_supercell(unitcell, ncell=(4, 4, 4))
"""

__version__ = "0.1.0"

# Import main functions for convenience
from .ddb_parser_complete import read_ddb, UnitcellData
from .ddb_writer import write_ddb
from .phonopy_export import PhonopyDdbExportResult, write_phonopy_from_ddb
from .xml_parser import read_coefficient_xml, write_coefficient_xml
from .supercell_builder import build_supercell, set_anharmonic_coeffs
from .datastructures import CrystalInfo, IFCData, SupercellPotential
from .potential import EffectivePotential
from . import dipdip

__all__ = [
    # DDB I/O
    'read_ddb',
    'write_ddb',
    'UnitcellData',
    'PhonopyDdbExportResult',
    'write_phonopy_from_ddb',
    
    # XML I/O
    'read_coefficient_xml',
    'write_coefficient_xml',
    
    # Supercell building
    'build_supercell',
    'set_anharmonic_coeffs',
    'SupercellPotential',
    'CrystalInfo',
    'IFCData',
    
    # Potential evaluation
    'EffectivePotential',
    
    # Phonon calculations
    'phonon',
    'dipdip',
]
