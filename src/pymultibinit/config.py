"""
Configuration file parser for MULTIBINIT potential initialization.

This module provides utilities to parse configuration files that specify
the paths and parameters needed to initialize a MULTIBINIT potential.

Supports multiple formats:
1. Simple format (3-4 lines): paths only
2. INI format: sections with key=value pairs
3. YAML-like format: key: value pairs
"""
import os
from typing import Dict, Tuple, Optional
from pathlib import Path


class MultibinitConfig:
    """
    Configuration for MULTIBINIT potential initialization.
    
    This class parses configuration files and provides parameters
    for initializing MultibinitPotential instances.
    """
    
    def __init__(self):
        """Initialize empty configuration."""
        # Required parameters
        self.ddb_file: Optional[str] = None
        self.sys_file: Optional[str] = None
        self.abi_file: Optional[str] = None
        
        # Optional parameters
        self.coeff_file: str = ""
        self.ncell: Tuple[int, int, int] = (1, 1, 1)
        self.ngqpt: Tuple[int, int, int] = (1, 1, 1)
        self.dipdip: int = 1
        
        # Backend configuration
        self.lib_path: Optional[str] = None
        self.backend: str = "ctypes"
        self.use_atomic_units: bool = False
        
        # Atom matching configuration
        self.auto_match_atoms: bool = True
        self.match_tolerance: float = 0.1
    
    @classmethod
    def from_file(cls, config_file: str) -> 'MultibinitConfig':
        """
        Load configuration from a file.
        
        Supports multiple formats:
        
        1. Simple format (minimal):
           ```
           ddb_file: path/to/system_DDB
           sys_file: path/to/system.xml
           abi_file: path/to/input.abi
           ```
           
        2. INI-like format:
           ```
           [files]
           ddb_file = path/to/system_DDB
           sys_file = path/to/system.xml
           
           [parameters]
           ncell = 2 2 2
           ngqpt = 4 4 4
           dipdip = 1
           
           [backend]
           lib_path = /path/to/libabinit.so
           backend = ctypes
           ```
        
        3. Comment lines starting with # are ignored
        4. Blank lines are ignored
        
        Args:
            config_file: Path to the configuration file
            
        Returns:
            Initialized MultibinitConfig instance
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If required parameters are missing or invalid
        """
        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        config = cls()
        config._parse_file(config_path)
        config._validate()
        config._resolve_paths(config_path.parent)
        
        return config
    
    def _parse_file(self, config_path: Path) -> None:
        """Parse the configuration file."""
        with open(config_path, 'r') as f:
            lines = f.readlines()
        
        current_section = None
        for line_no, line in enumerate(lines, start=1):
            # Strip whitespace and comments
            line = line.strip()
            if '#' in line:
                line = line[:line.index('#')].strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Check for section header [section_name]
            if line.startswith('[') and line.endswith(']'):
                current_section = line[1:-1].strip().lower()
                continue
            
            # Parse key-value pairs
            if '=' in line or ':' in line:
                # Support both '=' and ':' as separators
                separator = '=' if '=' in line else ':'
                parts = line.split(separator, 1)
                if len(parts) != 2:
                    continue
                
                key = parts[0].strip().lower()
                value = parts[1].strip()
                
                self._set_parameter(key, value, line_no)
    
    def _set_parameter(self, key: str, value: str, line_no: int) -> None:
        """Set a configuration parameter."""
        try:
            if key == 'ddb_file':
                self.ddb_file = value
            elif key == 'sys_file':
                self.sys_file = value
            elif key == 'abi_file':
                self.abi_file = value
            elif key == 'coeff_file':
                self.coeff_file = value
            elif key == 'ncell':
                self.ncell = self._parse_triple(value)
            elif key == 'ngqpt':
                self.ngqpt = self._parse_triple(value)
            elif key == 'dipdip':
                self.dipdip = int(value)
            elif key == 'lib_path':
                self.lib_path = value
            elif key == 'backend':
                if value not in ['ctypes', 'cffi']:
                    raise ValueError(f"Invalid backend: {value}")
                self.backend = value
            elif key == 'use_atomic_units':
                # Deprecated parameter - warn user
                import warnings
                warnings.warn(
                    "The 'use_atomic_units' parameter is deprecated and ignored. "
                    "PyMultibinit always uses Angstrom/eV for ASE compatibility. "
                    "Please remove this parameter from your configuration file.",
                    DeprecationWarning, stacklevel=2
                )
                self.use_atomic_units = self._parse_bool(value)
            elif key == 'auto_match_atoms':
                self.auto_match_atoms = self._parse_bool(value)
            elif key == 'match_tolerance':
                self.match_tolerance = float(value)
            else:
                # Unknown parameter - just warn, don't fail
                import warnings
                warnings.warn(f"Unknown configuration parameter '{key}' at line {line_no}")
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid value for '{key}' at line {line_no}: {e}")
    
    def _parse_triple(self, value: str) -> Tuple[int, int, int]:
        """Parse a triple of integers like '2 2 2' or '2,2,2'."""
        # Support both space and comma separators
        value = value.replace(',', ' ')
        parts = value.split()
        if len(parts) != 3:
            raise ValueError(f"Expected 3 integers, got: {value}")
        return (int(parts[0]), int(parts[1]), int(parts[2]))
    
    def _parse_bool(self, value: str) -> bool:
        """Parse a boolean value."""
        value = value.lower()
        if value in ['true', 'yes', '1', 'on']:
            return True
        elif value in ['false', 'no', '0', 'off']:
            return False
        else:
            raise ValueError(f"Invalid boolean value: {value}")
    
    def _validate(self) -> None:
        """Validate that required parameters are set."""
        # Must have either abi_file OR (ddb_file)
        if self.abi_file is None and self.ddb_file is None:
            raise ValueError(
                "Configuration must specify either 'abi_file' or 'ddb_file'"
            )
        
        # If not using abi_file, ddb_file is required
        if self.abi_file is None and self.ddb_file is None:
            raise ValueError("'ddb_file' is required when not using 'abi_file'")
    
    def _resolve_paths(self, base_dir: Path) -> None:
        """Resolve relative paths relative to the config file directory."""
        if self.ddb_file and not os.path.isabs(self.ddb_file):
            self.ddb_file = str(base_dir / self.ddb_file)
        
        if self.sys_file and not os.path.isabs(self.sys_file):
            self.sys_file = str(base_dir / self.sys_file)
        
        if self.abi_file and not os.path.isabs(self.abi_file):
            self.abi_file = str(base_dir / self.abi_file)
        
        if self.coeff_file and not os.path.isabs(self.coeff_file):
            self.coeff_file = str(base_dir / self.coeff_file)
        
        if self.lib_path and not os.path.isabs(self.lib_path):
            self.lib_path = str(base_dir / self.lib_path)
    
    def is_abi_mode(self) -> bool:
        """Check if configuration uses .abi file mode."""
        return self.abi_file is not None
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return {
            'ddb_file': self.ddb_file,
            'sys_file': self.sys_file,
            'abi_file': self.abi_file,
            'coeff_file': self.coeff_file,
            'ncell': self.ncell,
            'ngqpt': self.ngqpt,
            'dipdip': self.dipdip,
            'lib_path': self.lib_path,
            'backend': self.backend,
            'use_atomic_units': self.use_atomic_units,
            'auto_match_atoms': self.auto_match_atoms,
            'match_tolerance': self.match_tolerance,
        }
