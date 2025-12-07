"""
CFFI-based wrapper for MULTIBINIT C API.

This module provides an alternative to the ctypes wrapper using CFFI,
which offers better performance and ABI/API mode support.

CFFI has two modes:
- ABI mode (used here): No C compiler needed, loads existing shared library
- API mode: Requires C compiler, generates optimized bindings

This implementation uses ABI mode for simplicity and portability.
"""
from cffi import FFI
import numpy as np
import re
from pathlib import Path
from typing import Optional, Tuple
from .utils import find_library


class MultibinitWrapperCFFI:
    """
    CFFI-based wrapper for MULTIBINIT C API.
    
    Alternative to MultibinitWrapper (ctypes) with potentially better performance.
    Provides the same interface as MultibinitWrapper for drop-in compatibility.
    
    All units are atomic units (Bohr, Hartree).
    """
    
    def __init__(self, lib_path: Optional[str] = None):
        """
        Initialize the wrapper and load the shared library.
        
        Args:
            lib_path: Path to libabinit.so/dylib. If None, searches in standard locations.
        """
        if lib_path is None:
            lib_path = self._find_library()
        
        self.ffi = FFI()
        self._setup_cdefs()
        self.lib = self.ffi.dlopen(lib_path)
        self.handle = self.ffi.new("void**")
        self.handle[0] = self.ffi.NULL
    
    def _find_library(self) -> str:
        """
        Find the libabinit shared library.
        
        Searches for library with various extensions (.dylib, .so, .dll)
        across different platforms.
        """
        return find_library(lib_name="libabinit")
    
    @staticmethod
    def _parse_abi_file(abi_filename: str) -> Tuple[str, str, Tuple[int, int, int]]:
        """
        Parse .abi file to extract input_ddb, coeff_xml filenames, and ncell.
        
        Args:
            abi_filename: Path to the .abi file
            
        Returns:
            tuple: (ddb_filename, xml_filename, ncell) - paths relative to .abi file directory
            
        Raises:
            FileNotFoundError: If .abi file doesn't exist
            ValueError: If required fields are not found
        """
        abi_path = Path(abi_filename)
        if not abi_path.exists():
            raise FileNotFoundError(f"ABI file not found: {abi_filename}")
        
        abi_dir = abi_path.parent
        ddb_file = ""
        xml_file = ""
        ncell = (1, 1, 1)  # Default value
        
        with open(abi_path, 'r') as f:
            for line in f:
                # Look for: input_ddb = filename
                match = re.search(r'input_ddb\s*=\s*(\S+)', line, re.IGNORECASE)
                if match:
                    ddb_file = match.group(1).strip()
                
                # Look for: coeff_xml = filename
                match = re.search(r'coeff_xml\s*=\s*(\S+)', line, re.IGNORECASE)
                if match:
                    xml_file = match.group(1).strip()
                
                # Look for: ncell = nx ny nz
                match = re.search(r'ncell\s*=\s*(\d+)\s+(\d+)\s+(\d+)', line, re.IGNORECASE)
                if match:
                    ncell = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
        
        if not ddb_file:
            raise ValueError(f"input_ddb not found in {abi_filename}")
        
        # Resolve paths relative to .abi file directory
        ddb_path = str(abi_dir / ddb_file)
        xml_path = str(abi_dir / xml_file) if xml_file else ""
        
        return ddb_path, xml_path, ncell
    
    def _setup_cdefs(self):
        """Define C function signatures for CFFI."""
        self.ffi.cdef("""
            void mb_init_potential(const char* abi_input, int abi_len,
                                   const char* sys_filename, int sys_len,
                                   const char* coeff_filename, int coeff_len,
                                   const int* ncell,
                                   void** handle, int* status);
            
            void mb_init_potential_simple(const char* sys_filename, int sys_len,
                                          const char* coeff_filename, int coeff_len,
                                          const int* ncell, const int* ngqpt,
                                          int dipdip_flag,
                                          void** handle, int* status);
            
            void mb_evaluate(void* handle, const double* positions, const double* lattice,
                            int natom, double* energy, double* forces, double* stresses,
                            int* status);
            
            void mb_free_potential(void* handle, int* status);
        """)
    
    def init_from_abi_file(self, abi_filename: str, sys_file: str = "", coeff_file: str = "",
                          ncell: Tuple[int, int, int] = (1, 1, 1)):
        """
        Initialize potential from a .abi input file.
        
        Uses the standard MULTIBINIT input file parser (invars10) to read all
        parameters including DDB file, XML files, grid settings, etc.
        
        Args:
            abi_filename: Path to the .abi input file
            sys_file: Path to DDB file (optional, auto-parsed from .abi if not provided)
            coeff_file: Path to coefficient XML file (optional, auto-parsed from .abi if not provided)
            ncell: Supercell dimensions [nx, ny, nz] (default: 1,1,1)
            
        Raises:
            RuntimeError: If initialization fails
            FileNotFoundError: If .abi file doesn't exist
            ValueError: If required files not found in .abi
        """
        # If sys_file or coeff_file not provided, parse them from .abi file
        if not sys_file or not coeff_file:
            parsed_ddb, parsed_xml, parsed_ncell = self._parse_abi_file(abi_filename)
            if not sys_file:
                sys_file = parsed_ddb
            if not coeff_file:
                coeff_file = parsed_xml
            # Use parsed ncell if the default (1,1,1) was passed
            if ncell == (1, 1, 1):
                ncell = parsed_ncell
        
        abi_bytes = abi_filename.encode('utf-8')
        sys_bytes = sys_file.encode('utf-8') if sys_file else b""
        coeff_bytes = coeff_file.encode('utf-8') if coeff_file else b""
        ncell_arr = self.ffi.new("int[3]", ncell)
        status = self.ffi.new("int*")
        
        # Use a temporary handle pointer
        temp_handle = self.ffi.new("void**")
        
        self.lib.mb_init_potential(
            abi_bytes,
            len(abi_bytes),
            sys_bytes,
            len(sys_bytes),
            coeff_bytes,
            len(coeff_bytes),
            ncell_arr,
            temp_handle,
            status
        )
        
        if status[0] != 0:
            raise RuntimeError(f"mb_init_potential failed with status {status[0]}")
        
        self.handle[0] = temp_handle[0]
    
    def init_from_params(self, ddb_file: str, sys_file: str = "", coeff_file: str = "",
                        ncell: Tuple[int, int, int] = (1, 1, 1),
                        ngqpt: Tuple[int, int, int] = (1, 1, 1),
                        dipdip: int = 1):
        """
        Initialize potential from direct parameters (no .abi file needed).
        
        Args:
            ddb_file: Path to DDB file
            sys_file: Path to system XML file (optional, can read from DDB)
            coeff_file: Path to coefficient XML file (optional)
            ncell: Supercell dimensions [nx, ny, nz]
            ngqpt: q-point grid for phonon [nqx, nqy, nqz]
            dipdip: Dipole-dipole interaction flag (0=off, 1=on)
            
        Raises:
            RuntimeError: If initialization fails
        """
        ddb_bytes = ddb_file.encode('utf-8')
        sys_bytes = sys_file.encode('utf-8') if sys_file else b""
        coeff_bytes = coeff_file.encode('utf-8') if coeff_file else b""
        
        ncell_arr = self.ffi.new("int[3]", ncell)
        ngqpt_arr = self.ffi.new("int[3]", ngqpt)
        status = self.ffi.new("int*")
        
        # Use a temporary handle pointer
        temp_handle = self.ffi.new("void**")
        
        # Note: mb_init_potential_simple takes:
        #   sys_filename -> DDB file (ddb_file parameter)
        #   coeff_filename -> XML/coefficient file (sys_file or coeff_file parameter)
        # Use coeff_file if provided, otherwise sys_file, otherwise empty
        coeff_bytes_to_use = coeff_bytes if coeff_bytes else (sys_bytes if sys_bytes else b"")
        
        self.lib.mb_init_potential_simple(
            ddb_bytes, len(ddb_bytes),
            coeff_bytes_to_use, len(coeff_bytes_to_use),
            ncell_arr,
            ngqpt_arr,
            dipdip,
            temp_handle,
            status
        )
        
        if status[0] != 0:
            raise RuntimeError(f"mb_init_potential_simple failed with status {status[0]}")
        
        self.handle[0] = temp_handle[0]
    
    def evaluate(self, positions: np.ndarray, lattice: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Evaluate energy, forces, and stresses.
        
        Args:
            positions: Atomic positions in Bohr, shape (natom, 3)
            lattice: Lattice vectors in Bohr, shape (3, 3) - row vectors
            
        Returns:
            tuple: (energy (Hartree), forces (Hartree/Bohr) shape (natom,3), 
                   stresses (Hartree/Bohr^3) shape (6,) in Voigt order)
            
        Raises:
            RuntimeError: If not initialized or evaluation fails
        """
        if self.handle[0] == self.ffi.NULL:
            raise RuntimeError("Potential not initialized. Call init_from_abi_file or init_from_params first.")
        
        natom = positions.shape[0]
        
        # Ensure C-contiguous and correct dtype
        pos_flat = np.ascontiguousarray(positions.flatten(), dtype=np.float64)
        lat_flat = np.ascontiguousarray(lattice.flatten(), dtype=np.float64)
        
        energy = self.ffi.new("double*")
        forces = np.zeros(natom * 3, dtype=np.float64)
        stresses = np.zeros(6, dtype=np.float64)
        status = self.ffi.new("int*")
        
        # Get pointers to numpy arrays
        pos_ptr = self.ffi.cast("double*", pos_flat.ctypes.data)
        lat_ptr = self.ffi.cast("double*", lat_flat.ctypes.data)
        frc_ptr = self.ffi.cast("double*", forces.ctypes.data)
        strs_ptr = self.ffi.cast("double*", stresses.ctypes.data)
        
        self.lib.mb_evaluate(
            self.handle[0],
            pos_ptr,
            lat_ptr,
            natom,
            energy,
            frc_ptr,
            strs_ptr,
            status
        )
        
        if status[0] != 0:
            raise RuntimeError(f"mb_evaluate failed with status {status[0]}")
        
        return energy[0], forces.reshape((natom, 3)), stresses
    
    def free(self):
        """Free the underlying potential structure."""
        if self.handle[0] != self.ffi.NULL:
            status = self.ffi.new("int*")
            self.lib.mb_free_potential(self.handle[0], status)
            if status[0] != 0:
                print(f"Warning: mb_free_potential returned status {status[0]}")
            self.handle[0] = self.ffi.NULL
    
    def __enter__(self):
        """Context manager support."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup."""
        self.free()
        return False
    
    def __del__(self):
        """Destructor - ensure cleanup."""
        self.free()


# Alias for backward compatibility
MultibinitWrapperABI = MultibinitWrapperCFFI
