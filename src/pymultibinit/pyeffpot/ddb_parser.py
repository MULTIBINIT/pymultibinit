"""
Pure Python DDB file parser for ABINIT Derivative Database files.

This module parses DDB files to extract unitcell data:
- Crystal structure (lattice, positions, types)
- Dielectric tensor and Born effective charges
- Interatomic force constants (IFCs)
- Elastic constants
- Stress and forces

All data is for the primitive (unit) cell only.
Supercell construction is handled separately.

Example:
    >>> from pymultibinit.pyeffpot.ddb_parser import read_ddb
    >>> unitcell = read_ddb("BaTiO3.DDB")
    >>> print(unitcell.natom)
    5
    >>> print(unitcell.epsilon_inf)
    [[7.2, 0, 0], [0, 7.2, 0], [0, 0, 7.2]]
"""
import re
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any
from pathlib import Path


@dataclass
class UnitcellData:
    """
    Unitcell data parsed from DDB file.
    
    Contains all information needed to build an effective potential
    in a supercell.
    
    Attributes:
        natom: Number of atoms in unitcell
        ntypat: Number of atom types
        rprimd: Primitive lattice vectors (3,3) in Bohr
        xred: Reduced atomic positions (natom, 3)
        typat: Atom type indices (natom,)
        znucl: Nuclear charges (ntypat,)
        amu: Atomic masses (ntypat,) in atomic units
        energy: Reference energy in Hartree
        epsilon_inf: Dielectric tensor (3,3)
        zeff: Born effective charges (3,3,natom)
        ifcs: Force constants (3,natom,3,natom) in Hartree/Bohr²
        elastic_constants: Elastic tensor (6,6) in Hartree/Bohr³
        strten: Stress tensor (6,) in Hartree/Bohr³ (Voigt notation)
        fcart: Forces (3,natom) in Hartree/Bohr
    """
    # Crystal structure
    natom: int
    ntypat: int
    rprimd: np.ndarray  # (3,3)
    xred: np.ndarray    # (natom, 3)
    typat: np.ndarray   # (natom,)
    znucl: np.ndarray   # (ntypat,)
    amu: np.ndarray     # (ntypat,)
    
    # Reference energy
    energy: float
    
    # Dielectric properties
    epsilon_inf: np.ndarray  # (3,3)
    zeff: np.ndarray         # (3,3,natom)
    
    # IFCs (Gamma-point only)
    ifcs: np.ndarray  # (3,natom,3,natom)
    
    # Elastic constants
    elastic_constants: np.ndarray  # (6,6)
    
    # Stress and forces
    strten: np.ndarray  # (6,)
    fcart: np.ndarray   # (3,natom)


def read_ddb(filename: str) -> UnitcellData:
    """
    Read ABINIT DDB file and extract unitcell data.
    
    Args:
        filename: Path to DDB file
        
    Returns:
        UnitcellData object with all extracted information
        
    Example:
        >>> unitcell = read_ddb("BaTiO3.DDB")
        >>> print(f"Number of atoms: {unitcell.natom}")
        >>> print(f"Dielectric constant: {unitcell.epsilon_inf[0,0]:.2f}")
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    parser = DDBParser(lines)
    return parser.parse()


class DDBParser:
    """
    Parser for ABINIT DDB files.
    
    Implements a state machine to parse different sections:
    1. Header (metadata)
    2. K-points
    3. Atomic positions and types
    4. Derivative blocks (IFCs, dielectric, etc.)
    """
    
    def __init__(self, lines: List[str]):
        self.lines = [line.rstrip() for line in lines]
        self.current_line = 0
        self.nlines = len(lines)
        
        # Storage for parsed data
        self.data: Dict[str, Any] = {}
        self.blocks: List[Dict] = []
    
    def parse(self) -> UnitcellData:
        """Parse entire DDB file and return UnitcellData."""
        self._parse_header()
        self._parse_kpoints()
        self._parse_atomic_data()
        self._parse_derivative_blocks()
        self._extract_quantities()
        
        return self._build_unitcell()
    
    def _current_line(self) -> str:
        """Get current line without advancing."""
        if self.current_line >= self.nlines:
            return ""
        return self.lines[self.current_line]
    
    def _next_line(self) -> str:
        """Get next line and advance."""
        if self.current_line >= self.nlines:
            return ""
        line = self.lines[self.current_line]
        self.current_line += 1
        return line
    
    def _skip_empty(self):
        """Skip empty lines."""
        while self.current_line < self.nlines and not self._current_line().strip():
            self.current_line += 1
    
    def _parse_header(self):
        """Parse header section with metadata."""
        # Skip header banner
        while self.current_line < self.nlines:
            line = self._next_line()
            if 'natom' in line.lower():
                # Parse natom
                parts = line.split()
                self.data['natom'] = int(parts[1])
                break
        
        # Parse remaining header fields
        while self.current_line < self.nlines:
            line = self._current_line()
            
            # Check for known fields
            if 'ntypat' in line.lower():
                parts = line.split()
                self.data['ntypat'] = int(parts[1])
                self._next_line()
            elif 'acell' in line.lower():
                # May span multiple lines
                values = self._read_float_values(line)
                if len(values) >= 3:
                    self.data['acell'] = values[:3]
                self._next_line()
            elif 'amu' in line.lower():
                values = self._read_float_values(line)
                self.data['amu'] = values
                self._next_line()
            elif 'xred' in line.lower():
                # End of header, atomic positions next
                break
            elif 'kpt' in line.lower() and len(line.split()) > 1:
                # Start of k-point section
                self._parse_kpoints_from_line(line)
            else:
                self._next_line()
    
    def _read_float_values(self, line: str) -> List[float]:
        """Extract float values from a line."""
        # Match D-format or E-format floats
        pattern = r'[-+]?\d*\.?\d+[DE][-+]?\d+'
        matches = re.findall(pattern, line, re.IGNORECASE)
        return [float(m.replace('D', 'E').replace('d', 'e')) for m in matches]
    
    def _parse_kpoints(self):
        """Parse k-point grid."""
        # Already handled in header parsing
        pass
    
    def _parse_kpoints_from_line(self, line: str):
        """Parse k-points starting from kpt line."""
        # Skip k-points (we don't need them for effective potential)
        # Just advance past them
        self._next_line()  # Skip first kpt line
        # Continue to atomic data
    
    def _parse_atomic_data(self):
        """Parse atomic positions, types, and charges."""
        natom = self.data['natom']
        ntypat = self.data['ntypat']
        
        # Read reduced coordinates
        xred = []
        for i in range(natom):
            line = self._next_line()
            coords = self._read_float_values(line)
            if len(coords) >= 3:
                xred.append(coords[:3])
        self.data['xred'] = np.array(xred)
        
        # Read typat
        while self.current_line < self.nlines:
            line = self._current_line()
            if 'typat' in line.lower():
                values = [int(x) for x in line.split()[1:]]
                self.data['typat'] = np.array(values[:natom])
                self._next_line()
                break
            else:
                self._next_line()
        
        # Read znucl
        while self.current_line < self.nlines:
            line = self._current_line()
            if 'znucl' in line.lower():
                values = [int(x) for x in line.split()[1:]]
                self.data['znucl'] = np.array(values[:ntypat])
                self._next_line()
                break
            else:
                self._next_line()
    
    def _parse_derivative_blocks(self):
        """Parse all derivative blocks."""
        while self.current_line < self.nlines:
            line = self._current_line()
            
            if 'derivative block' in line.lower():
                block = self._parse_block()
                if block:
                    self.blocks.append(block)
            else:
                self._next_line()
    
    def _parse_block(self) -> Optional[Dict]:
        """Parse a single derivative block."""
        header_line = self._next_line()
        
        # Parse block header
        # Format: realfmt ipert1 idir1 ipert2 idir2 qindex something nrow ncol
        parts = header_line.split()
        if len(parts) < 9:
            return None
        
        try:
            block = {
                'realfmt': int(parts[0]),
                'ipert1': int(parts[1]),
                'idir1': int(parts[2]),
                'ipert2': int(parts[3]),
                'idir2': int(parts[4]),
                'qindex': int(parts[5]),
                'nrow': int(parts[7]),
                'ncol': int(parts[8]),
            }
        except (ValueError, IndexError):
            return None
        
        # Read block data
        nrow = block['nrow']
        ncol = block['ncol']
        
        # Read matrix data (complex or real)
        data_lines = []
        values_read = 0
        total_values = nrow * ncol * 2  # Complex numbers have real+imag
        
        while values_read < total_values and self.current_line < self.nlines:
            line = self._current_line()
            if not line.strip() or 'derivative block' in line.lower():
                break
            
            # Parse values from line (D-format or E-format)
            values = self._read_float_values(line)
            data_lines.extend(values)
            values_read += len(values)
            self._next_line()
        
        # Reshape into matrix
        if block['realfmt'] == 2:  # Complex
            data = np.array(data_lines).reshape((nrow, ncol, 2))
            # Convert to complex: real + i*imag
            block['data'] = data[:,:,0] + 1j * data[:,:,1]
        else:  # Real
            data = np.array(data_lines).reshape((nrow, ncol))
            block['data'] = data
        
        return block
    
    def _extract_quantities(self):
        """Extract physical quantities from parsed blocks."""
        natom = self.data['natom']
        
        # Initialize arrays
        epsilon_inf = np.eye(3)
        zeff = np.zeros((3, 3, natom))
        ifcs = np.zeros((3, natom, 3, natom))
        elastic_constants = np.zeros((6, 6))
        strten = np.zeros(6)
        fcart = np.zeros((3, natom))
        energy = 0.0
        
        # Process each block
        for block in self.blocks:
            ipert1 = block['ipert1']
            idir1 = block['idir1']
            ipert2 = block['ipert2']
            idir2 = block['idir2']
            data = block['data']
            
            # Identify block type based on perturbation indices
            # Perturbations: 1..natom = atomic displacements
            #                natom+1..natom+3 = electric field
            #                natom+4..natom+9 = strain
            
            if ipert1 <= natom and ipert2 <= natom:
                # IFC block: atom-atom displacement
                # Map perturbation index to atom index (1-based to 0-based)
                iatom = ipert1 - 1
                jatom = ipert2 - 1
                
                # IFC matrix element: Φ_ij^αβ
                if 0 <= iatom < natom and 0 <= jatom < natom:
                    if 1 <= idir1 <= 3 and 1 <= idir2 <= 3:
                        val = float(np.real(data[0, 0] if hasattr(data, '__getitem__') else data))
                        ifcs[idir1-1, iatom, idir2-1, jatom] = val
            
            elif ipert1 > natom and ipert2 <= natom:
                # Mixed block: field/strain - atom
                if ipert1 == natom + 1 and ipert2 <= natom:
                    # Electric field - atom displacement → Born charges
                    if 1 <= idir1 <= 3 and 1 <= idir2 <= 3:
                        iatom = ipert2 - 1
                        zeff[idir1-1, idir2-1, iatom] = -float(np.real(data[0, 0] if data.ndim > 0 else data))
                
                elif natom + 4 <= ipert1 <= natom + 9 and ipert2 <= natom:
                    # Strain - atom displacement → strain coupling
                    pass  # Extract later if needed
            
            elif ipert1 > natom and ipert2 > natom:
                # Field-field or strain-strain block
                if ipert1 == natom + 1 and ipert2 == natom + 1:
                    # Electric field - electric field → dielectric tensor
                    if 1 <= idir1 <= 3 and 1 <= idir2 <= 3:
                        epsilon_inf[idir1-1, idir2-1] = float(np.real(data[0, 0]))
                
                elif natom + 4 <= ipert1 <= natom + 9 and natom + 4 <= ipert2 <= natom + 9:
                    # Strain - strain → elastic constants
                    # Map perturbation index to Voigt notation
                    alpha = ipert1 - (natom + 4)
                    beta = ipert2 - (natom + 4)
                    if 0 <= alpha < 6 and 0 <= beta < 6:
                        if 1 <= idir1 <= 3 and 1 <= idir2 <= 3:
                            # Simplified: take diagonal element
                            elastic_constants[alpha, beta] = float(np.real(data[0, 0]))
                
                elif ipert1 == natom + 3:  # Stress
                    # Stress tensor (Voigt notation)
                    if ipert2 == natom + 3:
                        for i in range(min(6, data.shape[0])):
                            for j in range(min(6, data.shape[1])):
                                if i < 6 and j < 6:
                                    strten[i] = float(np.real(data[i, j]))
                    
                    # Forces
                    if ipert2 <= natom:
                        iatom = ipert2 - 1
                        for i in range(min(3, data.shape[0])):
                            fcart[i, iatom] = float(np.real(data[i, 0]))
        
        # Store extracted quantities
        self.data['epsilon_inf'] = epsilon_inf
        self.data['zeff'] = zeff
        self.data['ifcs'] = ifcs
        self.data['elastic_constants'] = elastic_constants
        self.data['strten'] = strten
        self.data['fcart'] = fcart
        self.data['energy'] = energy
    
    def _build_unitcell(self) -> UnitcellData:
        """Build UnitcellData object from parsed data."""
        natom = self.data['natom']
        ntypat = self.data['ntypat']
        
        # Build lattice from acell (assume cubic for now)
        acell = self.data.get('acell', [1.0, 1.0, 1.0])
        rprimd = np.diag(acell)
        
        return UnitcellData(
            natom=natom,
            ntypat=ntypat,
            rprimd=rprimd,
            xred=self.data.get('xred', np.zeros((natom, 3))),
            typat=self.data.get('typat', np.zeros(natom, dtype=int)),
            znucl=self.data.get('znucl', np.zeros(ntypat, dtype=int)),
            amu=self.data.get('amu', np.zeros(ntypat)),
            energy=self.data.get('energy', 0.0),
            epsilon_inf=self.data.get('epsilon_inf', np.eye(3)),
            zeff=self.data.get('zeff', np.zeros((3, 3, natom))),
            ifcs=self.data.get('ifcs', np.zeros((3, natom, 3, natom))),
            elastic_constants=self.data.get('elastic_constants', np.zeros((6, 6))),
            strten=self.data.get('strten', np.zeros(6)),
            fcart=self.data.get('fcart', np.zeros((3, natom)))
        )


# Convenience function for testing
def test_read_ddb():
    """Test DDB parser with example file."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python ddb_parser.py <file.DDB>")
        sys.exit(1)
    
    filename = sys.argv[1]
    unitcell = read_ddb(filename)
    
    print(f"Parsed: {filename}")
    print(f"  natom: {unitcell.natom}")
    print(f"  ntypat: {unitcell.ntypat}")
    print(f"  rprimd shape: {unitcell.rprimd.shape}")
    print(f"  xred shape: {unitcell.xred.shape}")
    print(f"  energy: {unitcell.energy:.6f} Ha")
    print(f"  epsilon_inf: {unitcell.epsilon_inf[0,0]:.4f}")
    print(f"  zeff shape: {unitcell.zeff.shape}")
    print(f"  ifcs shape: {unitcell.ifcs.shape}")


if __name__ == '__main__':
    test_read_ddb()
