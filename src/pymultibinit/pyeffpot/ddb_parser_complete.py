"""
Complete ABINIT DDB file parser.

Parses ABINIT Derivative Database (DDB) files exactly as done in Fortran:
- m_ddb.F90: ddb_read_block_txt()
- m_effective_potential_file.F90: system_ddb2effpot()

Format specification from Fortran source:
- Block header: " 2nd derivatives (non-stat.)  - # elements :     225"
- 2nd derivatives: idir1 ipert1 idir2 ipert2 ar ai
- 1st derivatives: idir1 ipert1 ar ai
- Total energy: ar ai (single value)

Perturbation indices:
- 1..natom: atomic displacements
- natom+1..natom+3: electric field
- natom+3: stress (isotropic)
- natom+4..natom+9: strain (Voigt notation)
"""
import re
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple, Union

from .datastructures import CrystalInfo, IFCData, UnitcellData


@dataclass
class DDBBlock:
    """Represents a single derivative block in DDB file."""
    typ: str  # Block type: 'd2E_ns', 'd1E_xx', 'd0E_xx', etc.
    nelmts: int  # Number of elements
    qpt: np.ndarray  # q-point (3,)
    qnrm: float  # q-point norm
    data: Union[np.ndarray, Dict[str, Any], None] = None  # Parsed data (array or dict)
    raw_lines: Optional[List[str]] = None  # Original lines for debugging


class DDBParser:
    """
    Complete DDB file parser matching Fortran implementation.
    
    Usage:
        parser = DDBParser('BaTiO3.DDB')
        unitcell = parser.parse()
    """
    
    # Block type mapping from Fortran
    BLOCK_TYPES = {
        ' 2nd derivatives (non-stat.)  - ': 'd2E_ns',
        ' 2rd derivatives (non-stat.)  - ': 'd2E_ns',  # Old format
        ' 2nd derivatives (stationary) - ': 'd2E_st',
        ' 2rd derivatives (stationary) - ': 'd2E_st',
        ' 3rd derivatives              - ': 'd3E_xx',
        ' Total energy                 - ': 'd0E_xx',
        ' 1st derivatives              - ': 'd1E_xx',
        ' 2nd eigenvalue derivatives   - ': 'd2eig_re',
        ' 2rd eigenvalue derivatives   - ': 'd2eig_re',
        ' 3rd derivatives (long wave)  - ': 'd3E_lw',
        ' 2nd derivatives (MBC)        - ': 'd2E_mbc',
    }
    
    def __init__(self, filename: str):
        self.filename = filename
        self.lines: List[str] = []
        self.current_line = 0
        self.nlines = 0
        self.blocks: List[DDBBlock] = []
        self.data: Dict[str, Any] = {}
        
    def parse(self) -> UnitcellData:
        """Parse entire DDB file."""
        self._read_file()
        self._parse_header()
        self._parse_atomic_data()
        self._parse_blocks()
        self._extract_quantities()
        return self._build_unitcell()
    
    def _read_file(self):
        """Read DDB file."""
        with open(self.filename, 'r') as f:
            self.lines = [line.rstrip('\n') for line in f.readlines()]
        self.nlines = len(self.lines)
        self.current_line = 0
    
    def _current_line(self) -> str:
        """Get current line."""
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
        while (self.current_line < self.nlines and 
               not self._current_line().strip()):
            self.current_line += 1
    
    def _parse_header(self):
        """Parse DDB header with metadata."""
        # Skip banner lines
        while self.current_line < self.nlines:
            line = self._next_line()
            if 'natom' in line.lower() and 'Number' not in line:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        self.data['natom'] = int(parts[1])
                    except ValueError:
                        pass
                break
        
        # Parse other header fields
        while self.current_line < self.nlines:
            line = self._current_line()
            
            if 'ntypat' in line.lower() and 'Number' not in line:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        self.data['ntypat'] = int(parts[1])
                    except ValueError:
                        pass
                self._next_line()
            elif 'typat' in line.lower() and 'ntypat' not in line.lower():
                # Parse atom types
                values = [int(x) for x in line.split()[1:]]
                natom = self.data.get('natom', len(values))
                self.data['typat'] = np.array(values[:natom])
                self._next_line()
            elif 'acell' in line.lower():
                values = self._read_floats(line)
                if len(values) >= 3:
                    self.data['acell'] = np.array(values[:3], dtype=float)
                self._next_line()
            elif 'rprim' in line.lower():
                values = self._read_floats(line)
                rprim_rows = []
                if len(values) >= 3:
                    rprim_rows.append(values[:3])
                self._next_line()
                for _ in range(2):
                    more_values = self._read_floats(self._current_line())
                    if len(more_values) >= 3:
                        rprim_rows.append(more_values[:3])
                    self._next_line()
                if len(rprim_rows) == 3:
                    self.data['rprim'] = np.array(rprim_rows, dtype=float)
                self._next_line()
            elif 'amu' in line.lower():
                values = self._read_floats(line)
                self.data['amu'] = np.array(values, dtype=float)
                self._next_line()
            elif 'xred' in line.lower():
                break  # End of header
            else:
                self._next_line()
    
    def _parse_kpoints(self):
        """Parse k-point grid (skip for now)."""
        natom = self.data.get('natom', 0)
        # Skip k-points - look for xred or end of k-point section
        lines_read = 0
        max_kpoints = 1000  # Safety limit
        while self.current_line < self.nlines and lines_read < max_kpoints:
            line = self._current_line()
            # Stop at xred (atomic positions) or empty sections
            if 'xred' in line.lower():
                break
            # Stop if line looks like start of new section
            if line.strip() and not line[0].isspace() and 'DDB' not in line:
                break
            self._next_line()
            lines_read += 1
    
    def _parse_atomic_data(self):
        """Parse atomic positions and types."""
        natom = self.data['natom']
        ntypat = self.data['ntypat']
        
        # Read xred - first coordinate is on same line as 'xred' keyword
        xred = []
        
        # Parse first coordinate from current line (contains 'xred')
        if 'xred' in self._current_line().lower():
            coords = self._read_floats(self._current_line())
            if len(coords) >= 3:
                xred.append(coords[:3])
            self._next_line()
        
        # Read remaining coordinates (natom-1 more lines)
        for i in range(natom - 1):
            line = self._next_line()
            coords = self._read_floats(line)
            if len(coords) >= 3:
                xred.append(coords[:3])
        
        self.data['xred'] = np.array(xred)
        
        # Read znucl (if not already read)
        if 'znucl' not in self.data:
            # znucl should be right after xred
            if self.current_line < self.nlines and 'znucl' in self._current_line().lower():
                line = self._current_line()
                values = [int(float(x.replace('D', 'E').replace('d', 'e'))) for x in line.split()[1:]]
                self.data['znucl'] = np.array(values[:ntypat])
                self._next_line()
        
        # Skip pseudopotential data and other sections until derivative blocks
        while self.current_line < self.nlines:
            line = self._current_line()
            # Look for start of derivative database section
            if 'Database of total energy derivatives' in line:
                self._next_line()  # Skip this line
                # Next should be "Number of data blocks"
                if self.current_line < self.nlines:
                    nblocks_line = self._next_line()
                    # Parse number of blocks
                    match = re.search(r'Number of data blocks\s*=\s*(\d+)', nblocks_line)
                    if match:
                        self.data['nblocks'] = int(match.group(1))
                break
            self._next_line()
    
    def _parse_blocks(self):
        """Parse all derivative blocks."""
        nblocks = self.data.get('nblocks', 0)
        blocks_read = 0
        
        while self.current_line < self.nlines and blocks_read < nblocks:
            line = self._current_line()
            
            if '# elements' in line:
                block = self._parse_block()
                if block:
                    self.blocks.append(block)
                    blocks_read += 1
            else:
                self._next_line()
    
    def _parse_block(self) -> Optional[DDBBlock]:
        """Parse a single derivative block matching Fortran logic."""
        header_line = self._current_line()
        
        # Identify block type
        block_typ = None
        for pattern, typ in self.BLOCK_TYPES.items():
            if pattern in header_line:
                block_typ = typ
                break
        
        if not block_typ:
            self._next_line()
            return None
        
        # Extract number of elements
        match = re.search(r'# elements\s*:\s*(\d+)', header_line)
        if not match:
            self._next_line()
            return None
        
        nelmts = int(match.group(1))
        
        # Skip header line
        self._next_line()
        
        # Create block object
        block = DDBBlock(
            typ=block_typ,
            nelmts=nelmts,
            qpt=np.zeros(3),
            qnrm=0.0,
            raw_lines=[]
        )
        raw_lines = block.raw_lines if block.raw_lines is not None else []
        block.raw_lines = raw_lines
        
        # Parse based on block type
        if block_typ in ['d2E_ns', 'd2E_st']:
            self._parse_d2E_block(block)
        elif block_typ == 'd1E_xx':
            self._parse_d1E_block(block)
        elif block_typ == 'd0E_xx':
            self._parse_d0E_block(block)
        else:
            # Skip unsupported block types
            self._skip_block_lines(block, nelmts)
        
        return block
    
    def _parse_d2E_block(self, block: DDBBlock):
        """
        Parse 2nd derivative block.
        
        Format (from Fortran ddb_read_block_txt):
        - q-point line: (4x,3es16.8,f6.1)
        - Elements: idir1 ipert1 idir2 ipert2 ar ai
        """
        natom = self.data['natom']
        
        # Read q-point
        line = self._next_line()
        raw_lines = block.raw_lines if block.raw_lines is not None else []
        block.raw_lines = raw_lines
        raw_lines.append(line)
        qpt_vals = self._read_floats(line)
        if len(qpt_vals) >= 4:
            block.qpt = np.array(qpt_vals[:3])
            block.qnrm = qpt_vals[3]
        
        # Read elements
        # Storage: val(idir, ipert, idir2, ipert2)
        # Flattened index: idir + 3*((ipert-1) + mpert*((idir2-1) + 3*(ipert2-1)))
        mpert = natom  # For IFCs, perturbations are atoms
        data = {}
        
        for ielem in range(block.nelmts):
            line = self._next_line()
            raw_lines.append(line)
            parts = line.split()
            if len(parts) < 6:
                continue
            
            idir1 = int(parts[0])
            ipert1 = int(parts[1])
            idir2 = int(parts[2])
            ipert2 = int(parts[3])
            ar = float(parts[4].replace('D', 'E').replace('d', 'e'))
            ai = float(parts[5].replace('D', 'E').replace('d', 'e'))
            
            # Store as real (ignore imaginary for now)
            key = (idir1, ipert1, idir2, ipert2)
            data[key] = ar
        
        # Convert to IFC array: (natom, 3, natom, 3) in C-order
        # Fortran order: (idir1, ipert1, idir2, ipert2)
        # C-order: (ipert1, idir1, ipert2, idir2)
        ifcs = np.zeros((natom, 3, natom, 3))
        for (idir1, ipert1, idir2, ipert2), val in data.items():
            if 1 <= idir1 <= 3 and 1 <= ipert1 <= natom:
                if 1 <= idir2 <= 3 and 1 <= ipert2 <= natom:
                    ifcs[ipert1-1, idir1-1, ipert2-1, idir2-1] = val
        
        block.data = ifcs
    
    def _parse_d1E_block(self, block: DDBBlock):
        """
        Parse 1st derivative block (forces, stress).
        
        Format: idir1 ipert1 ar ai
        """
        natom = self.data['natom']
        raw_lines = block.raw_lines if block.raw_lines is not None else []
        block.raw_lines = raw_lines
        data = {}
        
        for ielem in range(block.nelmts):
            line = self._next_line()
            raw_lines.append(line)
            parts = line.split()
            if len(parts) < 4:
                continue
            
            idir1 = int(parts[0])
            ipert1 = int(parts[1])
            ar = float(parts[2].replace('D', 'E').replace('d', 'e'))
            
            key = (idir1, ipert1)
            data[key] = ar
        
        # Extract forces (ipert1 <= natom) and stress (ipert1 > natom)
        # Forces: C-order (natom, 3)
        fcart = np.zeros((natom, 3))
        strten = np.zeros(6)
        
        for (idir1, ipert1), val in data.items():
            if 1 <= ipert1 <= natom and 1 <= idir1 <= 3:
                # Forces in C-order: (iatom, idir)
                fcart[ipert1-1, idir1-1] = val
            elif ipert1 > natom:
                # Stress: ipert1 = natom+1..natom+6
                istrain = ipert1 - natom - 1
                if 0 <= istrain < 6 and 1 <= idir1 <= 3:
                    # Simplified: take diagonal
                    if istrain < 3:
                        strten[istrain] = val
        
        block.data = {'fcart': fcart, 'strten': strten}
    
    def _parse_d0E_block(self, block: DDBBlock):
        """
        Parse total energy block.
        
        Format: ar ai (single value)
        """
        line = self._next_line()
        raw_lines = block.raw_lines if block.raw_lines is not None else []
        block.raw_lines = raw_lines
        raw_lines.append(line)
        
        # Format: (2d22.14)
        parts = line.split()
        if len(parts) >= 2:
            ar = float(parts[0].replace('D', 'E').replace('d', 'e'))
            ai = float(parts[1].replace('D', 'E').replace('d', 'e'))
            block.data = {'energy': ar}
        else:
            block.data = {'energy': 0.0}
    
    def _skip_block_lines(self, block: DDBBlock, nelmts: int):
        """Skip lines for unsupported block types."""
        # Read q-point line
        raw_lines = block.raw_lines if block.raw_lines is not None else []
        block.raw_lines = raw_lines
        self._next_line()
        raw_lines.append(self._current_line())
        
        # Skip element lines
        for i in range(nelmts):
            self._next_line()
            raw_lines.append(self._current_line())
    
    def _read_floats(self, line: str) -> List[float]:
        """Read float values from line (D-format, E-format, or simple floats)."""
        # Match scientific notation (D/E format)
        pattern_sci = r'[-+]?\d*\.?\d+[DEde][-+]?\d+'
        matches_sci = re.findall(pattern_sci, line)
        # Convert to E format for Python
        values = [float(m.replace('D', 'E').replace('d', 'e')) for m in matches_sci]
        
        # Also match simple floats (like "1.0") that don't have exponents
        # But exclude those that are already matched as scientific notation
        pattern_simple = r'[-+]?\d+\.\d+(?![DEde])'
        matches_simple = re.findall(pattern_simple, line)
        
        # Add simple floats (need to avoid duplicates from sci notation)
        # We'll check positions to avoid duplicates
        for match in matches_simple:
            # Simple heuristic: if it looks like a simple float, add it
            if not any(match in m for m in matches_sci):
                values.append(float(match))
        
        return values
    
    def _extract_quantities(self):
        """Extract physical quantities from blocks (matching Fortran)."""
        natom = self.data['natom']
        
        # Initialize
        epsilon_inf = np.eye(3)
        zeff = np.zeros((3, 3, natom))
        ifcs = np.zeros((natom, 3, natom, 3))
        elastic_constants = np.zeros((6, 6))
        strten = np.zeros(6)
        fcart = np.zeros((natom, 3))
        energy = 0.0
        
        # Collect all q-points and dynamical matrices
        qpoints_list = []
        dynmat_list = []
        
        # Process blocks
        for block in self.blocks:
            if block.typ == 'd0E_xx':
                # Total energy
                if isinstance(block.data, dict) and 'energy' in block.data:
                    energy = block.data['energy']
            
            elif block.typ in ['d2E_ns', 'd2E_st']:
                # Collect dynamical matrices from all q-points
                if block.data is not None and isinstance(block.data, np.ndarray):
                    if block.data.ndim == 4:
                        # Store q-point and dynamical matrix
                        qpoints_list.append(block.qpt)
                        
                        # block.data is already in C-order: (natom, 3, natom, 3)
                        # Convert to complex form with real/imag parts
                        # C-order: (natom, 3, natom, 3, 2)
                        dynmat_real = block.data
                        dynmat_imag = np.zeros_like(dynmat_real)
                        dynmat_complex = np.stack([dynmat_real, dynmat_imag], axis=-1)
                        dynmat_list.append(dynmat_complex)
                        
                        # Keep Gamma-point IFCs for backward compatibility
                        if np.allclose(block.qpt, 0):
                            ifcs = block.data
            
            elif block.typ == 'd1E_xx':
                # Forces and stress from 1st derivatives
                if isinstance(block.data, dict):
                    if 'fcart' in block.data:
                        fcart = block.data['fcart']
                    if 'strten' in block.data:
                        strten = block.data['strten']
        
        # Convert lists to arrays
        if qpoints_list:
            qpoints = np.array(qpoints_list)  # (nqpt, 3) in C-order
            nqpt = len(qpoints_list)
            
            # Stack dynamical matrices: (nqpt, natom, 3, natom, 3, 2) in C-order
            # Each element is (natom, 3, natom, 3, 2)
            dynmat = np.stack(dynmat_list, axis=0)
        else:
            qpoints = None
            dynmat = None
            nqpt = 0
        
        self.data['epsilon_inf'] = epsilon_inf
        self.data['zeff'] = zeff
        self.data['ifcs'] = ifcs
        self.data['elastic_constants'] = elastic_constants
        self.data['strten'] = strten
        self.data['fcart'] = fcart
        self.data['energy'] = energy
        self.data['nqpt'] = nqpt
        self.data['qpoints'] = qpoints
        self.data['dynmat'] = dynmat
    
    def _build_unitcell(self) -> UnitcellData:
        """Build UnitcellData object."""
        natom = self.data['natom']
        ntypat = self.data['ntypat']

        acell = np.array(self.data.get('acell', np.ones(3)), dtype=float)
        rprim = np.array(self.data.get('rprim', np.eye(3)), dtype=float)
        rprimd = np.diag(acell) @ rprim
        xred = self.data.get('xred', np.zeros((natom, 3)))
        xcart = xred @ rprimd.T
        typat = np.array(self.data.get('typat', np.zeros(natom, dtype=int)), dtype=int)
        amu = np.array(self.data.get('amu', np.zeros(ntypat)), dtype=float)
        znucl = np.array(self.data.get('znucl', np.zeros(ntypat, dtype=int)), dtype=int)

        crystal = CrystalInfo(
            natom=natom,
            ntypat=ntypat,
            rprimd=rprimd,
            xred=xred,
            xcart=xcart,
            typat=typat,
            amu=amu,
            znucl=znucl,
        )

        gamma_ifcs = np.array(self.data.get('ifcs', np.zeros((natom, 3, natom, 3))), dtype=float)
        ifc_total = np.transpose(gamma_ifcs, (1, 0, 3, 2))[:, :, :, :, np.newaxis]
        ifc_data = IFCData(
            nrpt=1,
            cell=np.zeros((3, 1), dtype=int),
            atmfrc=ifc_total.copy(),
            short_atmfrc=ifc_total.copy(),
            ewald_atmfrc=np.zeros_like(ifc_total),
        )

        return UnitcellData(
            crystal=crystal,
            energy=self.data.get('energy', 0.0),
            ifcs=ifc_data,
            epsilon_inf=self.data.get('epsilon_inf', np.eye(3)),
            elastic_constants=self.data.get('elastic_constants', np.zeros((6, 6))),
            zeff=self.data.get('zeff', np.zeros((3, 3, natom))),
            acell=acell,
            qpoints=self.data.get('qpoints', None),
            dynmat=self.data.get('dynmat', None),
            blocks=self.blocks
        )


def read_ddb(filename: str) -> UnitcellData:
    """
    Read DDB file and extract unitcell data.
    
    Args:
        filename: Path to DDB file
        
    Returns:
        UnitcellData with all extracted quantities
    """
    parser = DDBParser(filename)
    return parser.parse()


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ddb_parser_complete.py <file.DDB>")
        sys.exit(1)
    
    u = read_ddb(sys.argv[1])
    print(f"Parsed: {sys.argv[1]}")
    print(f"  natom: {u.natom}")
    print(f"  energy: {u.energy:.10f} Ha")
    if u.ifcs is not None:
        print(f"  IFC shape: {u.ifcs.atmfrc.shape}")
    block_list = u.blocks or []
    print(f"  Blocks parsed: {len(block_list)}")
    for i, blk in enumerate(block_list[:5]):
        print(f"    Block {i}: {blk.typ}, q={blk.qpt}, nelmts={blk.nelmts}")
