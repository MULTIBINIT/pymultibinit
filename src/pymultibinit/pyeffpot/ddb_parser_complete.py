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
- natom+2: electric field in ABINIT DDB response blocks
- natom+3: diagonal homogeneous strain, idir=1..3 -> xx, yy, zz
- natom+4: shear homogeneous strain, idir=1..3 -> yz, xz, xy
"""
import re
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple, Union

from .datastructures import CrystalInfo, IFCData, UnitcellData
from .symmetry import build_atom_mapping, expand_zeff_by_symmetry


def _strain_perturbation_to_voigt(idir: int, ipert: int, natom: int) -> Optional[int]:
    """Map ABINIT DDB homogeneous-strain perturbations to Voigt order."""
    if ipert == natom + 3 and 1 <= idir <= 3:
        return idir - 1
    if ipert == natom + 4 and 1 <= idir <= 3:
        return idir + 2
    return None


@dataclass
class DDBBlock:
    """Represents a single derivative block in DDB file."""
    typ: str  # Block type: 'd2E_ns', 'd1E_xx', 'd0E_xx', etc.
    nelmts: int  # Number of elements
    qpt: np.ndarray  # q-point (3,)
    qnrm: float  # q-point norm
    data: Union[np.ndarray, Dict[str, Any], None] = None  # Parsed data (array or dict)
    raw_lines: Optional[List[str]] = None  # Original lines for debugging
    derivatives: Optional[Dict[Tuple[int, int, int, int], complex]] = None


class DDBParser:
    """
    Complete DDB file parser matching Fortran implementation.
    
    Usage:
        parser = DDBParser('BaHfO3.DDB')
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
            elif 'nsym' in line.lower() and line.strip().startswith('nsym'):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        self.data['nsym'] = int(parts[1])
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
            elif line.strip().startswith('symrel'):
                # Parse all symmetry operations (nsym of them). ABINIT writes
                # symrel in Fortran column-major order.
                nsym = self.data.get('nsym', 0)
                symrel_list = []
                # First sym op may be on same line as 'symrel'
                vals = [int(x) for x in line.split()[1:]]
                if len(vals) >= 9:
                    symrel_list.append(np.array(vals[:9]).reshape(3, 3, order='F'))
                self._next_line()
                while len(symrel_list) < nsym and self.current_line < self.nlines:
                    next_line = self._current_line()
                    try:
                        vals = [int(x) for x in next_line.split()]
                        if len(vals) >= 9:
                            symrel_list.append(np.array(vals[:9]).reshape(3, 3, order='F'))
                            self._next_line()
                        else:
                            break
                    except ValueError:
                        break
                if symrel_list:
                    self.data['symrel'] = np.array(symrel_list, dtype=int)
            elif line.strip().startswith('tnons'):
                nsym = self.data.get('nsym', 0)
                tnons_list = []
                values = self._read_floats(line)
                if len(values) >= 3:
                    tnons_list.append(values[:3])
                self._next_line()
                while len(tnons_list) < nsym and self.current_line < self.nlines:
                    next_line = self._current_line()
                    try:
                        values = self._read_floats(next_line)
                        if len(values) >= 3:
                            tnons_list.append(values[:3])
                            self._next_line()
                        else:
                            break
                    except ValueError:
                        break
                if tnons_list:
                    self.data['tnons'] = np.array(tnons_list, dtype=float)
            elif 'xred' in line.lower():
                break  # End of header
            else:
                self._next_line()
    
    def _parse_kpoints(self):
        """Parse k-point grid (skip for now)."""
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
        
        # Read zion (ionic charges for Born effective charges)
        if 'zion' not in self.data:
            if self.current_line < self.nlines and 'zion' in self._current_line().lower():
                line = self._current_line()
                values = [float(x.replace('D', 'E').replace('d', 'e')) for x in line.split()[1:]]
                self.data['zion'] = np.array(values[:ntypat])
                self._next_line()
        
        # Skip pseudopotential data and other sections until derivative blocks
        while self.current_line < self.nlines:
            line = self._current_line()
            # Look for start of derivative database section
            if 'Database of total energy derivatives' in line:
                self._next_line()  # Skip this line
                self._skip_empty()
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
            
            # Store complex value
            key = (idir1, ipert1, idir2, ipert2)
            data[key] = ar + 1j * ai
        block.derivatives = data
        
        # Convert to IFC array: (natom, 3, natom, 3) complex
        dm_complex = np.zeros((natom, 3, natom, 3), dtype=complex)
        for (idir1, ipert1, idir2, ipert2), val in data.items():
            if 1 <= idir1 <= 3 and 1 <= ipert1 <= natom:
                if 1 <= idir2 <= 3 and 1 <= ipert2 <= natom:
                    dm_complex[ipert1-1, idir1-1, ipert2-1, idir2-1] = val
        
        block.data = dm_complex
    
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
        fred = np.zeros((natom, 3))
        strten = np.zeros(6)
        
        for (idir1, ipert1), val in data.items():
            if 1 <= ipert1 <= natom and 1 <= idir1 <= 3:
                # Store reduced forces
                fred[ipert1-1, idir1-1] = val
            elif ipert1 > natom:
                # Stress: ipert1 = natom+1..natom+6
                istrain = ipert1 - natom - 1
                if 0 <= istrain < 6 and 1 <= idir1 <= 3:
                    # Simplified: take diagonal
                    if istrain < 3:
                        strten[istrain] = val
        
        block.data = {'fred': fred, 'strten': strten}
    
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
        """Extract physical quantities from blocks (matching ABINIT cart29)."""
        natom = self.data['natom']
        typat = self.data['typat']
        
        # Get ionic charges from zion (if available) or znucl
        if 'zion' in self.data:
            zion = np.array(self.data['zion'])
        else:
            zion = np.array(self.data['znucl'])
        
        # Unit cell volume and lattice vectors
        acell = np.array(self.data.get('acell', np.ones(3)))
        rprim = np.array(self.data.get('rprim', np.eye(3)))
        rprimd = np.diag(acell) @ rprim
        ucvol = np.abs(np.linalg.det(rprimd))
        
        # Pre-compute transformation matrices matching ABINIT cart29
        # In ABINIT, rprimd has vectors in COLUMNS. Here rprimd has vectors in ROWS (A).
        # mat_MP (phonon): V_cart = rprimd^-T * V_red
        # mat_ME (electric field): V_cart = rprimd^T * V_red / (2pi)
        inv_rprimd_T = np.linalg.inv(rprimd).T
        mat_MP = inv_rprimd_T
        mat_ME = rprimd.T / (2.0 * np.pi)
        
        # Initialize
        epsilon_inf = np.eye(3)
        zeff = np.zeros((natom, 3, 3))
        ifcs = np.zeros((natom, 3, natom, 3))
        elastic_constants = np.zeros((6, 6))
        strain_coupling = np.zeros((6, 3, natom))
        strten = np.zeros(6)
        fcart = np.zeros((natom, 3))
        energy = 0.0
        
        # Collect all q-points and transformed dynamical matrices
        qpoints_list = []
        dynmat_list = []
        
        # Process blocks
        for block in self.blocks:
            if block.typ == 'd0E_xx':
                # Total energy
                if isinstance(block.data, dict) and 'energy' in block.data:
                    energy = block.data['energy']
            
            elif block.typ in ['d2E_ns', 'd2E_st']:
                if block.data is not None and isinstance(block.data, np.ndarray):
                    if block.data.ndim == 4:
                        phi_red = block.data
                        phi_cart = np.zeros((natom, 3, natom, 3), dtype=complex)
                        for i in range(natom):
                            for j in range(natom):
                                # standard coordinate transform: rprimd^-T * Phi_red * rprimd^-1
                                phi_cart[i, :, j, :] = mat_MP @ phi_red[i, :, j, :] @ mat_MP.T
                        
                        qpoints_list.append(block.qpt)
                        dynmat_complex = np.stack([phi_cart.real, phi_cart.imag], axis=-1)
                        dynmat_list.append(dynmat_complex)
                        
                        if np.allclose(block.qpt, 0):
                            ifcs = phi_cart.real

                if np.allclose(block.qpt, 0) and block.derivatives:
                    block_elastic, block_strain_coupling = self._extract_strain_derivatives(block.derivatives, mat_MP)
                    elastic_constants += block_elastic
                    strain_coupling += block_strain_coupling
                
                # 2. Extract Born Effective Charges and Dielectric Tensor (only from q=0 block)
                if block.raw_lines and np.allclose(block.qpt, 0):
                    # Parse mixed terms from raw lines
                    d2E_atom_E = np.zeros((natom, 3, 3))
                    d2E_E_E = np.zeros((3, 3))
                    has_electric_terms = False
                    for line in block.raw_lines:
                        parts = line.split()
                        if len(parts) < 6:
                            continue
                        try:
                            idir1, ipert1, idir2, ipert2 = map(int, parts[:4])
                            ar = float(parts[4].replace('D', 'E').replace('d', 'e'))
                            
                            if 1 <= ipert1 <= natom and ipert2 == natom + 2:
                                d2E_atom_E[ipert1-1, idir1-1, idir2-1] = ar
                                has_electric_terms = True
                            elif ipert1 == natom + 2 and 1 <= ipert2 <= natom:
                                d2E_atom_E[ipert2-1, idir2-1, idir1-1] = ar
                                has_electric_terms = True
                            elif ipert1 == natom + 2 and ipert2 == natom + 2:
                                d2E_E_E[idir1-1, idir2-1] = ar
                                has_electric_terms = True
                        except (ValueError, IndexError):
                            continue

                    if not has_electric_terms:
                        continue

                    # Transform Born Effective Charges: Z_cart = trans_ME @ Z_red @ trans_MP.T
                    for iat in range(natom):
                        # Z_red[elec_dir, atom_dir] = d2E_atom_E[iat, atom_dir, elec_dir]
                        Z_red = d2E_atom_E[iat].T
                        Z_cart = mat_ME @ Z_red @ mat_MP.T
                        zeff[iat, :, :] = Z_cart
                        # Add ionic charge
                        for i in range(3):
                            zeff[iat, i, i] += zion[typat[iat]-1]
                            
                    # 3. Expand Born charges by symmetry if partial
                    symrel = self.data.get('symrel')
                    if symrel is not None:
                        # Build atom mapping
                        # If tnons not in DDB header, assume all are zero (standard for many DDBs)
                        tnons = self.data.get('tnons', np.zeros((len(symrel), 3)))
                        indsym = build_atom_mapping(self.data['xred'], symrel, tnons)
                        self.data['atom_mapping'] = indsym
                        zeff = expand_zeff_by_symmetry(zeff, symrel, indsym, rprimd)
                    
                    # 4. Enforce Charge Acoustic Sum Rule (CHASR)
                    zsum = np.sum(zeff, axis=0) / natom
                    for iat in range(natom):
                        zeff[iat, :, :] -= zsum
                    
                    # Transform Dielectric Tensor: eps = 1 - 4pi/vol * (ME @ alpha_red @ ME.T)
                    alpha_cart = mat_ME @ d2E_E_E @ mat_ME.T
                    epsilon_inf = np.eye(3) - 4.0 * np.pi / ucvol * alpha_cart
            
            elif block.typ == 'd1E_xx':
                if isinstance(block.data, dict):
                    if 'fred' in block.data:
                        fred = block.data['fred']
                        # Transform forces: f_cart = fred @ inv_rprimd
                        fcart = fred @ np.linalg.inv(rprimd)
                    if 'strten' in block.data:
                        strten = block.data['strten']
        
        # Store extracted quantities
        self.data['epsilon_inf'] = epsilon_inf
        self.data['zeff'] = zeff
        self.data['ifcs'] = ifcs
        self.data['elastic_constants'] = elastic_constants
        self.data['strain_coupling'] = strain_coupling
        self.data['strten'] = strten
        self.data['fcart'] = fcart
        self.data['energy'] = energy
        self.data['nqpt'] = len(qpoints_list)
        self.data['qpoints'] = np.array(qpoints_list) if qpoints_list else None
        self.data['dynmat'] = np.stack(dynmat_list, axis=0) if dynmat_list else None
        self.data['ngqpt'] = self._infer_ngqpt(np.array(qpoints_list)) if qpoints_list else np.array([1, 1, 1])

    def _extract_strain_derivatives(self, derivatives: Dict[Tuple[int, int, int, int], complex], mat_MP: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract Gamma elastic and internal-strain terms from raw DDB derivatives."""
        natom = self.data['natom']
        elastic_constants = np.zeros((6, 6), dtype=float)
        strain_coupling_red = np.zeros((6, 3, natom), dtype=float)

        for (idir1, ipert1, idir2, ipert2), value in derivatives.items():
            strain1 = _strain_perturbation_to_voigt(idir1, ipert1, natom)
            strain2 = _strain_perturbation_to_voigt(idir2, ipert2, natom)
            real_value = float(np.real(value))
            if strain1 is not None and strain2 is not None:
                elastic_constants[strain1, strain2] = real_value
                elastic_constants[strain2, strain1] = real_value
                continue

            if strain1 is not None and 1 <= ipert2 <= natom and 1 <= idir2 <= 3:
                strain_coupling_red[strain1, idir2 - 1, ipert2 - 1] = real_value
            elif strain2 is not None and 1 <= ipert1 <= natom and 1 <= idir1 <= 3:
                strain_coupling_red[strain2, idir1 - 1, ipert1 - 1] = real_value

        strain_coupling = np.zeros_like(strain_coupling_red)
        for alpha in range(6):
            for iatom in range(natom):
                strain_coupling[alpha, :, iatom] = mat_MP @ strain_coupling_red[alpha, :, iatom]
        return elastic_constants, strain_coupling
    
    def _infer_ngqpt(self, qpoints: np.ndarray) -> np.ndarray:
        """
        Infer ngqpt from irreducible q-points.
        
        The grid size along each direction is 1 / min_nonzero_step.
        For a 4x4x4 grid, q-coords are multiples of 0.25, so ngqpt=4.
        """
        ngqpt = np.ones(3, dtype=int)
        for idir in range(3):
            coords = np.abs(qpoints[:, idir])
            # Use only values in [0, 0.5]
            coords = coords[coords > 1e-6]
            # Fold 0.5 to also count as 1/2
            coords_fold = np.where(coords > 0.5 - 1e-6, 1.0 - coords, coords)
            coords_fold = coords_fold[coords_fold > 1e-6]
            if len(coords_fold) > 0:
                min_step = np.min(coords_fold)
                n = int(np.round(1.0 / min_step))
                ngqpt[idir] = max(n, 1)
        return ngqpt

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
        ifc_total = gamma_ifcs[:, :, :, :, np.newaxis]
        ifc_data = IFCData(
            nrpt=1,
            cell=np.zeros((3, 1), dtype=int),
            atmfrc=ifc_total.copy(),
            short_atmfrc=ifc_total.copy(),
            ewald_atmfrc=np.zeros_like(ifc_total),
        )

        symrel_raw = self.data.get('symrel', None)
        tnons_raw = self.data.get('tnons', None)

        return UnitcellData(
            crystal=crystal,
            energy=self.data.get('energy', 0.0),
            ifcs=ifc_data,
            epsilon_inf=self.data.get('epsilon_inf', np.eye(3)),
            elastic_constants=self.data.get('elastic_constants', np.zeros((6, 6))),
            strain_coupling=self.data.get('strain_coupling', np.zeros((6, 3, natom))),
            zeff=self.data.get('zeff', np.zeros((natom, 3, 3))),
            acell=acell,
            qpoints=self.data.get('qpoints', None),
            dynmat=self.data.get('dynmat', None),
            blocks=self.blocks,
            ngqpt=self.data.get('ngqpt', None),
            symrel=symrel_raw,
            tnons=tnons_raw,
            nqshft=1,
            q1shft=np.zeros((1, 3)),
            atom_mapping=self.data.get('atom_mapping', None),
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
