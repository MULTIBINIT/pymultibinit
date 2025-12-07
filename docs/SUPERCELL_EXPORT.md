# Supercell Structure Export Guide

This guide explains how to extract and export the MULTIBINIT internal supercell structure for visualization and verification.

## Overview

When MULTIBINIT is initialized, it creates an internal supercell structure based on the `ncell` parameter. This structure is used as the reference for all subsequent calculations. Being able to export this structure is useful for:

- **Verification**: Ensure the supercell matches expectations
- **Visualization**: View the structure in tools like VESTA, Ovito, etc.
- **Atom matching**: Understand how atoms are ordered internally
- **Debugging**: Check if atom matching is working correctly

## Quick Start

```python
from pymultibinit import MultibinitPotential

# Initialize potential
pot = MultibinitPotential.from_params(
    ddb_file="system_DDB",
    sys_file="system.xml",
    ncell=(2, 2, 2)
)

# Do at least one evaluation to capture reference structure
import numpy as np
positions = np.random.rand(40, 3) * 10  # Dummy positions
lattice = np.eye(3) * 10
pot.evaluate(positions, lattice)

# Export supercell
atoms = pot.export_supercell_to_ase()
atoms.set_chemical_symbols(['Ba']*8 + ['Ti']*8 + ['O']*24)  # Set correct symbols
atoms.write('supercell.cif')
```

## Methods

### 1. Get Supercell Structure (Raw Data)

```python
structure = pot.get_supercell_structure()
if structure is not None:
    positions, lattice, atomic_numbers = structure
    print(f"Supercell has {len(positions)} atoms")
```

**Returns:**
- `positions`: numpy array shape (natom, 3) in current units
- `lattice`: numpy array shape (3, 3) in current units
- `atomic_numbers`: None (not available from C API)

### 2. Export to ASE Atoms Object

```python
atoms = pot.export_supercell_to_ase()
```

**Returns:** ASE Atoms object with:
- Positions and cell in Angstrom (ASE convention)
- Chemical symbols set to 'X' (unknown)
- PBC enabled

### 3. Export to File

```python
pot.export_supercell_to_file('supercell.cif')
pot.export_supercell_to_file('POSCAR', format='vasp')
pot.export_supercell_to_file('supercell.xyz')
```

**Supported formats:** Any format supported by ASE (CIF, VASP, XYZ, JSON, etc.)

## Complete Examples

### Example 1: Export After Initialization

```python
from pymultibinit import MultibinitPotential
import numpy as np

# Initialize
pot = MultibinitPotential.from_params(
    ddb_file="BaTiO3_DDB",
    sys_file="BaTiO3.xml",
    ncell=(2, 2, 2)  # 2x2x2 supercell
)

# Need to evaluate once to set reference
# Use dummy positions initially
natom = 5 * 8  # 5 atoms/cell * 8 cells = 40 atoms
positions_dummy = np.zeros((natom, 3))
lattice = np.eye(3) * 7.5  # Approximate cubic lattice

pot.evaluate(positions_dummy, lattice)

# Now export
atoms = pot.export_supercell_to_ase()

# Set correct chemical symbols (BaTiO3 has 1 Ba, 1 Ti, 3 O per cell)
# For 2x2x2 supercell: 8 Ba, 8 Ti, 24 O
symbols = ['Ba'] * 8 + ['Ti'] * 8 + ['O'] * 24
atoms.set_chemical_symbols(symbols)

# Save to file
atoms.write('BaTiO3_supercell.cif')
print(f"Exported {len(atoms)} atoms to BaTiO3_supercell.cif")
```

### Example 2: Export with Explicit Reference Setting

```python
from pymultibinit import MultibinitPotential
from ase.io import read

# Initialize potential
pot = MultibinitPotential.from_params(
    ddb_file="system_DDB",
    sys_file="system.xml",
    ncell=(3, 3, 3)
)

# If you know the supercell structure, set it explicitly
# (e.g., from a reference calculation)
reference_atoms = read('reference_supercell.cif')
pot.set_reference_structure(
    reference_atoms.get_positions(),
    reference_atoms.get_cell()
)

# Export immediately (no evaluation needed)
atoms = pot.export_supercell_to_ase()
atoms.write('supercell.xyz')
```

### Example 3: Verify Supercell Construction

```python
from pymultibinit import MultibinitPotential
from ase.io import read

# Read primitive cell
primitive = read('primitive.cif')
print(f"Primitive cell: {len(primitive)} atoms")

# Initialize with supercell
pot = MultibinitPotential.from_params(
    ddb_file="system_DDB",
    sys_file="system.xml",
    ncell=(2, 2, 2)
)

# Evaluate with primitive cell positions (will be matched)
pot.evaluate(primitive.get_positions(), primitive.get_cell())

# Export supercell
supercell = pot.export_supercell_to_ase()
print(f"Supercell: {len(supercell)} atoms")

# Should be primitive * 2*2*2 = 8x
assert len(supercell) == len(primitive) * 8

# Set symbols based on primitive
primitive_symbols = primitive.get_chemical_symbols()
supercell_symbols = primitive_symbols * 8
supercell.set_chemical_symbols(supercell_symbols)

supercell.write('verified_supercell.cif')
```

### Example 4: Use in ASE Calculator Workflow

```python
from pymultibinit import MultibinitCalculator
from ase.io import read
from ase.optimize import BFGS

# Load structure
atoms = read('initial.cif')

# Create calculator
calc = MultibinitCalculator.from_config_file('multibinit.conf')
atoms.calc = calc

# Run calculation (this sets the reference)
energy = atoms.get_potential_energy()

# Export the internal supercell for visualization
supercell = calc.potential.export_supercell_to_ase()
supercell.set_chemical_symbols(atoms.get_chemical_symbols() * 8)  # Assuming 2x2x2
supercell.write('supercell_reference.cif')

# Continue with optimization
opt = BFGS(atoms)
opt.run(fmax=0.01)
```

### Example 5: Multiple Formats

```python
from pymultibinit import MultibinitPotential

pot = MultibinitPotential.from_config_file('multibinit.conf')

# ... initialize and evaluate ...

# Export to multiple formats
atoms = pot.export_supercell_to_ase()
atoms.set_chemical_symbols(['Sr']*8 + ['Ti']*8 + ['O']*24)

# CIF for visualization (VESTA, etc.)
atoms.write('supercell.cif')

# VASP POSCAR
atoms.write('POSCAR', format='vasp')

# XYZ for quick viewing
atoms.write('supercell.xyz')

# Extended XYZ with properties
atoms.write('supercell.extxyz', format='extxyz')

# JSON for programmatic access
atoms.write('supercell.json', format='json')
```

### Example 6: Checking Atom Matching

```python
from pymultibinit import MultibinitPotential
from ase.io import read
import numpy as np

# Initialize
pot = MultibinitPotential.from_params(
    ddb_file="system_DDB",
    sys_file="system.xml",
    ncell=(2, 2, 2),
)

# Read input structure
input_atoms = read('input.cif')

# Evaluate to trigger atom matching
pot.evaluate(input_atoms.get_positions(), input_atoms.get_cell())

# Export reference structure to see internal ordering
reference = pot.export_supercell_to_ase()
reference.set_chemical_symbols(['Ba', 'Ti', 'O', 'O', 'O'] * 8)
reference.write('reference_ordering.cif')

# Check atom mapping
mapping_info = pot.get_atom_mapping()
if mapping_info:
    mapping, inverse_mapping = mapping_info
    print("Atom mapping:")
    print(f"  Reference atom 0 matches input atom {mapping[0]}")
    print(f"  Input atom 0 corresponds to reference atom {inverse_mapping[0]}")
```

## Setting Chemical Symbols

Since the C API doesn't provide atomic numbers, you need to set chemical symbols manually:

### Method 1: Simple Pattern

```python
atoms = pot.export_supercell_to_ase()

# For BaTiO3 2x2x2 supercell: 8 cells * (1 Ba + 1 Ti + 3 O) = 8+8+24 atoms
symbols = ['Ba'] * 8 + ['Ti'] * 8 + ['O'] * 24
atoms.set_chemical_symbols(symbols)
```

### Method 2: From Primitive Cell

```python
from ase.io import read

# Read primitive cell with correct symbols
primitive = read('primitive.cif')
primitive_symbols = primitive.get_chemical_symbols()

# Replicate for supercell (e.g., 2x2x2 = 8 times)
nx, ny, nz = 2, 2, 2
n_replicas = nx * ny * nz
supercell_symbols = primitive_symbols * n_replicas

atoms = pot.export_supercell_to_ase()
atoms.set_chemical_symbols(supercell_symbols)
```

### Method 3: Interactive Editing

```python
# Export with dummy symbols
pot.export_supercell_to_file('supercell.cif')

# Manually edit supercell.cif to set correct atom types
# Then read it back
from ase.io import read
atoms = read('supercell.cif')  # Now with correct symbols
```

## Important Notes

1. **Reference Structure Availability**
   - Reference is set on first `evaluate()` call (if `auto_match_atoms=True`)
   - Or explicitly via `set_reference_structure()`
   - Export methods raise `RuntimeError` if reference not available

2. **Units**
   - ASE export is always in Angstrom (ASE convention)
   - Automatic conversion if `use_atomic_units=True`
   - `get_supercell_structure()` returns in current units

3. **Chemical Symbols**
   - Always 'X' (unknown) from C API
   - Must be set manually based on your system
   - Symbol order must match atom ordering in supercell

4. **Atom Ordering**
   - Internal MULTIBINIT ordering may differ from input
   - Use atom matching to handle this automatically
   - Export shows the internal ordering

## Troubleshooting

### RuntimeError: Reference structure not available

```
RuntimeError: Reference structure not available. Call evaluate() at least once...
```

**Solution:** Call `evaluate()` at least once or use `set_reference_structure()`

```python
# Option 1: Evaluate with dummy data
pot.evaluate(dummy_positions, dummy_lattice)
atoms = pot.export_supercell_to_ase()

# Option 2: Set reference explicitly
pot.set_reference_structure(known_positions, known_lattice)
atoms = pot.export_supercell_to_ase()
```

### ImportError: ASE is required

```
ImportError: ASE is required for this function. Install with: pip install ase
```

**Solution:** Install ASE

```bash
pip install ase
# or
uv pip install ase
```

### Wrong Number of Atoms

If exported supercell has unexpected number of atoms:

```python
# Check supercell dimensions
ncell = (2, 2, 2)  # Should be 2x2x2 = 8 times primitive

# Check primitive cell atom count
from ase.io import read
primitive = read('primitive.cif')
expected_atoms = len(primitive) * 8

# Compare with export
atoms = pot.export_supercell_to_ase()
print(f"Expected: {expected_atoms}, Got: {len(atoms)}")
```

## See Also

- `docs/ATOM_MATCHING_USAGE.md` - Atom matching and reordering
- `docs/CONFIG_FILE_USAGE.md` - Configuration file usage
- `docs/api/API_REFERENCE.md` - Complete API reference
- ASE documentation: https://wiki.fysik.dtu.dk/ase/
