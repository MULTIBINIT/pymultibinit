# pymultibinit - Python Interface to MULTIBINIT

Python bindings for ABINIT's MULTIBINIT effective potential library, enabling molecular dynamics and structure optimization using machine-learned potentials.

## Features

- **Three initialization modes:**
  - From `.abi` input file (standard MULTIBINIT format)
  - Direct parameter initialization (no `.abi` file required)
  - From configuration file (recommended for reproducibility)
  
- **Automatic unit conversion:**
  - Internal: atomic units (Bohr, Hartree)
  - Python API: eV/Angstrom by default (ASE-compatible)
  
- **ASE Calculator interface** for seamless integration with ASE workflows

- **CFFI-based wrapper** for efficient C library binding

- **Cross-platform support:**
  - macOS (.dylib), Linux (.so), Windows (.dll)
  - Automatic library detection with multiple extension support

- **Three API levels:**
  1. **High-level:** `MultibinitPotential` and `MultibinitCalculator`
  2. **Low-level:** `MultibinitWrapperCFFI`
  3. **C API:** Direct C interface (for advanced users)

## Installation

### Prerequisites

1. **Build the C library:**
   ```bash
   cd abinit_mb_clib
   rm -rf build && mkdir build && cd build
   CC=mpicc FC=mpif90 cmake -C ~/.abinit/build/shared.cmake ..
   make -j8
   ```

2. **Install Python package:**
   ```bash
   cd pymultibinit
   uv sync  # or: pip install -e .
   ```
   
   CFFI is a required dependency and will be installed automatically.

## Quick Start

### Using the High-Level API

```python
from pymultibinit import MultibinitPotential
import numpy as np

# Method 1: Initialize from .abi file
pot = MultibinitPotential.from_abi("input.abi")

# Method 2: Initialize from parameters (no .abi file)
pot = MultibinitPotential.from_params(
    ddb_file="system_DDB",
    sys_file="system.xml",
    ncell=(2, 2, 2),
    ngqpt=(4, 4, 4),
    dipdip=1
)

# Evaluate energy, forces, stress (in eV/Angstrom by default)
positions = np.array([[0, 0, 0], [2.0, 0, 0]])  # Angstrom
lattice = np.array([[4, 0, 0], [0, 4, 0], [0, 0, 4]])  # Angstrom

energy, forces, stress = pot.evaluate(positions, lattice)
print(f"Energy: {energy} eV")
print(f"Forces: {forces} eV/Angstrom")
print(f"Stress: {stress} eV/Angstrom^3")

pot.free()  # Clean up resources
```

### Using the ASE Calculator

```python
from pymultibinit import MultibinitCalculator
from ase import Atoms
from ase.optimize import BFGS

# Method 1: From parameters
calc = MultibinitCalculator.from_params(
    ddb_file="system_DDB",
    sys_file="system.xml",
    ncell=(2, 2, 2)  # Creates 2×2×2 supercell internally
)

# Method 2: From config file
calc = MultibinitCalculator.from_config_file("multibinit.conf")

# IMPORTANT: Build supercell matching ncell parameter
unit_cell = Atoms('BaHfO3',
                 scaled_positions=[
                     [0, 0, 0],        # Ba
                     [0.5, 0.5, 0.5],  # Hf
                     [0.5, 0, 0.5],    # O
                     [0, 0.5, 0.5],    # O
                     [0.5, 0.5, 0]     # O
                 ],
                 cell=[4.15, 4.15, 4.15],
                 pbc=True)

# If ncell=(2,2,2), create 2×2×2 supercell (40 atoms for 5-atom unit cell)
atoms = unit_cell * (2, 2, 2)
atoms.calc = calc

# Use standard ASE interface
energy = atoms.get_potential_energy()  # eV
forces = atoms.get_forces()            # eV/Angstrom
stress = atoms.get_stress()            # eV/Angstrom^3

# Structure optimization with ASE
opt = BFGS(atoms)
opt.run(fmax=0.01)
```

## API Reference

### `MultibinitPotential`

High-level potential interface with automatic unit conversions.

**Class Methods:**
- `from_abi(abi_file, lib_path=None, use_atomic_units=False)` - Initialize from .abi file
- `from_params(ddb_file, sys_file="", coeff_file="", ncell=(1,1,1), ngqpt=(1,1,1), dipdip=1, ...)` - Direct initialization
- `from_config_file(config_file, lib_path=None)` - Initialize from configuration file

**Instance Methods:**
- `evaluate(positions, lattice)` → `(energy, forces, stress)` - Compute properties
- `free()` - Release resources

**Parameters:**
- `positions`: array shape `(natom, 3)` - atomic positions
- `lattice`: array shape `(3, 3)` - lattice vectors (row vectors)
- Units: Angstrom/eV by default, Bohr/Hartree if `use_atomic_units=True`

**Important:** The structure passed to `evaluate()` must match the `ncell` supercell size. If `ncell=(2,2,2)`, provide a 2×2×2 supercell.

### `MultibinitCalculator`

ASE calculator interface.

**Class Methods:**
- `from_abi(abi_file, lib_path=None, **kwargs)` - Create from .abi file
- `from_params(ddb_file, sys_file, ..., **kwargs)` - Create from parameters
- `from_config_file(config_file, **kwargs)` - Create from configuration file

**Properties Implemented:**
- `energy` (eV)
- `forces` (eV/Angstrom)
- `stress` (eV/Angstrom^3, Voigt notation)

**Important:** The structure passed to the calculator must match the `ncell` supercell size. If `ncell=(2,2,2)`, create a 2×2×2 supercell from your unit cell.

### `MultibinitWrapperCFFI`

Low-level CFFI wrapper for C API (advanced users).

**Methods:**
- `init_from_abi_file(abi_filename)` - Initialize from .abi
- `init_from_params(ddb_file, sys_file, coeff_file, ncell, ngqpt, dipdip)` - Parameter init
- `evaluate(natom, positions, lattice)` - Evaluate (all atomic units)
- `free()` - Release resources

**Units:** All atomic units (Bohr, Hartree)

## Initialization Modes

### 1. From .abi File

Uses standard MULTIBINIT input parser. All parameters from .abi file.

```python
pot = MultibinitPotential.from_abi("input.abi")
```

**Required in .abi file:**
- DDB file path
- System XML file path (optional)
- Coefficient XML file path (optional)
- Grid parameters: `ncell`, `ngqpt`
- Interaction flags: `dipdip`, etc.

### 2. From Parameters

No .abi file required. Parameters passed directly.

```python
pot = MultibinitPotential.from_params(
    ddb_file="tmulti_l_6_DDB",          # Required
    sys_file="tmulti_l_8_1.xml",        # Optional (read from DDB if omitted)
    coeff_file="coefficients.xml",      # Optional
    ncell=(2, 2, 2),                    # Supercell size
    ngqpt=(4, 4, 4),                    # q-point grid
    dipdip=1                            # Dipole-dipole interactions (0=off, 1=on)
)
```

**Default parameters:**
- `asr=2` (acoustic sum rule)
- `rfmeth=1` (response function method)
- `symdynmat=1` (symmetrize dynamical matrix)
- `nph1l=1` (Gamma-point only)

### 3. From Configuration File (Recommended)

Convenient way to specify all parameters in a separate file. **This is the recommended approach for production use.**

```python
# For MultibinitPotential
pot = MultibinitPotential.from_config_file("multibinit.conf")

# For MultibinitCalculator (ASE)
calc = MultibinitCalculator.from_config_file("multibinit.conf")
```

**Example config file (`multibinit.conf`):**
```ini
# Required: DDB and system files
ddb_file: system_DDB
sys_file: system.xml

# Required: Supercell dimensions
# IMPORTANT: Your ASE structure must match this size!
ncell: 2 2 2

# Optional: q-point grid (default: 1 1 1)
ngqpt: 4 4 4

# Optional: Dipole-dipole interactions (default: 1)
dipdip: 1

# Optional: Unit system (default: false = Angstrom/eV)
use_atomic_units: false

# Optional: Atom matching (default: true)
auto_match_atoms: true
match_tolerance: 0.1
```

**Configuration File Formats:**

The config parser supports two formats:

1. **Simple format** (recommended):
```ini
# Comments with #
ddb_file: system_DDB
sys_file: system.xml
ncell: 2 2 2  # Inline comments work too
```

2. **INI format with sections**:
```ini
[files]
ddb_file = system_DDB
sys_file = system.xml

[parameters]
ncell = 2 2 2
ngqpt = 4 4 4
dipdip = 1
```

**Path Resolution:**
- Relative paths are resolved relative to the config file directory
- Absolute paths are used as-is

**Benefits:**
- **Reproducibility:** All parameters documented in one place
- **Portability:** Easy to share configs across machines
- **Version control:** Keep configs in git alongside code
- **Batch processing:** Different configs for different systems
- **Separation of concerns:** Data separate from code

**Complete Usage Example:**

```python
from pymultibinit import MultibinitCalculator
from ase import Atoms
from ase.optimize import BFGS

# Load calculator from config
calc = MultibinitCalculator.from_config_file("multibinit.conf")

# Build unit cell
unit_cell = Atoms('BaHfO3',
                 scaled_positions=[
                     [0, 0, 0],        # Ba
                     [0.5, 0.5, 0.5],  # Hf
                     [0.5, 0, 0.5],    # O
                     [0, 0.5, 0.5],    # O
                     [0.5, 0.5, 0]     # O
                 ],
                 cell=[4.15, 4.15, 4.15],
                 pbc=True)

# IMPORTANT: Create supercell matching config ncell
atoms = unit_cell * (2, 2, 2)  # Match ncell: 2 2 2
atoms.calc = calc

# Run optimization
opt = BFGS(atoms)
opt.run(fmax=0.01)
```

**Available Configuration Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ddb_file` | str | Required | Path to DDB file |
| `sys_file` | str | Optional | Path to system XML file |
| `coeff_file` | str | Optional | Path to coefficients XML |
| `abi_file` | str | Optional | Path to .abi file (alternative to above) |
| `ncell` | 3 ints | (1,1,1) | Supercell dimensions **[MUST MATCH STRUCTURE SIZE]** |
| `ngqpt` | 3 ints | (1,1,1) | q-point grid |
| `dipdip` | int | 1 | Dipole-dipole interactions (0=off, 1=on) |
| `use_atomic_units` | bool | false | Use Bohr/Hartree instead of Angstrom/eV |
| `auto_match_atoms` | bool | true | Enable automatic atom ordering |
| `match_tolerance` | float | 0.1 | Atom matching tolerance (Angstrom) |
| `lib_path` | str | Auto | Path to libabinit library |

See [docs/CONFIG_FILE_USAGE.md](docs/CONFIG_FILE_USAGE.md) for complete documentation with more examples.

## Command-Line Tools

### mbtools - MultiBinit Tools

A consolidated CLI for working with MULTIBINIT potentials and structures.

**Installation:**

After installing the package, the tool is available as:
```bash
mbtools [command] [options]
```

Or run directly as a Python module:
```bash
python -m pymultibinit.cli [command] [options]
```

---

### Command: `export-ref`

Export the MULTIBINIT internal reference structure (supercell) to various file formats.

**Usage:**
```bash
mbtools export-ref config.conf output.cif [options]
```

**Examples:**

```bash
# Export to CIF format (auto-detected from extension)
mbtools export-ref multibinit.conf structure.cif

# Export to other formats with explicit format specification
mbtools export-ref multibinit.conf structure.xyz -f xyz
mbtools export-ref multibinit.conf POSCAR -f vasp
mbtools export-ref multibinit.conf structure.json -f json

# Export with chemical symbols (40 atoms for 2×2×2 supercell)
mbtools export-ref multibinit.conf structure.cif \
    -s "Ba,Ba,Ba,Ba,Ba,Ba,Ba,Ba,Ti,Ti,Ti,Ti,Ti,Ti,Ti,Ti,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O"

# Verbose output for debugging
mbtools export-ref multibinit.conf structure.cif -v

# Export after evaluating with test structure (recommended)
mbtools export-ref multibinit.conf reference.cif \
    --from-structure test_supercell.cif \
    -s "Ba,Ba,Ba,Ba,Ba,Ba,Ba,Ba,Ti,Ti,Ti,Ti,Ti,Ti,Ti,Ti,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O" \
    -v
```

**Options:**
- `-f, --format`: Output format (cif, xyz, vasp, json, extxyz). Auto-detected from extension.
- `-s, --symbols`: Chemical symbols as comma-separated list (e.g., "Ba,Ti,O,O,O")
- `--from-structure`: Structure file to evaluate first (must match ncell). Recommended method.
- `-v, --verbose`: Print detailed information

**Important Notes:**
- Exported structure is the MULTIBINIT internal **supercell** (`ncell × unit_cell`), NOT the DDB unit cell
- For `ncell: 2 2 2` with 5-atom unit cell → 40 atoms (8×Ba, 8×Ti/Hf, 24×O for perovskite)
- Chemical symbols must be provided via `-s` since MULTIBINIT doesn't store them
- The `--from-structure` method evaluates potential once to extract reference structure

---

### Command: `make-supercell`

Create a supercell from a unit cell structure file by repeating it nx × ny × nz times.

**Usage:**
```bash
mbtools make-supercell input.cif output.cif nx ny nz [options]
```

**Examples:**

```bash
# Create 2×2×2 supercell from unit cell
mbtools make-supercell unit_cell.cif supercell.cif 2 2 2

# Create supercell from VASP POSCAR
mbtools make-supercell POSCAR POSCAR_222 2 2 2 -f vasp

# Create 3×3×1 supercell with verbose output
mbtools make-supercell structure.xyz super.xyz 3 3 1 -v

# Mixed formats (CIF → VASP)
mbtools make-supercell unit_cell.cif POSCAR_222 2 2 2 -f vasp
```

**Options:**
- `-f, --format`: Output format (cif, xyz, vasp, json, extxyz). Auto-detected from extension.
- `-v, --verbose`: Print detailed information

**Why Use This?**
- Create supercells matching `ncell` parameter in your config file
- Prepare test structures for `mbtools export-ref --from-structure`
- Quick supercell generation for MD/optimization without writing Python scripts

---

**Supported Formats:**

| Format | Extension | Description |
|--------|-----------|-------------|
| CIF | `.cif` | Crystallographic Information File |
| XYZ | `.xyz` | Simple XYZ format |
| VASP | `POSCAR` | VASP POSCAR format |
| JSON | `.json` | JSON representation (ASE Atoms) |
| ExtXYZ | `.extxyz` | Extended XYZ with metadata |

---

**Complete Workflow Example:**

```bash
# 1. Create configuration file
cat > multibinit.conf <<EOF
ddb_file: BaTiO3_DDB
sys_file: BaTiO3.xml
ncell: 2 2 2
ngqpt: 4 4 4
dipdip: 1
EOF

# 2. Create a unit cell structure (Python/ASE)
python <<EOF
from ase import Atoms
from ase.io import write

unit_cell = Atoms('BaTiO3',
                 scaled_positions=[[0,0,0], [0.5,0.5,0.5], [0.5,0,0.5], [0,0.5,0.5], [0.5,0.5,0]],
                 cell=[4.0, 4.0, 4.0], pbc=True)
write('unit_cell.cif', unit_cell)
EOF

# 3. Create 2×2×2 supercell using mbtools
mbtools make-supercell unit_cell.cif supercell_222.cif 2 2 2 -v

# 4. Export MULTIBINIT reference structure
mbtools export-ref multibinit.conf reference.cif \
    --from-structure supercell_222.cif \
    -s "Ba,Ba,Ba,Ba,Ba,Ba,Ba,Ba,Ti,Ti,Ti,Ti,Ti,Ti,Ti,Ti,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O,O" \
    -v

# 5. View the structure
ase gui reference.cif
```

**Why Use mbtools?**

1. **Verify structure:** Check the internal MULTIBINIT supercell is correct
2. **Debug:** Ensure `ncell` parameter creates the expected structure
3. **Interoperability:** Export for use in other codes (VASP, Quantum ESPRESSO, etc.)
4. **Visualization:** Generate files for structure viewers (VESTA, Ovito, etc.)
5. **Quick supercell creation:** No need to write Python scripts for simple supercells

## Unit Conventions

| Quantity | Atomic Units | ASE/Python Units |
|----------|-------------|------------------|
| Length   | Bohr        | Angstrom         |
| Energy   | Hartree     | eV               |
| Force    | Hartree/Bohr | eV/Angstrom     |
| Stress   | Hartree/Bohr³ | eV/Angstrom³   |

**Conversion factors:**
- 1 Bohr = 0.529177210903 Angstrom
- 1 Hartree = 27.211386245988 eV

## Implementation

pymultibinit uses CFFI (C Foreign Function Interface) for efficient Python-C binding:

- ✅ Efficient memory handling
- ✅ Good performance
- ✅ Automatic type conversions
- ✅ Cross-platform support (macOS, Linux, Windows)

The CFFI wrapper provides a clean interface to the MULTIBINIT C library while maintaining good performance characteristics.

## Testing

### Import Test
```bash
cd pymultibinit
uv run python tests/test_import.py
```

### Functional Tests
```bash
cd pymultibinit
uv run python tests/test_functional.py
```

**Note:** Functional tests require MPI to be properly configured. If tests hang, it may be an MPI initialization issue.

## File Structure

```
pymultibinit/
├── src/
│   └── pymultibinit/
│       ├── __init__.py          # Main package
│       ├── wrapper_cffi.py      # Low-level CFFI wrapper
│       ├── potential.py         # High-level potential API
│       ├── calculator.py        # ASE calculator interface
│       ├── config.py            # Configuration file parsing
│       ├── atom_matching.py     # Atom ordering utilities
│       └── utils.py             # Library finding utilities
├── tests/
│   ├── test_import.py           # Import/library loading test
│   ├── test_functional.py       # Functional tests
│   ├── test_zero_forces_supercell.py  # Zero force verification
│   └── test_api.py              # Comprehensive API tests
├── examples/
│   ├── BaTiO3_example/          # Full example with data files
│   │   └── 01_basic_usage.py
│   └── simple_example.py        # Minimal working example
├── pyproject.toml
└── README.md
```

## ⚠️ Important Notes

### Supercell Size Requirement

**The most common source of errors:** Your structure must match the `ncell` parameter.

```python
# Config has: ncell: 2 2 2
calc = MultibinitCalculator.from_config_file("config.conf")

# ✓ CORRECT: Build matching supercell
unit_cell = Atoms(...)  # 5 atoms
atoms = unit_cell * (2, 2, 2)  # 40 atoms for 2×2×2 supercell

# ✗ WRONG: Using unit cell directly
atoms = unit_cell  # 5 atoms - will fail with "mb_evaluate failed with status 3"
```

**Why?** MULTIBINIT builds an internal supercell using `ncell` during initialization. Your input structure must have exactly the same number of atoms (natom × ncell[0] × ncell[1] × ncell[2]).

**Error you'll see if you get this wrong:**
```
RuntimeError: mb_evaluate failed with status 3
```

### Other Known Issues

1. **MPI initialization:** The C library initializes MPI internally. Running multiple tests in sequence may cause MPI conflicts. Use separate Python processes for each test.

2. **Library path:** The wrapper auto-detects the library across platforms (.dylib, .so, .dll). For custom locations, pass `lib_path` to constructors. See [PLATFORM_SUPPORT.md](PLATFORM_SUPPORT.md) for details.

3. **Stress sign convention:** ASE uses pressure-like convention (negative of standard Voigt stress). The calculator handles this automatically.

## Development

Built with:
- Python 3.10+
- CFFI (C Foreign Function Interface)
- NumPy
- ASE (for calculator interface and structure manipulation)

**Dependencies:**
```toml
dependencies = [
    "ase>=3.26.0",
    "numpy>=2.2.6",
    "cffi>=1.15.0",
]
```

## License

Same as ABINIT (GPL v3)

## Citation

If you use this software, please cite ABINIT and MULTIBINIT:
- ABINIT: https://www.abinit.org
- MULTIBINIT: https://docs.abinit.org/guide/multibinit/

## Documentation

For detailed technical documentation, see the [../docs/](../docs/) directory:
- **[Implementation Guide](../docs/implementation/)** - C binding procedures and implementation details
- **[FORTRAN Internals](../docs/fortran/)** - MULTIBINIT supercell mapping and code documentation
- **[Platform Support](../docs/implementation/PLATFORM_SUPPORT.md)** - Cross-platform library support details

## Examples

### Minimal Example
```bash
cd pymultibinit/examples
uv run python simple_example.py
```

A minimal example using test data (BaHfO3) that demonstrates:
- Loading potential from DDB and XML files
- Building a reference 2×2×2 supercell
- Evaluating forces (verifies zero forces at equilibrium)

### Full Example with ASE
```bash
cd pymultibinit/examples/BaTiO3_example
uv run python 01_basic_usage.py
```

A comprehensive example showing:
- Loading potential from config file
- Building equilibrium structures with ASE
- Evaluating energy, forces, and stress
- Exporting structures to CIF format

## See Also

- C API documentation: `../abinit_mb_clib/README.md`
- MULTIBINIT manual: https://docs.abinit.org/guide/multibinit/
- ASE documentation: https://wiki.fysik.dtu.dk/ase/
