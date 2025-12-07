# pymultibinit - Python Interface to MULTIBINIT

Python bindings for ABINIT's MULTIBINIT effective potential library, enabling molecular dynamics and structure optimization using machine-learned potentials.

## Features

- **Two initialization modes:**
  - From `.abi` input file (standard MULTIBINIT format)
  - Direct parameter initialization (no `.abi` file required)
  
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

# Create calculator
calc = MultibinitCalculator.from_params(
    ddb_file="system_DDB",
    sys_file="system.xml",
    ncell=(2, 2, 2)
)

# Attach to ASE atoms
atoms = Atoms('SrTiO3', positions=..., cell=..., pbc=True)
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

**Instance Methods:**
- `evaluate(positions, lattice)` → `(energy, forces, stress)` - Compute properties
- `free()` - Release resources

**Parameters:**
- `positions`: array shape `(natom, 3)` - atomic positions
- `lattice`: array shape `(3, 3)` - lattice vectors (row vectors)
- Units: Angstrom/eV by default, Bohr/Hartree if `use_atomic_units=True`

### `MultibinitCalculator`

ASE calculator interface.

**Class Methods:**
- `from_abi(abi_file, lib_path=None, **kwargs)` - Create from .abi file
- `from_params(ddb_file, sys_file, ..., **kwargs)` - Create from parameters

**Properties Implemented:**
- `energy` (eV)
- `forces` (eV/Angstrom)
- `stress` (eV/Angstrom^3, Voigt notation)

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

## Known Issues

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
