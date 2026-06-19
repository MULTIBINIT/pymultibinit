# pymultibinit - Python Interface to MULTIBINIT

Python bindings for ABINIT's MULTIBINIT effective potential library.

## Installation

```bash
# 1. Build ABINIT library
cd abinit
rm -rf build && mkdir build && cd build
CC=mpicc FC=mpif90 cmake -C ~/.abinit/build/shared.cmake ..
make -j8

# 2. Set library path (choose one):

# Option A: LIBABINIT_PATH (recommended)
export LIBABINIT_PATH=/path/to/abinit/build/src/98_main/libabinit.dylib  # macOS
export LIBABINIT_PATH=/path/to/abinit/build/src/98_main/libabinit.so     # Linux

# Option B: Add to LD_LIBRARY_PATH (Linux) or DYLD_LIBRARY_PATH (macOS)
export LD_LIBRARY_PATH=/path/to/abinit/build/src/98_main:$LD_LIBRARY_PATH

# Add to ~/.bashrc or ~/.zshrc for persistence

# 3. Install Python package
pip install pymultibinit
# or: pip install -e .
```

## Quick Start

This shows how to use the multibinit potential as a ASE calculator.

```python
from pymultibinit import MultibinitCalculator
from ase import Atoms
from ase.optimize import BFGS

# Load from config file (recommended)
calc = MultibinitCalculator.from_config_file("multibinit.conf")

# Or from parameters (requires libabinit / Fortran)
calc = MultibinitCalculator.from_params(
    ddb_file="system_DDB",
    sys_file="system.xml",
    ncell=(2, 2, 2),
    ngqpt=(4, 4, 4)
)

# Or pure Python (no libabinit required) - reads DDB directly,
# optional XML coefficients, toggles dipole-dipole and supercell size.
calc = MultibinitCalculator.from_pyeffpot(
    ddb_file="system.DDB",
    xml_file="coeffs.xml",   # optional, None to skip
    ncell=(2, 2, 2),
    dipdip=True,
)

# Build supercell (must match ncell!)
unit_cell = Atoms('BaHfO3', 
                 scaled_positions=[[0,0,0], [0.5,0.5,0.5], 
                                   [0.5,0,0.5], [0,0.5,0.5], [0.5,0.5,0]],
                 cell=[4.0, 4.0, 4.0], pbc=True)
atoms = unit_cell * (2, 2, 2)  # Match ncell=(2,2,2)
atoms.calc = calc

# Run calculation
energy = atoms.get_potential_energy()  # eV
forces = atoms.get_forces()            # eV/Angstrom
stress = atoms.get_stress()            # eV/Angstrom^3

# Optimize structure
opt = BFGS(atoms)
opt.run(fmax=0.01)
```

## Configuration File

Create `multibinit.conf`:

```ini
# Required
ddb_file: system_DDB
sys_file: system.xml
ncell: 2 2 2

# Optional
ngqpt: 4 4 4
dipdip: 1
```

## Building Models With the MULTIBINIT Binary

`pymultibinit` provides a thin model-building runner that calls the external
`multibinit` executable. It does not use ABINIT library mode for training.

```python
from pymultibinit import train_multibinit_model

result = train_multibinit_model(
    ddb="system.ddb",
    hist="training_HIST.nc",
    config="train.abi",
    output_dir="model_out",
    executable="/path/to/multibinit",  # optional if MULTIBINIT_BINARY or PATH is set
)

print(result.model_config)
print(result.metadata_file)
```

The same workflow is available from `mbtools`:

```bash
mbtools train system.ddb training_HIST.nc \
  --config train.abi \
  --output-dir model_out \
  --executable /path/to/multibinit
```

The runner exposes resolved input paths to the binary through environment
variables: `PYMULTIBINIT_DDB`, `PYMULTIBINIT_HIST`, `PYMULTIBINIT_CONFIG`, and
`PYMULTIBINIT_OUTPUT_DIR`. It captures stdout/stderr logs and writes
`pymultibinit_training_result.json` in the output directory.

## Pure-Python Model Training

`pymultibinit.training` also provides a pure-Python fitting path that reads DDB,
HIST, and XML basis inputs, evaluates coefficient features, solves fitted
values, and writes fitted coefficient XML without invoking `multibinit`.

```bash
mbtools train-python system.ddb training_HIST.nc \
  --basis-xml candidate_basis.xml \
  --output-xml fit_coeffs.xml \
  --diagnostics-json fit_diagnostics.json \
  --ncell 2 2 2 \
  --selection greedy \
  --ncoeff 20
```

For the detailed procedure, including dataset residuals, feature matrices,
displacement-only term generation, symmetry orbit materialization, and greedy
term selection, see `docs/PURE_PYTHON_TRAINING.md`.

## Exporting DDB Files to Phonopy

`pymultibinit` can export ABINIT text DDB harmonic data to a self-contained
`phonopy_params.yaml` file without invoking ABINIT binaries. The embedded force
constants use phonopy's default VASP-compatible unit convention, so a plain
`phonopy.load("phonopy_params.yaml")` reports frequencies in THz.

```python
from pymultibinit import write_phonopy_from_ddb

result = write_phonopy_from_ddb("system_DDB", "phonopy_from_ddb")
print(result.phonopy_params_yaml)
```

The same workflow is available from `mbtools`:

```bash
mbtools ddb-to-phonopy system_DDB phonopy_from_ddb --verbose
```

The export writes only `phonopy_params.yaml`. The default supercell is the DDB
`ngqpt` q-grid. If `--supercell NX NY NZ` is provided, it must match that
q-grid.

## API Reference

### `MultibinitCalculator` (ASE interface)

Two backends are supported. The **CFFI backend** requires `libabinit.so`
(Fortran); the **pyeffpot backend** is pure Python and needs only the DDB file.

```python
# --- CFFI backend (requires libabinit) ---

# From config file (recommended when libabinit is available)
calc = MultibinitCalculator.from_config_file("config.conf")

# From parameters
calc = MultibinitCalculator.from_params(
    ddb_file="system_DDB",
    sys_file="system.xml",
    ncell=(2, 2, 2),
    ngqpt=(4, 4, 4),
    dipdip=1
)

# From .abi file
calc = MultibinitCalculator.from_abi("input.abi")

# --- Pure Python backend (no libabinit required) ---

# Reads DDB directly; XML coefficients are optional.
calc = MultibinitCalculator.from_pyeffpot(
    ddb_file="system.DDB",
    xml_file="model.xml",   # optional, pass None or omit to skip
    ncell=(2, 2, 2),        # supercell size
    dipdip=True,            # long-range dipole-dipole correction
    asr=True,               # acoustic sum rule
)
```

### `MultibinitPotential` (Low-level interface)

```python
from pymultibinit import MultibinitPotential
import numpy as np

pot = MultibinitPotential.from_params(
    ddb_file="system_DDB",
    sys_file="system.xml",
    ncell=(2, 2, 2)
)

# Evaluate (Angstrom/eV by default)
positions = np.array([[0, 0, 0], [2.0, 0, 0]])  # Angstrom
lattice = np.array([[4, 0, 0], [0, 4, 0], [0, 0, 4]])  # Angstrom
energy, forces, stress = pot.evaluate(positions, lattice)

# Export structures
pot.export_supercell_to_file('supercell.cif')  # Export to file
atoms = pot.export_supercell_to_ase()           # Get ASE Atoms object
structure = pot.get_supercell_structure()       # Get raw arrays

pot.free()
```

## Exporting Structures

### Export Reference/Supercell Structure

```python

```

### Command-Line Tools

```bash
# Export MULTIBINIT reference structure
mbtools export-ref config.conf output.cif

```

## Important Notes

### Supercell Size

**Your structure must match the `ncell` parameter!**

```python
# Config: ncell: 2 2 2
calc = MultibinitCalculator.from_config_file("config.conf")

# ✓ CORRECT
atoms = unit_cell * (2, 2, 2)  # Match ncell

# ✗ WRONG - will fail with "status 3" error
atoms = unit_cell  # Size mismatch
```

### Library Not Found?

If you get "Could not find libabinit":

```bash
# Option 1: Set LIBABINIT_PATH
export LIBABINIT_PATH=/full/path/to/libabinit.dylib

# Option 2: Add to library search path
export LD_LIBRARY_PATH=/path/to/dir/containing/libabinit:$LD_LIBRARY_PATH

# Verify
python -c "from pymultibinit.utils import find_library; print(find_library())"
```

### Unit Conventions

| Quantity | Python API | Internal |
|----------|------------|----------|
| Length   | Angstrom   | Bohr     |
| Energy   | eV         | Hartree  |
| Force    | eV/Å       | Ha/Bohr  |

Conversions are automatic.

## Examples

```bash
cd pymultibinit/examples
python simple_example.py
```

BaHfO3 binary-training tutorial and example:

- `docs/BAHFO3_TRAINING_TUTORIAL.md`
- `examples/BaHfO3_training/01_train_bahfo3.py`

Runnable dry run:

```bash
uv run python examples/BaHfO3_training/01_train_bahfo3.py --dry-run
```

## License

GPL v3 (same as ABINIT)

## Citation

- ABINIT: https://www.abinit.org
- MULTIBINIT: https://docs.abinit.org/guide/multibinit/
