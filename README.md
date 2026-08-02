# pymultibinit

A Python package for building and evaluating MULTIBINIT effective-potential
models of lattice dynamics. It provides a **native pure-Python implementation**
of the harmonic (IFC) and anharmonic effective potential, plus an optional
**CFFI backend** that wraps the ABINIT Fortran library for bit-exact parity.

## Features

- **Pure-Python effective potential** (`pyeffpot`): reads a DDB file directly,
  builds the supercell IFCs with Fourier interpolation, dipole-dipole Ewald
  correction, acoustic sum rule, and optional anharmonic XML coefficients.
  No Fortran compilation required.
- **ASE calculator interface**: energy, forces, and stress through the standard
  `atoms.get_potential_energy()` / `get_forces()` / `get_stress()` API.
- **CFFI backend** (optional): wraps `libabinit.so` for bit-exact agreement with
  the standalone MULTIBINIT binary.
- **Energy/force/stress decomposition** into dipdip, harmonic, anharmonic terms.
- **DDB → phonopy export** without running ABINIT.
- **Model training**: pure-Python least-squares fitting or binary-based training.

## Installation

```bash
pip install pymultibinit
# or from source:
pip install -e .
```

That's it for the pure-Python backend. No Fortran compilation needed.

### Optional: CFFI backend (libabinit)

The CFFI backend wraps the ABINIT shared library. It is required only for
`from_abi`, `from_abi_spawned`, `from_params`, and `from_config_file`.
`from_pyeffpot` works without it.

```bash
# 1. Build ABINIT with shared library
cd abinit
rm -rf build && mkdir build && cd build
CC=mpicc FC=mpif90 cmake -DBUILD_SHARED_LIBS=ON ..
make -j8

# 2. Set library path (choose one):
export LIBABINIT_PATH=/path/to/abinit/build/src/98_main/libabinit.so
# or:
export LD_LIBRARY_PATH=/path/to/abinit/build/src/98_main:$LD_LIBRARY_PATH
```

### Optional dependencies

```bash
pip install pymultibinit[jax]   # JAX acceleration for anharmonic evaluation
```

## Quick Start

```python
from pymultibinit import MultibinitCalculator
from ase import Atoms
from ase.optimize import BFGS

# Create calculator from DDB (pure Python, no libabinit needed)
calc = MultibinitCalculator.from_pyeffpot(
    ddb_file="system.DDB",
    ncell=(2, 2, 2),     # supercell size
    dipdip=True,         # dipole-dipole long-range correction
    asr=True,            # acoustic sum rule
)

# Get the reference supercell structure
atoms = calc.get_reference_atoms()

# Evaluate energy, forces, stress (eV, eV/Å, eV/Å³)
atoms.calc = calc
print(atoms.get_potential_energy())
print(atoms.get_forces())
print(atoms.get_stress())

# Optimize
opt = BFGS(atoms)
opt.run(fmax=0.01)

calc.close()
```

## Calculator Backends

### pyeffpot (pure Python, recommended)

Reads the DDB directly, builds IFCs with Fourier interpolation, applies the
dipole-dipole Ewald correction, and evaluates energy/forces/stress in Python.
No Fortran library required.

```python
calc = MultibinitCalculator.from_pyeffpot(
    ddb_file="system.DDB",
    xml_file="coeffs.xml",   # optional anharmonic coefficients
    ncell=(2, 2, 2),
    dipdip=True,
    asr=True,
)
```

### CFFI (Fortran, optional)

Wraps `libabinit.so` for bit-exact parity with the standalone MULTIBINIT
binary. Requires the ABINIT shared library.

```python
# From .abi input file
calc = MultibinitCalculator.from_abi("input.abi", lib_path="libabinit.so")

# Spawned (isolated child process, safe for repeated initialization)
calc = MultibinitCalculator.from_abi_spawned("input.abi", lib_path="libabinit.so")

# From direct parameters
calc = MultibinitCalculator.from_params(
    ddb_file="system_DDB",
    ncell=(2, 2, 2),
    ngqpt=(4, 4, 4),
)
```

`from_abi_spawned` runs the libabinit initialization in a child process,
avoiding the process-global state limitation where a second `from_abi` call
crashes after the first calculator is closed.

## Dipole-Dipole (dipdip) Correction

When `dipdip=True`, the long-range dipole-dipole contribution is subtracted
from the DDB dynamical matrix in q-space, Fourier-interpolated, then
re-added in real space via anisotropic Ewald summation (`ewald9`).

- **`ncell == ngqpt`**: subtraction and re-addition cancel exactly (energy-neutral).
- **`ncell > ngqpt`**: extends the dipdip to longer range, correctly modifying
  forces and energy.
- **`ncell < ngqpt`**: dipdip has no additional effect (the q-grid already
  resolves the supercell).

The Python Ewald implementation matches the Fortran `ewald9` kernel:
reciprocal-space term with dielectric-weighted norm $K_\mu \varepsilon_{\mu\nu} K_\nu$,
real-space term with anisotropic screened distance, and self-interaction
correction.

## Energy / Force / Stress Decomposition

The pyeffpot backend can split the total energy, forces, and stress into
per-term contributions:

```python
atoms.calc = calc
contrib = calc.get_contributions(atoms)

contrib.energy["dipdip"]            # eV
contrib.forces["harmonic_local"]    # (natom, 3), eV/Å
contrib.stress["anharmonic"]        # (6,) Voigt, eV/Å³

contrib.total_forces()   # == atoms.get_forces(), exactly
```

See [`docs/ENERGY_FORCE_STRESS_DECOMPOSITION.md`](docs/ENERGY_FORCE_STRESS_DECOMPOSITION.md).

## DDB → Phonopy Export

Export ABINIT DDB harmonic data to `phonopy_params.yaml` without running ABINIT:

```python
from pymultibinit import write_phonopy_from_ddb

result = write_phonopy_from_ddb("system_DDB", "phonopy_from_ddb")
```

```bash
mbtools ddb-to-phonopy system_DDB phonopy_from_ddb
```

## Model Training

### Pure-Python fitting

```bash
mbtools train-python system.ddb training_HIST.nc \
  --basis-xml candidate_basis.xml \
  --output-xml fit_coeffs.xml \
  --ncell 2 2 2 \
  --selection greedy \
  --ncoeff 20
```

See [`docs/PURE_PYTHON_TRAINING.md`](docs/PURE_PYTHON_TRAINING.md).

### Binary-based training

```bash
mbtools train system.ddb training_HIST.nc \
  --config train.abi \
  --output-dir model_out \
  --executable /path/to/multibinit
```

## API Reference

### `MultibinitCalculator` (ASE interface)

| Constructor | Backend | Requires |
|---|---|---|
| `from_pyeffpot(ddb_file, ...)` | Pure Python | DDB file |
| `from_abi(abi_file, ...)` | CFFI | libabinit |
| `from_abi_spawned(abi_file, ...)` | CFFI (child process) | libabinit |
| `from_params(ddb_file, ...)` | CFFI | libabinit |
| `from_config_file(config_file)` | CFFI | libabinit |

Key methods:

- `get_reference_atoms()` → ASE `Atoms` for the reference supercell
- `calculate(atoms)` → energy, forces, stress (called automatically by ASE)
- `close()` / context manager → release resources
- `get_contributions(atoms)` → per-term decomposition (pyeffpot only)

### Unit Conventions

| Quantity | API |
|---|---|
| Length | Angstrom |
| Energy | eV |
| Force | eV/Å |
| Stress | eV/Å³ (Voigt: xx, yy, zz, yz, xz, xy) |

## Examples

- `examples/contributions_decomposition/` — energy/force/stress decomposition
- `debugs/BFO_arijit_harmonic_update/compare_3x3x3_rattle.py` — three-backend BFO parity
- `debugs/BFO_arijit_harmonic_update/compare_hist_pyeffpot.py` — HIST energy comparison

## License

GPL v3 (same as ABINIT)

## Citation

- ABINIT: https://www.abinit.org
- MULTIBINIT: https://docs.abinit.org/guide/multibinit/
