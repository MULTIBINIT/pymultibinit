# Configuration File Usage Guide

This guide explains how to use configuration files to initialize MULTIBINIT potentials in pymultibinit.

## Overview

Configuration files provide a convenient way to specify all the parameters needed to initialize a MULTIBINIT potential without hardcoding them in your Python scripts. This is especially useful for:

- **Reproducibility**: Keep all parameters in one place
- **Batch processing**: Different configs for different systems
- **Separation of concerns**: Keep code and data separate
- **Easy parameter changes**: No need to modify Python code

## Quick Start

### Minimal Configuration

Create a file `multibinit.conf`:

```
ddb_file: system_DDB
sys_file: system.xml
ncell: 2 2 2
```

Use it in Python:

```python
from pymultibinit import MultibinitCalculator
from ase import Atoms

# Create calculator from config file
calc = MultibinitCalculator.from_config_file("multibinit.conf")

# IMPORTANT: Structure must match the supercell size specified in ncell
# If ncell: 2 2 2, you need a 2×2×2 supercell (not unit cell)
unit_cell = Atoms(...)  # Unit cell
atoms = unit_cell * (2, 2, 2)  # Create 2×2×2 supercell to match ncell

atoms.calc = calc
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
```

## Configuration File Formats

### 1. Simple Format

The simplest format uses `key: value` pairs:

```
# Comments start with #
ddb_file: system_DDB
sys_file: system.xml
ncell: 2 2 2
ngqpt: 4 4 4
dipdip: 1
```

Features:
- Use `:` or `=` as separator (both work)
- Comments with `#`
- Inline comments supported: `ncell: 2 2 2  # comment`
- Blank lines ignored

### 2. INI Format

More structured format with sections:

```ini
[files]
ddb_file = system_DDB
sys_file = system.xml

[parameters]
ncell = 2 2 2
ngqpt = 4 4 4
dipdip = 1

[backend]
backend = ctypes
auto_match_atoms = true
```

Sections are optional and for organization only.

## ⚠️ Important: Supercell Size Requirement

**The structure you pass to the calculator MUST match the `ncell` parameter in your config file.**

- If `ncell: 2 2 2` → You need a **2×2×2 supercell** (40 atoms for 5-atom unit cell)
- If `ncell: 3 3 3` → You need a **3×3×3 supercell** (135 atoms for 5-atom unit cell)

**Example:**
```python
# Config file has: ncell: 2 2 2
calc = MultibinitCalculator.from_config_file("config.conf")

# Build matching supercell
unit_cell = Atoms(...)  # 5 atoms
atoms = unit_cell * (2, 2, 2)  # 40 atoms ✓ CORRECT

# atoms = unit_cell  # 5 atoms ✗ WRONG - will fail!
```

**Why?** MULTIBINIT internally builds a supercell during initialization using `ncell`. Your input structure must have the same number of atoms as this internal supercell.

## Configuration Parameters

### Required Parameters (Choose One)

**Option 1: Use .abi file**
```
abi_file: path/to/input.abi
```

**Option 2: Use DDB + XML files**
```
ddb_file: path/to/system_DDB
sys_file: path/to/system.xml
```

### Optional File Parameters

```
# Coefficient XML file (if not included in sys_file)
coeff_file: path/to/coefficients.xml
```

### Supercell and Grid Parameters

```
# Supercell dimensions (default: 1 1 1)
# Can use spaces: "2 2 2" or commas: "2,2,2"
ncell: 2 2 2

# q-point grid (default: 1 1 1)
ngqpt: 4 4 4

# Dipole-dipole interactions (default: 1)
# 0 = off, 1 = on
dipdip: 1
```

### Backend Configuration

```
# Backend selection (default: ctypes)
# Options: ctypes, cffi
backend: ctypes

# Path to libabinit library (optional, auto-detected if omitted)
lib_path: /path/to/libabinit.so

# Use atomic units (default: false)
# false = Angstrom/eV (recommended for ASE)
# true = Bohr/Hartree
use_atomic_units: false
```

### Atom Matching Configuration

```
# Automatic atom matching (default: true)
# Recommended: keep enabled for safety
auto_match_atoms: true

# Matching tolerance in Angstrom (default: 0.1)
# Maximum distance for atom matching
match_tolerance: 0.1
```

## Path Resolution

- **Relative paths**: Resolved relative to the config file directory
- **Absolute paths**: Used as-is

Example:
```
# If config file is at /home/user/project/config.conf
ddb_file: data/system_DDB          # → /home/user/project/data/system_DDB
sys_file: ../shared/system.xml     # → /home/user/shared/system.xml
coeff_file: /abs/path/coeff.xml    # → /abs/path/coeff.xml
```

## Usage Examples

### Example 1: ASE Calculator

```python
from pymultibinit import MultibinitCalculator
from ase import Atoms
from ase.optimize import BFGS

# Build unit cell structure
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

# IMPORTANT: Create supercell matching ncell in config
# If config has ncell: 2 2 2, create 2×2×2 supercell
atoms = unit_cell * (2, 2, 2)

# Create calculator from config (which specifies ncell: 2 2 2)
calc = MultibinitCalculator.from_config_file("multibinit.conf")
atoms.calc = calc

# Optimize geometry
opt = BFGS(atoms)
opt.run(fmax=0.01)

# Get final energy
final_energy = atoms.get_potential_energy()
print(f"Optimized energy: {final_energy:.3f} eV")
```

### Example 2: Direct Potential Use

```python
from pymultibinit import MultibinitPotential
import numpy as np

# Load potential from config
pot = MultibinitPotential.from_config_file("multibinit.conf")

# Prepare structure
positions = np.array([...])  # Shape (natom, 3) in Angstrom
lattice = np.array([...])    # Shape (3, 3) in Angstrom

# Evaluate
energy, forces, stress = pot.evaluate(positions, lattice)

print(f"Energy: {energy:.3f} eV")
print(f"Forces:\n{forces}")
print(f"Stress: {stress}")

# Cleanup
pot.free()
```

### Example 3: Molecular Dynamics

```python
from pymultibinit import MultibinitCalculator
from ase.io import read
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase import units

# Load structure
atoms = read("initial.cif")

# Create calculator
calc = MultibinitCalculator.from_config_file("multibinit.conf")
atoms.calc = calc

# Initialize velocities (300 K)
MaxwellBoltzmannDistribution(atoms, temperature_K=300)

# Run MD
dyn = VelocityVerlet(atoms, timestep=1.0*units.fs)

def print_energy(a=atoms):
    """Print energy during MD"""
    epot = a.get_potential_energy()
    ekin = a.get_kinetic_energy()
    print(f"E_pot = {epot:.3f} eV, E_kin = {ekin:.3f} eV")

dyn.attach(print_energy, interval=10)
dyn.run(1000)  # 1000 steps = 1 ps
```

### Example 4: Batch Processing

```python
from pymultibinit import MultibinitCalculator
from ase.io import read
from pathlib import Path

# Process multiple systems
systems = ["system1", "system2", "system3"]

for system in systems:
    # Each system has its own config
    config_file = f"configs/{system}.conf"
    structure_file = f"structures/{system}.cif"
    
    # Load
    atoms = read(structure_file)
    calc = MultibinitCalculator.from_config_file(config_file)
    atoms.calc = calc
    
    # Compute
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    
    # Save results
    result_file = f"results/{system}_results.txt"
    with open(result_file, 'w') as f:
        f.write(f"Energy: {energy:.6f} eV\n")
        f.write(f"Forces:\n{forces}\n")
    
    print(f"{system}: E = {energy:.3f} eV")
```

### Example 5: Multiple Configurations

Compare different supercell sizes:

**config_2x2x2.conf:**
```
ddb_file: system_DDB
sys_file: system.xml
ncell: 2 2 2
```

**config_3x3x3.conf:**
```
ddb_file: system_DDB
sys_file: system.xml
ncell: 3 3 3
```

**config_4x4x4.conf:**
```
ddb_file: system_DDB
sys_file: system.xml
ncell: 4 4 4
```

Python code:
```python
from pymultibinit import MultibinitPotential

configs = ["config_2x2x2.conf", "config_3x3x3.conf", "config_4x4x4.conf"]

for config_file in configs:
    pot = MultibinitPotential.from_config_file(config_file)
    
    # ... perform calculations ...
    
    pot.free()
```

## Best Practices

1. **Use relative paths**: Makes configs portable across machines
2. **Keep configs versioned**: Store in git alongside code
3. **Document parameters**: Use comments to explain choices
4. **One config per system**: Don't try to reuse configs for different materials
5. **Enable auto_match_atoms**: Prevents atom ordering issues
6. **Use Angstrom/eV**: Matches ASE conventions (unless you have specific needs)

## Example Config Templates

See the `examples/` directory for:
- `multibinit_simple.conf` - Minimal simple format
- `multibinit_ini.conf` - Full INI format with all options

## Troubleshooting

### FileNotFoundError: Config file not found
```
FileNotFoundError: Configuration file not found: config.conf
```
**Solution**: Check the path to your config file

### ValueError: Must specify either 'abi_file' or 'ddb_file'
```
ValueError: Configuration must specify either 'abi_file' or 'ddb_file'
```
**Solution**: Add at least `ddb_file` or `abi_file` to your config

### ValueError: Expected 3 integers
```
ValueError: Expected 3 integers, got: 2 2
```
**Solution**: Ensure `ncell` and `ngqpt` have exactly 3 values:
```
ncell: 2 2 2  # Correct
ncell: 2 2    # Wrong
```

### ValueError: Invalid backend
```
ValueError: Invalid backend: invalid_name
```
**Solution**: Use `backend: ctypes` or `backend: cffi`

## API Reference

### MultibinitPotential.from_config_file()

```python
@classmethod
def from_config_file(cls, config_file: str) -> 'MultibinitPotential'
```

Create potential from configuration file.

**Parameters:**
- `config_file` (str): Path to configuration file

**Returns:**
- `MultibinitPotential`: Initialized potential instance

**Raises:**
- `FileNotFoundError`: If config file doesn't exist
- `ValueError`: If required parameters are missing or invalid

### MultibinitCalculator.from_config_file()

```python
@classmethod
def from_config_file(cls, config_file: str, **kwargs) -> 'MultibinitCalculator'
```

Create ASE calculator from configuration file.

**Parameters:**
- `config_file` (str): Path to configuration file
- `**kwargs`: Additional ASE Calculator arguments

**Returns:**
- `MultibinitCalculator`: Initialized calculator instance

**Raises:**
- `FileNotFoundError`: If config file doesn't exist
- `ValueError`: If required parameters are missing or invalid

## See Also

- `docs/ATOM_MATCHING_USAGE.md` - Atom matching and ordering
- `docs/DEVELOPER_GUIDE.md` - Developer documentation
- `docs/api/API_REFERENCE.md` - Complete API reference
