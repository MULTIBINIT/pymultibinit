# Energy / Force / Stress Decomposition Guide

This guide explains how to decompose the MULTIBINIT effective-potential energy,
forces, and stress into their physical contributions.

## Overview

The total energy, forces, and stress can be written as a sum of terms:

| key               | physical meaning                              | theory label |
|-------------------|-----------------------------------------------|--------------|
| `dipdip`          | dipole-dipole (long-range Coulomb, Ewald)     | **(a)**      |
| `harmonic_local`  | local (short-range) harmonic interatomic force constants | **(b)** |
| `anharmonic`      | higher-order coefficients (from an XML fit)   | **(c)**      |
| `reference`       | constant equilibrium energy                   | —            |
| `elastic`         | homogeneous elastic / strain term             | —            |
| `strain_coupling` | phonon-strain coupling (when present)         | —            |

Only the contributions that are active for a given model appear as keys.
Summed over **all** keys, the arrays reproduce `get_potential_energy()`,
`get_forces()`, and `get_stress()` exactly.

This is useful for:

- **Physical analysis** — see how much of the energy or force comes from the
  short-range harmonic part vs. the long-range dipole-dipole part vs. the
  anharmonic part.
- **Model debugging** — check which term dominates the forces or drives an
  instability.
- **Validation** — verify that a fitted anharmonic model contributes sensibly
  relative to the harmonic baseline.

## Quick Start

```python
from pymultibinit import MultibinitCalculator

# Pure-Python backend (no libabinit required)
calc = MultibinitCalculator.from_pyeffpot(
    ddb_file="system.DDB",
    xml_file="model.xml",   # anharmonic coefficients (optional)
    ncell=(2, 2, 2),
    dipdip=True,
)

atoms = calc.get_reference_atoms()
atoms.calc = calc

# Standard ASE getters give the totals
energy = atoms.get_potential_energy()   # eV
forces = atoms.get_forces()             # (natom, 3), eV/Angstrom
stress = atoms.get_stress(voigt=True)   # (6,), eV/Angstrom^3

# The decomposition
contrib = calc.get_contributions(atoms)

print(contrib.terms)
# ('reference', 'harmonic_local', 'dipdip', 'elastic', 'anharmonic')

# Per-term dicts (same units as the ASE getters)
contrib.energy["dipdip"]            # float, eV
contrib.forces["harmonic_local"]    # (natom, 3), eV/Angstrom
contrib.stress["anharmonic"]        # (6,) Voigt, eV/Angstrom^3
```

## The `Contributions` object

`get_contributions()` returns a `Contributions` dataclass with three dicts
keyed by term name, plus total accessors:

```python
from pymultibinit import Contributions

contrib = calc.get_contributions(atoms)

contrib.energy   # dict[str, float]            -- eV
contrib.forces   # dict[str, np.ndarray]       -- (natom, 3), eV/Angstrom
contrib.stress   # dict[str, np.ndarray]       -- (6,) Voigt, eV/Angstrom^3
contrib.terms    # tuple[str, ...]             -- ordered term names

# Sum of all terms == the ASE getters, exactly:
contrib.total_energy() == atoms.get_potential_energy()
np.allclose(contrib.total_forces(), atoms.get_forces())
np.allclose(contrib.total_stress(), atoms.get_stress(voigt=True))
```

Forces are returned in the **input atom order**, exactly like `get_forces()`.

## Analyzing individual terms

```python
# How much does each term contribute to the energy?
for term in contrib.terms:
    print(f"{term:>16s}  {contrib.energy[term]:+.6f} eV")

# The force on atom 0 from each mechanism
for term in contrib.terms:
    print(f"{term:>16s}  {contrib.forces[term][0]}")

# Dipole-dipole contribution to the stress tensor (Voigt)
print(contrib.stress["dipdip"])
```

## Three API layers

The decomposition is available at three levels, from ASE-friendly to low-level:

### 1. ASE calculator (recommended)

```python
contrib = calc.get_contributions(atoms)   # -> Contributions dataclass
```

The result is also cached in `calc.results["contributions"]`.

### 2. Potential (dict, no ASE Atoms needed)

Returns the same decomposition as a plain dict
`{term: (energy, forces, stress)}` in eV / eV-Angstrom / eV-Angstrom^3 units:

```python
from pymultibinit import MultibinitPotential

pot = MultibinitPotential.from_pyeffpot("system.DDB", "model.xml", ncell=(2, 2, 2))
raw = pot.evaluate_contributions(positions, lattice)   # -> dict
energy, forces, stress = raw["dipdip"]
```

### 3. Pure-Python backend (atomic units)

At the lowest level, the `EffectivePotential` evaluates each term in atomic
units (Hartree, Bohr). Stress is returned as a full (3, 3) tensor here:

```python
from pymultibinit.pyeffpot import EffectivePotential

ep = pot._pyeffpot_potential
raw = ep.evaluate_contributions(xcart, rprimd)   # Hartree / Bohr
```

## How the harmonic term is split

The harmonic interatomic force constants combine two parts in the supercell:

- **local** (`atmfrc`): short-range force constants from the DDB.
- **dipdip** (`ewald_atmfrc`): long-range dipole-dipole (Ewald) force constants
  built from the Born effective charges and dielectric tensor.

The decomposition builds two separate force-constant matrices and applies the
acoustic-sum-rule diagonal correction to each independently. Because the
correction is linear, the two parts sum **exactly** to the combined
ASR-corrected harmonic matrix, so:

```
harmonic_local + dipdip == harmonic (combined)
```

For every term, the mass-weighted residual-force projection (which enforces
zero net force / translational invariance) is applied to that term alone; since
it is linear in the forces, the per-term results still sum to the total.

## Backend support

| backend          | decomposition? | notes                                   |
|------------------|:--------------:|-----------------------------------------|
| `pyeffpot` (pure Python) | ✅ | Default for `from_pyeffpot`. Supports all terms. |
| CFFI (Fortran)   | ❌ | The Fortran library does not expose the split. Raises `NotImplementedError`. |
| spawned process  | ❌ | Same; raises `NotImplementedError`. |

Term presence is controlled by the model:

- Omit `xml_file` (pass `None`) → no `anharmonic` term.
- Set `dipdip=False` → no `dipdip` term.
- Models without elastic constants → no `elastic` term.

## Runnable example

A complete, self-contained example is in:

```
examples/contributions_decomposition/
├── get_contributions.py   # runnable, annotated script
├── BaHfO3_DDB
└── BaHfO3.xml
```

```bash
pip install pymultibinit ase numpy scipy
python examples/contributions_decomposition/get_contributions.py
```

## See also

- [Supercell Structure Export](SUPERCELL_EXPORT.md)
- [Pure-Python Model Training](PURE_PYTHON_TRAINING.md)
- ASE calculator API: the `MultibinitCalculator` section of the README
