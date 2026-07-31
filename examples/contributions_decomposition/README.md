# Energy / force / stress decomposition — example

This folder is a **self-contained, runnable example** of the ASE-compatible API
that splits the MULTIBINIT effective-potential energy, forces, and stress into
their physical contributions.

## What it shows

The total energy, forces, and stress can be written as a sum of terms:

| key               | physical meaning                          | theory label |
|-------------------|-------------------------------------------|--------------|
| `dipdip`          | dipole-dipole (long-range Coulomb, Ewald) | **(a)**      |
| `harmonic_local`  | local (short-range) harmonic IFCs         | **(b)**      |
| `anharmonic`      | higher-order coefficients (from XML fit)  | **(c)**      |
| `reference`       | constant equilibrium energy               | —            |
| `elastic`         | homogeneous elastic / strain term         | —            |
| `strain_coupling` | phonon-strain coupling (when present)     | —            |

Only the terms that are active for a given model appear as keys. Summed over all
keys, the arrays reproduce `get_potential_energy()`, `get_forces()`, and
`get_stress()` **exactly**.

## The API (one line)

```python
from pymultibinit import MultibinitCalculator

calc = MultibinitCalculator.from_pyeffpot("BaHfO3_DDB", "BaHfO3.xml", ncell=(2, 2, 2))
atoms = calc.get_reference_atoms()
atoms.calc = calc

contrib = calc.get_contributions(atoms)

contrib.energy["dipdip"]            # float, eV
contrib.forces["harmonic_local"]    # (natom, 3) array, eV/Ang
contrib.stress["anharmonic"]        # (6,) Voigt array, eV/Ang^3

contrib.total_energy() == atoms.get_potential_energy()   # exact
contrib.total_forces()  == atoms.get_forces()
contrib.total_stress()  == atoms.get_stress(voigt=True)
```

`get_contributions()` returns a `Contributions` dataclass with three dicts
(`energy`, `forces`, `stress`) keyed by term name, plus `total_energy()` /
`total_forces()` / `total_stress()` accessors.

## Files

| file                 | description                                   |
|----------------------|-----------------------------------------------|
| `get_contributions.py` | the runnable example script                  |
| `BaHfO3_DDB`         | input derivative database (harmonic + Born)   |
| `BaHfO3.xml`         | anharmonic coefficient fit                    |

## Run it

```bash
pip install pymultibinit ase numpy scipy
python get_contributions.py
```

No Fortran / `libabinit` is required — this uses the pure-Python **pyeffpot**
backend, which is the only backend that currently exposes per-term
contributions. (The CFFI/Fortran and spawned-process backends raise
`NotImplementedError` for `get_contributions`.)

## Notes

- The harmonic IFC term is split into `harmonic_local` and `dipdip` via two
  independently acoustic-sum-rule-corrected force-constant matrices; the two
  parts sum exactly to the combined harmonic matrix.
- Forces and stress are returned in the **input atom order**, exactly like the
  standard ASE getters.
- If the anharmonic XML is omitted (`xml_file=None`), the `anharmonic` term is
  simply absent. If `dipdip=False`, the `dipdip` term is absent.
