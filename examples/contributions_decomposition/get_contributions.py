#!/usr/bin/env python
"""
Example: decomposing the MULTIBINIT energy / forces / stress by physical term.

The effective potential energy, forces and stress can be written as a sum of
contributions:

    (a) dipole-dipole       -- long-range Coulomb (Born effective charges)
    (b) local harmonic IFC  -- short-range interatomic force constants
    (c) anharmonic          -- higher-order coefficients from an XML fit

plus, when the model contains them, a constant reference energy, an elastic
(strain) term, and a phonon-strain coupling term.

This script shows the ASE-compatible API that returns every term separately,
and verifies that the per-term arrays sum back to the standard ASE getters.

How to run
----------
    pip install pymultibinit ase numpy scipy
    python get_contributions.py

(No Fortran / libabinit needed: this uses the pure-Python "pyeffpot" backend.)
"""

import numpy as np

from pymultibinit import MultibinitCalculator, Contributions

DDB_FILE = "BaHfO3_DDB"
XML_FILE = "BaHfO3.xml"
NCELL = (2, 2, 2)


def banner(text):
    print("\n" + "=" * 64)
    print(text)
    print("=" * 64)


def main():
    # ------------------------------------------------------------------ #
    # 1. Build the ASE calculator from a DDB + anharmonic XML fit.
    #    dipdip=True  -> include the dipole-dipole (long-range) term.
    #    The XML file  -> supplies the anharmonic coefficients.
    # ------------------------------------------------------------------ #
    banner("1. Build the calculator (pure-Python pyeffpot backend)")
    calc = MultibinitCalculator.from_pyeffpot(
        ddb_file=DDB_FILE,
        xml_file=XML_FILE,
        ncell=NCELL,
        dipdip=True,
    )

    # The equilibrium supercell, with the calculator attached.
    atoms = calc.get_reference_atoms()
    atoms.calc = calc

    # ------------------------------------------------------------------ #
    # 2. Standard ASE getters at equilibrium.
    #    With zero displacement the only non-zero energy is the reference.
    # ------------------------------------------------------------------ #
    banner("2. Equilibrium structure (standard ASE API)")
    print(f"  energy        = {atoms.get_potential_energy(): .6f} eV")
    print(f"  max |force|   = {np.max(np.abs(atoms.get_forces())):.3e} eV/Ang")

    # ------------------------------------------------------------------ #
    # 3. Displace atoms + apply a small strain so every term is active.
    # ------------------------------------------------------------------ #
    banner("3. Displace atoms and apply a small strain")
    rng = np.random.default_rng(42)
    atoms.set_positions(
        atoms.get_positions() + rng.normal(scale=0.04, size=atoms.positions.shape)
    )
    cell = atoms.get_cell().array.copy()
    cell[1, 1] *= 1.005
    atoms.set_cell(cell, scale_atoms=True)

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    stress = atoms.get_stress(voigt=True)

    print(f"  energy        = {energy: .6f} eV")
    print(f"  max |force|   = {np.max(np.abs(forces)):.6f} eV/Ang")

    # ------------------------------------------------------------------ #
    # 4. THE DECOMPOSITION API.
    #    get_contributions() returns a Contributions object whose per-term
    #    dicts add up to the ASE getters above.
    # ------------------------------------------------------------------ #
    banner("4. Energy / force / stress decomposition")
    contrib = calc.get_contributions(atoms)

    print(f"\n  terms present: {list(contrib.terms)}")
    print("\n  Per-term contributions:")
    print("  " + "-" * 60)
    print(f"  {'term':<18s}{'energy (eV)':>14s}{'max|F| (eV/Ang)':>18s}"
          f"{'stress_xx (eV/Ang^3)':>22s}")
    print("  " + "-" * 60)
    for term in contrib.terms:
        print(f"  {term:<18s}{contrib.energy[term]:>14.6f}"
              f"{np.max(np.abs(contrib.forces[term])):>18.6f}"
              f"{contrib.stress[term][0]:>22.6e}")

    # Map to the (a)/(b)/(c) labels from the theory:
    print("\n  Physical interpretation:")
    print("    (a) dipole-dipole      -> 'dipdip'")
    print("    (b) local harmonic IFC -> 'harmonic_local'")
    print("    (c) anharmonic         -> 'anharmonic'")

    # ------------------------------------------------------------------ #
    # 5. Accessing individual terms (the whole point).
    # ------------------------------------------------------------------ #
    banner("5. Accessing individual terms")
    e_dd = contrib.energy["dipdip"]
    f_local = contrib.forces["harmonic_local"]
    s_anh = contrib.stress["anharmonic"]
    print(f"  dipole-dipole energy            = {e_dd: .6f} eV")
    print(f"  local-harmonic force on atom 0  = {f_local[0]} eV/Ang")
    print(f"  anharmonic stress (Voigt)       = {s_anh}")

    # ------------------------------------------------------------------ #
    # 6. Verify the sum reproduces the ASE getters exactly.
    # ------------------------------------------------------------------ #
    banner("6. Check: sum of contributions == ASE getters")
    d_energy = contrib.total_energy() - energy
    d_forces = np.max(np.abs(contrib.total_forces() - forces))
    d_stress = np.max(np.abs(contrib.total_stress() - stress))
    print(f"  d(energy) = {d_energy:.3e} eV")
    print(f"  d(forces) = {d_forces:.3e} eV/Ang   (max abs)")
    print(f"  d(stress) = {d_stress:.3e} eV/Ang^3 (max abs)")
    assert d_forces < 1e-6, "force decomposition does not sum to total"
    print("\n  -> decomposition is exact.")

    # ------------------------------------------------------------------ #
    # 7. Bonus: the lower-level potential API (no ASE Atoms needed).
    #    Returns a plain dict {term: (energy, forces, stress)} in eV units.
    # ------------------------------------------------------------------ #
    banner("7. Bonus: potential-level API (dict, same units)")
    raw = calc.potential.evaluate_contributions(
        atoms.get_positions(), atoms.get_cell().array
    )
    print(f"  keys: {sorted(raw)}")
    e, f, s = raw["dipdip"]
    print(f"  raw['dipdip'] -> energy={e:.6f} eV, forces shape={f.shape}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
