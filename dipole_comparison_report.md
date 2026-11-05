# Comparing harmonic and geometry-dependent dipole interactions

## Purpose and conclusions

This report compares the **dipole–dipole contribution** to energy and forces in two `pymultibinit` models. For the BaHfO₃ example and displacement pattern below, they agree very closely at small displacements; their difference grows at larger displacements. This is a comparison between two models, **not a measurement of error against first-principles calculations**.

The harmonic model is the default. The optional geometry model recomputes a periodic screened point-dipole interaction at each structure, using fixed reference Born effective charges and dielectric tensor. Its dipoles are `p_i = Z_i @ (u_i - mean(u))`; the zero reciprocal-space mode is omitted. The implemented dipole contribution retains the original harmonic IFC energy and adds the current-minus-reference geometry point-dipole energy at the same displacements. Therefore its harmonic Hessian matches the original **by construction**; small-displacement agreement checks implementation consistency, not independent physical accuracy.

## Requirements

- A `pymultibinit` checkout containing commit `825a91e` or a descendant, and its Python dependencies. The pure-Python backend does not require libabinit. From the checkout root, install with `pip install -e .` in your chosen environment.
- The bundled DDB `examples/BaHfO3_example/BaHfO3_DDB`. Use the same file to reproduce these numbers. No XML anharmonic model is used here.
- Run from the repository root. `JAX_PLATFORMS=cpu` can be set to avoid GPU initialization when JAX is installed.

## Reproduce the comparison

Save the following as `compare_dipoles.py` outside or inside the checkout and run `JAX_PLATFORMS=cpu python compare_dipoles.py` from the checkout root. The atom ordering is the DDB's reference-supercell ordering; keep it unchanged. The displacement amplitude is the **maximum Euclidean displacement of any atom** in Å, not the RMS amplitude.

```python
import numpy as np
from pymultibinit import MultibinitPotential

DDB = "examples/BaHfO3_example/BaHfO3_DDB"
options = dict(ddb_file=DDB, ncell=(1, 1, 1), dipdip=True)
harmonic = MultibinitPotential.from_pyeffpot(**options, dipdip_model="harmonic")
geometry = MultibinitPotential.from_pyeffpot(**options, dipdip_model="geometry")
positions, cell, _ = geometry.get_supercell_structure()  # Å; cell vectors are rows

# Rows correspond to atoms in the returned reference ordering.
pattern = np.array([
    [ 0.08, -0.03,  0.01],
    [-0.05,  0.04,  0.02],
    [ 0.02,  0.01, -0.03],
    [ 0.00, -0.02,  0.05],
    [-0.05,  0.00, -0.05],
])
pattern /= np.max(np.linalg.norm(pattern, axis=1))


def dipdip(potential, displaced):
    return potential.evaluate_contributions(
        displaced, cell, skip_atom_matching=True
    )["dipdip"]  # energy eV, force eV/Å, stress eV/Å³


def rms(array):
    return np.sqrt(np.mean(array**2))  # over all atom and Cartesian components


print("amplitude_A E_h_eV E_g_eV deltaE_meV Fh_rms_eV_A deltaF_rms_eV_A deltaF_percent")
for amplitude in [0.001, 0.01, 0.05, 0.1, 0.2, 0.4, 0.6]:
    displaced = positions + amplitude * pattern
    eh, fh, _ = dipdip(harmonic, displaced)
    eg, fg, _ = dipdip(geometry, displaced)
    df = rms(fg - fh)
    print(f"{amplitude:.3f} {eh:.9g} {eg:.9g} {1000*(eg-eh):.8g} "
          f"{rms(fh):.8g} {df:.8g} {100*df/rms(fh):.6g}")
```

`skip_atom_matching=True` is appropriate here only because positions stay in the reference atom order. For independently supplied structures with unknown ordering, use the normal atom-matching behavior instead and check its tolerance. Both models must receive the same positions, lattice, DDB, supercell and other settings.

## Observed results

Energy difference is `E_geometry - E_harmonic`, in meV per simulated 1×1×1 cell. Relative force difference is `100 × RMS(F_geometry - F_harmonic) / RMS(F_harmonic)`, with the RMS over all atom and Cartesian components. Energies here are the isolated `dipdip` contribution, not total potential energies; a negative contribution is not by itself an error.

| Maximum displacement (Å) | Harmonic energy (eV/cell) | Geometry energy (eV/cell) | Energy difference (meV/cell) | Force RMS difference (eV/Å) | Relative force RMS difference |
|---:|---:|---:|---:|---:|---:|
| 0.001 | −0.0000114312403 | −0.0000114312385 | 0.00000000178 | 0.0000000172 | 0.000128% |
| 0.010 | −0.00114312403 | −0.00114310618 | 0.0000178 | 0.0000172 | 0.0128% |
| 0.050 | −0.0285781008 | −0.0285669463 | 0.0112 | 0.00214 | 0.320% |
| 0.100 | −0.114312403 | −0.114133906 | 0.178 | 0.0171 | 1.28% |
| 0.200 | −0.457249612 | −0.454391039 | 2.86 | 0.136 | 5.09% |
| 0.400 | −1.82899845 | −1.78287862 | 46.1 | 1.070 | 20.0% |
| 0.600 | −4.11524651 | −3.87544943 | 239.8 | 3.497 | 43.5% |

These are results for **one direction**, not universal displacement thresholds. Numerical values may vary slightly with dependencies and platform. Relative force errors are ill-conditioned if the harmonic force RMS approaches zero; inspect the absolute difference too.

## Longitudinal- and transverse-like distortions

The preceding five-atom pattern is an arbitrary test distortion, **not a phonon eigenmode**. As a directional comparison, choose a notional wavevector along Cartesian x. Displace atom 0 by `-A` and atom 1 by `+A`, leaving the other atoms fixed. For the longitudinal-like case use the x direction; for the transverse-like case use y. This is a zone-center, cell-uniform relative displacement, not a computed LO or TO eigenvector.

For the same BaHfO₃ 1×1×1 model and units as above, both directions gave identical values to the printed precision:

| Amplitude (Å) | Harmonic energy (eV/cell) | Geometry energy (eV/cell) | Energy difference (meV/cell) | Force RMS difference (eV/Å) | Relative force RMS difference |
|---:|---:|---:|---:|---:|---:|
| 0.001 | −0.00000902 | −0.00000902 | approximately 0 | <0.000001 | 0.00011% |
| 0.010 | −0.00090215 | −0.00090227 | −0.000120 | 0.000013 | 0.01067% |
| 0.050 | −0.02255366 | −0.02262828 | −0.074614 | 0.001662 | 0.26691% |
| 0.100 | −0.09021466 | −0.09140438 | −1.189717 | 0.013309 | 1.06847% |
| 0.200 | −0.36085864 | −0.37962791 | −18.769273 | 0.106833 | 4.28833% |
| 0.400 | −1.44343456 | −1.72590591 | −282.471353 | 0.871444 | 17.49012% |

To reproduce, use the script above and replace `pattern` with a zero array of shape `(5, 3)`; set `pattern[:, axis] = [-1, 1, 0, 0, 0]`, using `axis=0` for the longitudinal-like and `axis=1` for the transverse-like displacement. No further normalization is necessary. Use amplitudes `[0.001, 0.01, 0.05, 0.1, 0.2, 0.4]`.

These results do **not** resolve physical LO–TO splitting: the displacement is uniform in a 1×1×1 cell, so the notional wavevector is not actually present, and the Ewald zero-wavevector term is omitted. A true LO/TO comparison needs either directional nonanalytic q→0 dynamical matrices with the corresponding eigenvectors and boundary conditions, or finite-q modes in a commensurate supercell. Compare those eigenvectors at the same q direction; do not infer a physical LO–TO splitting from the identical directional values above.

## Finite-q comparison: q = (1/8, 0, 0)

Using the BaHfO₃ DDB, an 8×1×1 supercell and the harmonic supercell IFC matrix, Fourier-transform the blocks between cell 0 and cells j=0,…,7 with `exp(2π i j/8)`, mass-weight, and diagonalize. The DDB q-grid is 4×4×4, so this wavevector is interpolated. For each mass-weighted eigenvector, divide by square-root atomic masses, calculate its Born-charge-weighted mode polarization, and select the lowest eigenvalue after excluding the three lowest acoustic-like modes with polarization more than 80% parallel to x (LO) or less than 20% parallel to x (TO). This yields zero-based eigenvector indices 5 and 3 respectively; the TO is degenerate, so the eigensolver can choose a different linear combination in that subspace.

The real supercell distortion in cell j is `Re(e_i exp(2π i j/8))`, where `e_i` is the complex Cartesian displacement eigenvector. It is normalized so its largest atomic displacement across the supercell is the indicated amplitude. This is a frozen real standing wave at finite q, not a q→0 nonanalytic-boundary calculation. No XML terms were used. Energies below are **dipole contributions for the entire 8-cell supercell**, not per primitive cell.

| Mode | Maximum displacement (Å) | Harmonic energy (eV/supercell) | Geometry energy (eV/supercell) | Geometry − harmonic (meV/supercell) | Force RMS difference (eV/Å) | Relative force RMS difference |
|---|---:|---:|---:|---:|---:|---:|
| LO | 0.001 | −0.00000959509 | −0.00000959510 | −0.00000000650 | 0.00000149 | 0.0282% |
| LO | 0.01 | −0.000959509 | −0.000959574 | −0.0000650 | 0.000149 | 0.282% |
| LO | 0.05 | −0.0239877 | −0.0240282 | −0.0404 | 0.00374 | 1.41% |
| LO | 0.10 | −0.0959509 | −0.0965900 | −0.639 | 0.0151 | 2.85% |
| LO | 0.20 | −0.383804 | −0.393521 | −9.72 | 0.0625 | 5.90% |
| TO | 0.001 | 0.000000881585 | 0.000000881596 | 0.0000000105 | 0.000000715 | 0.0928% |
| TO | 0.01 | 0.0000881585 | 0.0000882639 | 0.000105 | 0.0000716 | 0.930% |
| TO | 0.05 | 0.00220396 | 0.00226981 | 0.0658 | 0.00187 | 4.86% |
| TO | 0.10 | 0.00881585 | 0.00986859 | 1.053 | 0.00843 | 10.95% |
| TO | 0.20 | 0.0352634 | 0.0520586 | 16.80 | 0.0459 | 29.78% |

The acoustic/optical assignment here uses eigenvalue ordering at this q; branch tracking from Γ and convergence with respect to the DDB q-grid were not checked. The mode polarization is used for classification, not as a substitute for those checks. These differences compare two effective models and do not establish which is closer to first-principles forces.

## Using the models on your own structures

For the ASE interface, select the model at construction:

```python
from pymultibinit import MultibinitCalculator

calc = MultibinitCalculator.from_pyeffpot(
    "system.DDB", ncell=(2, 2, 2), dipdip=True,
    dipdip_model="geometry",  # or "harmonic"; harmonic is the default
)
atoms = calc.get_reference_atoms()
atoms.calc = calc
energy = atoms.get_potential_energy()  # total, eV
forces = atoms.get_forces()            # total, eV/Å
parts = calc.get_contributions(atoms)
print(parts.energy["dipdip"], parts.forces["dipdip"])
```

For meaningful validation, compare multiple displacement patterns and amplitudes; check multiple cells and Ewald/supercell convergence. For cell deformation, compare stress as well. To assess physical accuracy, compare **total** energies and forces to independent first-principles calculations with matching structures and electric boundary conditions, ideally using configurations outside the fit set.

## Limitations

- The geometry option is a fixed-response point-dipole model, not exact electronic electrostatics: Born charges, dielectric screening, and higher multipoles are not recomputed from electronic structure at each geometry.
- Anharmonic XML coefficients fitted to first-principles data may already include nonlinear dipolar effects. Adding the geometry correction can double-count them; refit or validate the combined model.
- The normal backend wraps each fractional displacement relative to the reference. Large motion crossing an image branch can change the displacement representation; it is not a model for arbitrary diffusion or structural reconstruction.
- Geometry mode requires active dipole IFCs and Born/dielectric data, does not support standalone compatibility or analytic Hessian blocks, and is more expensive than the precomputed harmonic model.

Further details: `docs/ENERGY_FORCE_STRESS_DECOMPOSITION.md` and `dipole_dipole.md` in the checkout.
