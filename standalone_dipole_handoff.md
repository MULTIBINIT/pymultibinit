# Standalone dipole–dipole package: implementation handoff

## Decision and scope

Create a small, independent Python package (working import name `dipole_lr`; choose/check the distribution name before release) that owns a reusable periodic Born-dipole Ewald energy/force/stress kernel and an ASE calculator adapter. It must not import `pymultibinit`, parse ABINIT DDB files, or require libabinit. `pymultibinit` remains the owner of DDB/XML parsing, IFC construction, and its existing user-facing APIs; it should integrate the new package through a narrow compatibility adapter.

Source snapshot for a fresh project: [MULTIBINIT/pymultibinit at 853247f](https://github.com/MULTIBINIT/pymultibinit/tree/853247fe53cd9aee36ce770ffe44b64f002886c6). Repository-relative paths below refer to that snapshot.

The safe first model is **3D-periodic, fixed-reference Born dipoles for a crystal**. The package may be composed with an arbitrary ASE MLIP calculator, but an additive composition is physically valid only when that MLIP was trained for the residual after subtracting the exact same long-range model. The wrapper cannot infer or repair training-target semantics. This is not a general-purpose electrostatics model for liquids, reactive systems, diffusion, heterogeneous dielectric interfaces, or arbitrary MLIPs.

Keep two use cases sharply separate:

1. **General residual-MLIP sum:** `E_total = E_MLIP,residual + E_dipole`. The MLIP was explicitly fit to `DFT − E_dipole` (and corresponding force/stress residuals).
2. **pymultibinit geometry correction:** the existing harmonic DDB dipole IFC term remains in place, and the geometry kernel contributes only a current-minus-reference correction. It is not the general additive formula above.

## What pymultibinit currently means

The normal DDB path computes a dipole–dipole dynamical matrix on the q-grid, subtracts it from the DDB dynamical matrices before Fourier transformation, and keeps the resulting short-range and Ewald IFC arrays separate (`src/pymultibinit/pyeffpot/supercell_builder.py:619-638,698-727,788-824`). `EffectivePotential` builds the harmonic dipole matrix once and evaluates it as `E = 1/2 uᵀ Φ_dd u`, `F = −Φ_dd u`; the default is this frozen harmonic model (`src/pymultibinit/pyeffpot/potential.py:91-117,555-591`; [`dipole_dipole.md`](dipole_dipole.md#what-happens-at-large-displacement)). It does not reevaluate the Ewald geometry, Born charges, or dielectric response at each configuration.

The opt-in `dipdip_model="geometry"` path evaluates a fixed-response point-dipole Ewald energy, but **does not replace or add a second full Ewald energy to the harmonic baseline**. Its energy correction is

\[
\Delta E_{\rm geom}=E_{\rm geom}(x,h,u)-E_{\rm geom}(x_{\rm ref},h_{\rm ref},u),
\]

and the returned dipole contribution is the existing `1/2 uᵀ Φ_dd u` **plus** this correction. The reference subtraction uses displacement-only force/stress derivatives; the reference stress is scaled to the current cell volume (`src/pymultibinit/pyeffpot/potential.py:273-288,616-636`). The correction starts beyond the shared harmonic Hessian, so the original harmonic baseline is retained without counting its quadratic response twice. Do not replace this with `E_geom(x,h,u)` on top of `Φ_dd`.

This geometry model uses `p_i = Z_i* @ (u_i − mean(u))`, reference Born tensors and fixed `epsilon_inf`, with the reciprocal `G=0` term omitted (`src/pymultibinit/pyeffpot/nonlinear_dipdip.py:1-18,41-70,115-117,195-269`). It is still an approximation: no updated Born charges, changing dielectric response, nonlinear polarization, charge transfer, or higher multipoles. Existing fitted XML anharmonic terms can already contain nonlinear electrostatic effects; validate or refit the combined model rather than assuming the added correction is independent (`docs/ENERGY_FORCE_STRESS_DECOMPOSITION.md:162-190`; [`dipole_comparison_report.md`](dipole_comparison_report.md#limitations)). Harmonic-versus-geometry agreement at small displacement checks consistency of the two constructions, not accuracy against DFT (`dipole_comparison_report.md:3-7,137-144`).

A code audit also found distinct kernel families that must not be casually treated as interchangeable. `dipdip.py` includes a simple direct kernel, Gamma Ewald and q-space dynamical-matrix routines; the anisotropic Ewald metric and cutoffs differ among these paths (`src/pymultibinit/pyeffpot/dipdip.py:14-70,78-419,588-735,994-1051`). The geometry energy kernel is separately implemented in `nonlinear_dipdip.py`. Extract the latter first; do not silently replace the DDB harmonic q-grid subtraction/reconstruction with the nonlinear energy routine. The normal DDB path keeps short-range and Ewald IFCs separate, while the IFC-only fallback appears to place the computed DD array into both `atmfrc` and `ewald_atmfrc` (`supercell_builder.py:1068-1100`). This is a source-level suspected double count, not a reproduced failure; explicitly resolve it with tests before claiming migration parity for that route.

## Proposed package interface

### Core numerical API

Keep the public seam small, array-based, and independent of ASE/pymultibinit. One proposed interface:

```python
from dipole_lr import BornDipoleModel, EwaldSettings

model = BornDipoleModel(
    born_charges=zeff,             # (N, 3, 3), e
    epsilon_inf=epsilon_inf,       # (3, 3), dimensionless
    ewald=EwaldSettings(tolerance=1e-12, eta=None),  # eta in Å⁻¹; None selects automatically
    boundary="tin-foil",
)
result = model.evaluate(
    positions=positions,           # (N, 3), Å
    cell=cell,                     # (3, 3), lattice vectors as rows, Å
    displacements=u,               # (N, 3), Å
    pbc=(True, True, True),
)
# result.energy: float, eV; result.forces: (N,3), eV/Å;
# result.stress: (3,3), eV/Å³; result is for the whole periodic cell.
```

Define the tensor convention explicitly: `born_charges[i, alpha, beta] = dP_alpha / du_(i,beta)`; hence `p_i[alpha] = sum_beta Z[i,alpha,beta] * u[i,beta]`. `epsilon_inf` is a Cartesian, symmetric positive-definite **electronic/high-frequency** dielectric tensor. Require finite correctly shaped inputs, nonsingular cell, no coincident distinct dipole sites, and Born-charge neutrality within a documented tolerance; do not silently coerce a bad dielectric or alter user tensors. The pmb DDB parser already adds ionic charges and applies a charge-acoustic-sum correction (`ddb_parser_complete.py:593-649`), so its adapter passes those corrected tensors through unchanged.

`EwaldSettings` must make accuracy/convergence control visible (tolerance plus optional splitting parameter `eta`, in inverse Å) and record the actual real/reciprocal cutoffs used. Provide deterministic results and convergence tests as tolerance is tightened. The legacy harmonic dynamical-matrix routine currently has finite integer-shell defaults `nreal=5`, `nrecip=8` (`dipdip.py:588-608`); the nonlinear geometry routine uses its own `eta=1 Bohr⁻¹` in dielectric-transformed Bohr coordinates and `tolerance=1e-13` (`nonlinear_dipdip.py:16-18,118-123`). Do not conflate or hide these legacy settings; compatibility mode should preserve the kernel/settings it replaces, while new standalone callers should have documented convergence semantics.

The core accepts displacements explicitly; it must not guess atom matching or infer a displacement branch. This makes reference choice and mapping visible and allows the pmb adapter to preserve its existing behavior. A standalone ASE reference adapter may accept `reference_atoms` and require equal atom count, element sequence, and site order; compute the reference sites affinely in the current cell and wrap fractional displacement components by nearest integer, matching the current crystal-reference convention. Document that this is a bounded-displacement crystal model, not diffusion. `pymultibinit` currently maps positions to the current cell and wraps componentwise in fractional coordinates; this is not always the shortest Cartesian image in a skew cell (`potential.py:481-516`). Preserve that mapping in its compatibility adapter rather than changing it during extraction.

### ASE adapter and MLIP composition

Provide an optional `dipole_lr.ase.BornDipoleCalculator` implementing ASE `energy`, `forces`, and `stress` for the core model. ASE is an optional package extra (`dipole_lr[ase]`); the NumPy/SciPy core must import without ASE. Use ASE's built-in composition rather than reimplementing its calculator cache/state protocol:

```python
from ase.calculators.mixing import SumCalculator
from dipole_lr.ase import BornDipoleCalculator

lr = BornDipoleCalculator(model, reference_atoms=reference_atoms)
atoms.calc = SumCalculator([residual_mlip, lr])
energy = atoms.get_potential_energy()  # eV, cell total
forces = atoms.get_forces()            # eV/Å
stress = atoms.get_stress()            # eV/Å³, ASE Voigt order
```

The user-facing convenience function should implement this same composition:

```python
from dipole_lr.ase import add_dipole_to_potential

dressed = add_dipole_to_potential(residual_mlip, model, reference_atoms, residual_model=True)
atoms.calc = dressed  # an ASE calculator; no changes to residual_mlip
```

This proposed function returns `SumCalculator([residual_mlip, BornDipoleCalculator(model, reference_atoms)])`; it does not mutate the supplied calculator or attach itself to an `Atoms` object. It must document and validate its declared residual-model assumption. The names are a proposal for the new project, not an existing implementation.

ASE calculators are black-box energy/force/stress providers attached as `atoms.calc`; `SumCalculator` sums terms and only exposes the intersection of child `implemented_properties` (ASE calculator and mixing docs below). Therefore the dipole calculator must provide each advertised property, and it must not fabricate zero stress when the base MLIP does not implement stress. Return forces in input atom order. The ASE stress vector uses `[xx, yy, zz, yz, xz, xy]`; document the sign/strain convention and verify stress by finite strain. In the API-level `result`, prefer a full symmetric `(3,3)` stress tensor and let the adapter convert to ASE Voigt. Use ASE units at this seam: eV and Å are ASE base units, stress is eV/Å³ (`ase/units.html`; current pmb `Contributions` documents matching units at `calculator.py:19-24`).

`SumCalculator` is a mechanical sum, not a physical safety check. Do not make a wrapper appear to validate training provenance. In documentation and example names, call the MLIP `residual_mlip`, and require the user to state that the training target was residual before composition. A future helper may reject a manifest not marked `target_partition="residual"`, but that metadata is only an assertion, not proof. A normal full-total-energy MLIP must not be augmented this way just because it is an ASE calculator.

## Physical assumptions and boundaries

The reference Born-dipole term approximates the leading long-range harmonic response of a polar **insulating crystal**. It uses fixed reference `Z*` and fixed electronic `epsilon_inf`; `Z*` is a polarization-response tensor, not an oxidation state or a static partial charge. The dielectric screening is electronic, not the static dielectric that includes ionic response. The model has no self-consistent charge equilibration, nonlinear response, permanent monopoles, higher multipoles, or metallic screening/Friedel tails. These assumptions follow the DFPT IFC decomposition described by Gonze and Lee and the ABINIT `anaddb dipdip` procedure (sources below; also `dipole_dipole.md:27-35`).

For the first standalone release, support **fully 3D periodic cells only** and reject `pbc` values other than `(True, True, True)`. ASE supports partial PBC (including slabs), so silently applying 3D Ewald to a slab, wire, or molecule is wrong. A 2D Ewald/open-boundary formulation is a separate physical feature, not a flag on the 3D sum. State the electrical boundary convention: `G=0` is omitted (conducting/tin-foil convention in the geometry implementation). Results can differ for a finite-shape/vacuum surface term, imposed macroscopic field, or directional non-analytic `q→0` boundary condition; the caller must match that convention to the reference data. Do not infer LO–TO splitting from a Gamma result with the zero mode removed.

The periodic Ewald sum is not by itself “full electrostatics.” It is a summation method for the chosen dipole kernel. Higher-multipole tails and changed electronic screening are outside this model. Fixed reference tensors and a fixed site map make the model unsuitable as-is for a liquid, reactive network, phase reconstruction, or long-range diffusion. A heterogeneous dielectric interface also has no single unambiguous bulk `epsilon_inf` in this approximation.

## MLIP training and double-counting contract

Most MLIPs supplied through ASE predict a fitted total potential-energy surface; whether a particular architecture learns long-range effects depends on its features, cutoffs, data, and explicit physics. Neither the name “MLIP” nor `SumCalculator` tells the wrapper what is already included. Adding a standalone dipole term to a full-DFT-target model can count the same electrostatics twice (or otherwise distort a fitted decomposition); subtracting the term only after inference does not undo what the MLIP learned.

The safe additive workflow is:

1. Fix one declared dipole model and convention (same `Z*`, `epsilon_inf`, reference mapping, Ewald tolerance, self term, `G=0`, and periodicity) before making training targets.
2. Evaluate that model on every DFT training configuration, including cells/strains, to obtain `E_dd`, `F_dd`, and `stress_dd`.
3. Train the short-range MLIP to `E_res = E_DFT − E_dd`, `F_res = F_DFT − F_dd`, and `stress_res = stress_DFT − stress_dd`, using consistent cell-total energies and the same stress convention.
4. Validate `MLIP_residual + dipole_model` against held-out DFT energies, forces, and stresses, including distortions/strains and supercell-size convergence. Compare absolute errors to the application target; agreement between two model variants is not independent validation.

This is not an unusual requirement: DeePMD's DPLR documentation explicitly defines `E = E_DP + E_ele` and fits its short-range DP to `E* − E_ele`. LES instead learns latent environment-dependent electrostatic quantities jointly from total energies/forces; those latent charges are not interchangeable with fixed DFPT Born tensors. Both examples show that long-range physics and the residual model must be co-designed and trained, rather than bolted onto a full-target model after training. If the MLIP already implements its own long-range electrostatics, use that model's documented partition; do not stack another Ewald term without demonstrating the residual target.

For `pymultibinit`, keep its existing harmonic IFC baseline and its public default unchanged. If geometry correction is enabled, verify that the XML anharmonic fit did not already absorb the same nonlinear contribution. If it did, refit against targets with that correction removed (or train/validate the full sum jointly); do not call the current-minus-reference rule a universal MLIP residual formula.

## Migration plan into pymultibinit

1. **Freeze the contract first.** Record the numerical kernel, units, tensor orientation, row/column lattice convention, displacement wrapping, `G=0` convention, stress definition, and Ewald cutoffs. Establish baseline fixtures from the current `nonlinear_dipdip.py` and harmonic DDB path before moving code.
2. **Extract only the reusable geometry-energy kernel initially.** Put the Ewald arithmetic and analytic derivatives in the standalone package. Keep DDB parsing, Born/dielectric extraction, q-grid interpolation, harmonic IFC subtraction/reconstruction, XML training, and site/reference mapping in `pymultibinit`. Do not merge the distinct functions in `dipdip.py` merely because they share the name “dipole–dipole.”
3. **Add a thin pmb adapter, not a second implementation.** It converts pmb atomic units/column lattice conventions to the standalone public eV/Å/row-cell contract and back, while passing the existing displacement array. Preserve validation and error behavior. Make the `pymultibinit` dependency on the new package explicit; an optional extra plus a clear import-time error for only `dipdip_model="geometry"` avoids burdening users of the unchanged harmonic default.
4. **Retain every current public entry point and default.** Preserve `MultibinitPotential.from_pyeffpot(..., dipdip_model="harmonic"|"geometry")`, `MultibinitCalculator.from_pyeffpot(...)`, top-level exports, energy/force/stress units, atom-order behavior, `get_contributions()` and the `dipdip` term meaning (`src/pymultibinit/potential.py`, `calculator.py`, `__init__.py`; `docs/ENERGY_FORCE_STRESS_DECOMPOSITION.md`). `standalone_compat` currently rejects geometry mode; keep that explicit unless the separately cached image-shift semantics are designed and tested.
5. **Check both construction routes.** Verify q-grid DDB subtraction/reconstruction and the IFC-only fallback independently; resolve the source-level suspected duplicate noted above before claiming both are supported. Test that default harmonic mode is numerically unchanged and geometry mode retains the existing baseline/correction formula.
6. **Update the pmb changelog and docs when integrating.** Document installation extra, exact physical model, limitations, and training residual requirements. Keep the standalone package's docs sufficient for users with no pmb checkout.

The current wrappers report energy in eV, forces in eV/Å, stress in eV/Å³; low-level `pyeffpot` uses Hartree/Bohr and its geometry kernel takes column lattice vectors in Bohr. Convert in one adapter and test the conversions rather than exposing pmb internal conventions in the new package (`calculator.py:19-24,557-586`; `nonlinear_dipdip.py:41-63`).

## Test matrix and release acceptance

| Area | Required tests / observable acceptance |
|---|---|
| Install and independence | Fresh install of core without importing `pymultibinit`, ABINIT, or ASE; ASE extra installs the adapter. Core public API uses documented array shapes/units only. |
| Input contract | Reject malformed/non-finite arrays, singular cells, non-SPD/asymmetric dielectric, coincident distinct sites, non-neutral Born tensors outside tolerance, and unsupported partial/non-periodic PBC with actionable errors. |
| Kernel reference | Compare energies/forces to an independent implementation or high-precision reference for isotropic and anisotropic dielectric, orthogonal and skew cells; include a neutral simple case. Check self term and zero-mode convention explicitly. |
| Ewald convergence | Vary tolerance/splitting/cutoffs and show stable energy, force, stress; report actual cutoff settings. Verify splitting-parameter invariance within declared tolerance. |
| Analytic derivatives | Finite-difference energy vs every position and displacement component; finite-strain stress with the documented affine convention. Check symmetric stress, zero net force, translation invariance, zero displacement baseline, and Hessian/IFC agreement as displacement tends to zero. |
| Symmetry and mapping | Atom permutation with matching tensor/site permutation gives permuted forces and unchanged energy. Check reference ordering and periodic-image wrapping in skew cells; a displacement crossing the supported image branch must be documented, not silently treated as diffusion. |
| ASE behavior | Direct getters and `SumCalculator` agree with component sums for energy, forces, and stress; properties are the child-property intersection; positions/cell/PBC changes invalidate cached results; unsupported stress is not returned as zero. |
| Double-count guardrails | A synthetic residual target satisfies `E_res + E_dd = E_DFT` and likewise for force/stress. A full-target MLIP example is explicitly documented as unsafe to add. Confirm composition adds the DD contribution exactly once. |
| pmb parity | `dipdip_model="harmonic"` is unchanged. On existing DDB fixtures (e.g. `examples/BaHfO3_example/BaHfO3_DDB`), geometry energy/forces/stress and `Contributions` match the pre-extraction behavior, retain the harmonic Hessian, and finite-difference consistently. Test the normal and IFC-only builder routes separately. |
| Physical validation | Held-out first-principles configurations test total energy/forces/stress and intended strains/displacements. A kernel parity test or harmonic/geometry comparison alone is not described as validation of physical accuracy. |
| Licensing | License and notices are reviewed before copying code or publishing a license for the new distribution. |

Release is acceptable only when the standalone package works without pmb, the above numerical and ASE contracts pass, the pmb API/defaults retain their semantics, MLIP residual training requirements and unsupported physical regimes are explicit, and code provenance/license is settled. Do not claim support for arbitrary MLIPs, slabs, diffusion, or changing-charge systems in the first release.

## Licensing and provenance

The repository's [`LICENSE`](LICENSE) is GNU GPL version 3. Reusing or moving implementation code from `src/pymultibinit/pyeffpot/nonlinear_dipdip.py` or `dipdip.py` into a separately distributed package carries source-license/provenance consequences; keep GPL-compatible licensing and required notices unless the copyright holder grants another license. Do not advertise a permissive standalone license based only on the equations being published. If a permissive package is required, obtain authorization or have counsel review an independent implementation from the published equations before release. Also record provenance for example DDB/XML fixtures before redistributing them with the new package.

## Primary sources and repository anchors

### Repository

- [`src/pymultibinit/pyeffpot/potential.py`](src/pymultibinit/pyeffpot/potential.py): harmonic matrices, displacement wrapping, geometry correction (especially lines 91-130, 273-288, 481-516, 616-636).
- [`src/pymultibinit/pyeffpot/nonlinear_dipdip.py`](src/pymultibinit/pyeffpot/nonlinear_dipdip.py): current finite-geometry point-dipole Ewald, fixed dielectric/Born tensors, derivative and boundary conventions.
- [`src/pymultibinit/pyeffpot/dipdip.py`](src/pymultibinit/pyeffpot/dipdip.py): DDB harmonic IFC/dynamical-matrix kernels and finite-shell defaults.
- [`src/pymultibinit/pyeffpot/supercell_builder.py`](src/pymultibinit/pyeffpot/supercell_builder.py): q-space subtraction, local/Ewald reconstruction, and IFC-only fallback.
- [`src/pymultibinit/pyeffpot/ddb_parser_complete.py`](src/pymultibinit/pyeffpot/ddb_parser_complete.py): DDB Born-charge/dielectric extraction and charge-neutrality correction.
- [`src/pymultibinit/calculator.py`](src/pymultibinit/calculator.py) and [`src/pymultibinit/__init__.py`](src/pymultibinit/__init__.py): ASE contract and public exports.
- [`docs/ENERGY_FORCE_STRESS_DECOMPOSITION.md`](docs/ENERGY_FORCE_STRESS_DECOMPOSITION.md), [`docs/PURE_PYTHON_TRAINING.md`](docs/PURE_PYTHON_TRAINING.md), [`dipole_dipole.md`](dipole_dipole.md), and [`dipole_comparison_report.md`](dipole_comparison_report.md): existing model, fit, limitations, and validation caveats.
- [`pyproject.toml`](pyproject.toml): current pmb dependencies (ASE, NumPy, SciPy are already required); [`LICENSE`](LICENSE): GPL-3.

### External primary/maintainer documentation

- X. Gonze and C. Lee, *Dynamical matrices, Born effective charges, dielectric permittivity tensors, and interatomic force constants from density-functional perturbation theory*, Phys. Rev. B 55, 10355 (1997), [doi:10.1103/PhysRevB.55.10355](https://doi.org/10.1103/PhysRevB.55.10355).
- ABINIT, [ANADDB `dipdip` variable](https://docs.abinit.org/variables/anaddb/#dipdip): long-range dipole contribution, subtraction before Fourier transform and Ewald reconstruction.
- ASE, [calculator contract](https://docs.ase-lib.org/ase/calculators/calculators.html), [calculator mixing / `SumCalculator`](https://docs.ase-lib.org/ase/calculators/mixing.html), [Atoms and partial periodicity](https://docs.ase-lib.org/ase/atoms.html), and [units](https://docs.ase-lib.org/ase/units.html).
- DeePMD-kit, [DPLR theory/training docs](https://docs.deepmodeling.com/projects/deepmd/en/latest/model/dplr.html): explicit `E = E_DP + E_ele`, with the short-range model fit to `E* − E_ele`.
- LES maintainers, [theory and limitations](https://les.readthedocs.io/en/latest/theory.html); B. Cheng, *Latent Ewald summation for machine learning of long-range interactions*, npj Comput. Mater. 11, 80 (2025), [doi:10.1038/s41524-025-01577-7](https://doi.org/10.1038/s41524-025-01577-7). These document an alternative jointly learned electrostatic partition, not a drop-in fixed-Born-charge correction.
