# Pure-Python Model Training

This document explains the pure-Python training workflow in `pymultibinit.training`.
It focuses on how training data is prepared, how polynomial terms are generated or loaded, how feature matrices are built, and how terms are selected and solved.

The pure-Python path does not invoke the external `multibinit` executable. The binary runner remains available through `train_multibinit_model()` and `mbtools train`.

## Inputs And Outputs

Required inputs:

- DDB file: reference unit-cell data used to build the training supercell.
- ABINIT `HIST.nc`: training frames containing lattice, positions, energies, forces, and stresses.
- Candidate basis XML or generated displacement basis: polynomial terms whose coefficient values will be fitted.

Main outputs:

- Fitted coefficient XML, written with `write_fitted_xml()` and readable by `read_coefficient_xml()`.
- Diagnostics from `PythonFitResult` or `mbtools train-python --diagnostics-json`.

Minimal API example:

```python
from pymultibinit.training import PythonFitConfig, fit_multibinit_model_python

result = fit_multibinit_model_python(
    ddb="system.DDB",
    hist="training_HIST.nc",
    basis_xml="candidate_basis.xml",
    output_xml="fit_coeffs.xml",
    config=PythonFitConfig(
        ncell=(2, 2, 2),
        fit_on=(True, True, True),      # forces, stress, energy
        fit_factors=(1.0, 1.0, 1.0),
        regularization=1e-8,
        selection="greedy",
        ncoeff=20,
    ),
)
```

CLI example:

```bash
mbtools train-python system.DDB training_HIST.nc \
  --basis-xml candidate_basis.xml \
  --output-xml fit_coeffs.xml \
  --diagnostics-json fit_diagnostics.json \
  --ncell 2 2 2 \
  --selection greedy \
  --ncoeff 20 \
  --regularization 1e-8
```

## Training Procedure

The public API `fit_multibinit_model_python()` wires the following steps:

1. Validate paths and configuration.
2. Read the DDB reference structure and build the requested supercell.
3. Read `HIST.nc` frames in raw ABINIT units.
4. Map each frame onto the reference supercell and compute residual targets.
5. Load candidate basis functions from XML, or generate a displacement-only basis separately.
6. Evaluate each candidate coefficient into linear energy, force, and stress feature arrays.
7. Solve all coefficients, or run greedy coefficient selection and solve the selected subset.
8. Write fitted XML and diagnostics.

The implementation lives primarily in `src/pymultibinit/training.py`.

## Units And Shapes

`read_hist_frames()` keeps ABINIT HIST quantities in ABINIT units:

- `rprimd`: Bohr, shape `(3, 3)` per frame.
- `xred`: fractional coordinates, shape `(natom, 3)`.
- `xcart`: Bohr, shape `(natom, 3)`.
- `energy`: Hartree.
- `forces`: Hartree/Bohr, shape `(natom, 3)`.
- `stress`: Hartree/Bohr^3, Voigt order `(xx, yy, zz, yz, xz, xy)`.

`TrainingDataset` uses Python array order `(time, ...)`:

- `displacement`: `(ntime, natom, 3)`.
- `du_delta`: `(ntime, 6, natom, 3)`.
- `strain`: `(ntime, 6)`.
- `ucvol`: `(ntime,)`.
- `sqomega`: `(ntime,)`.
- `energy_diff`: `(ntime,)`.
- `force_diff`: `(ntime, natom, 3)`.
- `stress_diff`: `(ntime, 6)`.

Feature matrices are solver-compatible:

- `energy`: `(ntime, ncoeff)`.
- `forces`: `(ntime, natom, 3, ncoeff)`.
- `stress`: `(ntime, 6, ncoeff)`.

## Dataset Mapping And Residuals

`build_training_dataset()` compares every HIST frame against the reference supercell.

The displacement mapping removes homogeneous strain before computing internal displacements:

```text
reference positions in HIST cell = reference xred @ hist rprimd.T
displacement = hist xcart - reference positions in HIST cell
```

This avoids treating pure cell deformation as an atomic displacement.

Strain is the Green-Lagrange strain in Voigt order `(xx, yy, zz, yz, xz, xy)`. The cell volume is `abs(det(rprimd))`. The MULTIBINIT/Sheppard factor is:

```text
sqomega = ucvol**(4/3) / natom**(1/3)
```

`du_delta` is the displacement derivative with respect to strain. It follows the MULTIBINIT formula using `(I + strain)^-1 u` and half-symmetric Voigt components.

Residual targets subtract an optional fixed model:

```text
energy_diff = HIST energy - fixed_model energy
force_diff  = HIST forces - fixed_model forces
stress_diff = HIST stress - fixed_model stress
```

If no fixed model is supplied, the fixed model is zero.

When constructing a DDB-backed fixed model manually, match the MULTIBINIT input `dipdip` setting. For example, use `build_supercell(unitcell, ncell, dipdip=False)` when the fitting input has `dipdip 0`; otherwise Python will recompute/add dipole-dipole IFCs whenever Born charges are present.

## Loading An XML Basis

`load_xml_basis()` converts parsed MULTIBINIT XML coefficients into immutable `XmlBasisFunction` objects.

Each basis function preserves:

- coefficient number, initial value, and text label.
- term weight.
- displacement-difference factors: `atom_a`, `atom_b`, Cartesian direction, power, `cell_a`, and `cell_b`.
- strain factors: Voigt index and power.

`basis_to_coefficients()` and `write_fitted_xml()` convert the internal basis back into `PolynomialCoefficient` objects and write XML through the existing XML writer.

## Feature Evaluation Logic

`evaluate_basis_features()` evaluates each basis function as a linear column in the least-squares problem.

For a displacement factor, the convention matches the existing Python potential evaluator:

```text
diff = u_b(direction) - u_a(direction)
```

For a term with weight `w`, displacement factors `diff_i**p_i`, and strain factors `eta_j**q_j`, the energy feature is:

```text
E_feature = sum_over_supercell_origins w * product(diff_i**p_i) * product(eta_j**q_j)
```

Force features are the negative derivative of the term energy with respect to atomic displacement. For a displacement factor involving atoms `a` and `b`:

```text
dE/ddiff = w * p * diff**(p-1) * other_factors
F_b += -dE/ddiff
F_a +=  dE/ddiff
```

Stress features include:

- direct derivative with respect to strain factors.
- correction from `du_delta` and force features.
- final volume/strain scaling, matching the MULTIBINIT-style feature construction.

Supercell atom ordering follows `supercell_builder.py`: `ix` outer loop, then `iy`, then `iz`, then atom index inside the unit cell.

## Solving Coefficients

`solve_weighted_least_squares()` assembles MULTIBINIT-style weighted normal equations.

The `fit_on` and `fit_factors` tuple order is:

```text
(forces, stress, energy)
```

Objective factors are:

```text
force factor  = fit_factor_force  / (3 * natom * ntime)
stress factor = fit_factor_stress / (6 * ntime)
energy factor = fit_factor_energy / ntime
```

Stress rows are weighted by `sqomega`. Energy rows are weighted by `1/sqrt(sqomega)`. Optional frame weights multiply all three targets.

Regularization is ridge-style:

```text
A_regularized = A + lambda * I
```

Diagnostics include:

- goal function components: force+stress, force, stress, energy.
- residual norm.
- matrix rank.
- condition number.
- regularization value.
- solver info code.

## Term Selection Logic

If `PythonFitConfig(selection="all")`, every basis coefficient is solved at once.

If `PythonFitConfig(selection="greedy")`, `select_greedy_coefficients()` performs one-by-one selection:

1. Start with any preselected coefficients.
2. For each unselected and unbanned candidate, solve the trial selected subset.
3. Score the trial by configured residual norm.
4. Select the candidate with the lowest score.
5. Use lower coefficient index as deterministic tie-breaker.
6. Repeat until `ncoeff` coefficients are selected.

Constraints:

- `banned` candidates are never considered.
- `preselected` candidates are included before the first greedy step.
- overlapping `banned` and `preselected` sets are rejected.
- impossible requests fail fast.
- singular candidate additions are skipped and counted in step diagnostics.
- singular preselected sets raise a clear error.

The public `fit_multibinit_model_python()` dispatches to greedy selection whenever `config.selection == "greedy"`.

## Generating Displacement-Only Terms

`generate_displacement_basis()` creates candidate displacement-only `XmlBasisFunction` objects. Strain terms are intentionally deferred.

Inputs:

- `xcart`: reference Cartesian positions for one unit cell.
- `cutoff`: maximum pair distance.
- `power_range`: inclusive polynomial power range.
- `ncell`: supercell image range used to include periodic pair factors.
- optional `symrel`, `rprimd`, and `atom_mappings` for symmetry actions.

### Pair Factor Generation

Primitive factors are `PairKey` objects:

```python
PairKey(direction, atom_a, atom_b, cell_b)
```

They represent one displacement difference component:

```text
u(atom_a, direction) - u(atom_b + cell_b, direction)
```

The generator scans atom pairs and periodic images inside the requested `ncell` image range, keeps factors whose Cartesian distance is within `cutoff`, skips the zero self-pair, and canonicalizes inverse orientations.

Inverse pair orientations are equivalent up to sign:

```text
u_b - u_a = -(u_a - u_b)
```

`normalize_pair_key()` chooses the lexicographically smaller orientation and returns the sign change.

### Monomial Enumeration And Compatibility

Candidate monomials are multisets of pair factors with total power in `power_range`. They are represented by `MonomialKey`:

```python
MonomialKey(((PairKey(...), power), ...))
```

The first implementation uses `combinations_with_replacement()` and then prunes incompatible combinations. Compatibility is cell-aware: pair factors form a graph whose nodes are exact `(atom, cell)` references. A monomial is kept only if that graph is connected.

This avoids accepting unrelated pair products that are not connected through a shared atom-image node.

### Symmetry Action Maps

`build_factor_action_map()` precomputes how each `PairKey` transforms under each symmetry operation.

It handles:

- Cartesian signed-axis direction transforms.
- pair orientation sign changes.
- optional atom mappings.
- existing `build_atom_mapping()` arrays with shape `(4, nsym, natom)` by converting inverse atom mappings and rotated translations into direct mappings.

For non-orthogonal cells, pass `rprimd` so fractional `symrel` operations are converted to Cartesian direction actions using the same convention as the symmetry utilities.

### Orbit Canonicalization And Filtering

For each monomial, the generator builds its symmetry orbit and chooses one canonical representative. If a symmetry maps a monomial to itself with a negative sign, the monomial is anti-invariant and is filtered out.

Symmetry-related monomials are materialized as multiple XML terms within one coefficient. That preserves symmetry invariance while avoiding duplicate coefficients.

Output ordering is deterministic. The generated basis can be written with `write_fitted_xml()` and read back with `read_coefficient_xml()`.

## Current Limitations

- Generated basis currently covers displacement-only terms. Strain and strain-displacement generated terms are deferred.
- Compatibility pruning is a minimal connectedness filter, not a full future max-body graph generator.
- Symmetry actions currently support Cartesian signed-axis operations for displacement directions.
- Full repository tests still require optional external assets such as `libabinit.so` and sibling ABINIT reference files.
