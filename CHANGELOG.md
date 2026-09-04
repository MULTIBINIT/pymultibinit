# Changelog

All notable changes to pymultibinit will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [Unreleased]

## [0.3.14] - 2026-09-04

### Added
- **IFC-aware model fitting** (`fitwithphonon` feature line): Cartesian
  interatomic force-constant matrices as a fourth linear least-squares
  channel beside energy, forces, and stress — fitting FC matrices, never
  phonon frequencies.
  - `pymultibinit.pyeffpot.ifc_targets`: canonical `IfcTargetSpec`
    (import from `FORCE_CONSTANTS` + provenance sidecar, or generate by
    symmetry-reduced finite differences with any ASE calculator),
    `load_ifc_target` / `generate_ifc_target` with fingerprinted caching,
    reciprocal/ASR validation, `IfcUnitCell` shared Phonopy skeleton,
    `fixed_ifc`/`with_fitted_values` (K_fixed, exact by coefficient
    linearity).
  - `fit_multibinit_model_python(..., ifc_targets=...)`: channel with
    mean-normalized Frobenius residual (per-target weight
    `w_k/((3N_k)^2 n_active)`, global `PythonFitConfig.ifc_factor`),
    design-backed accumulators exact at any basis size,
    `PythonFitResult.ifc_report` with per-target RMSE/max-abs in
    eV/angstrom^2; greedy/lasso/screened-greedy all IFC-aware.
  - Symbolic derivation artifact `docs/derivations/ifc_fitting_derivation.md`
    (+ sympy script) covering term energies, product-rule per-term IFC
    features, coefficient linearity, objective/normal equations, greedy
    closed forms, and unit conversion — all machine-verified.
- `docs/PURE_PYTHON_TRAINING.md`: IFC fitting channel section.

### Fixed
- **IFC accumulator exactness**: the chunked normal accumulation filled
  only within-chunk diagonal blocks; cross-chunk terms were dropped, so
  `goal_ifc` could evaluate negative and multi-chunk bases were not
  solved as true least squares. Accumulators are now per-target
  design-matrix backed (`submatrix`/`normal_matvec`/`normal_column`),
  exact at any basis size (dense normal cached only for small bases).

### Added
- **Energy/force/stress decomposition API**: Decompose the total energy, forces,
  and stress into per-term contributions (reference, local harmonic IFCs,
  dipole-dipole, anharmonic, elastic, strain-coupling).
  - New ASE-compatible method `MultibinitCalculator.get_contributions(atoms)`
    returning a `Contributions` dataclass with per-term `energy`/`forces`/
    `stress` dicts and `total_energy()`/`total_forces()`/`total_stress()`
    accessors that reproduce the ASE getters exactly.
  - New `MultibinitPotential.evaluate_contributions(positions, lattice)`
    returning the same decomposition as a `{term: (e, f, s)}` dict in
    eV / eV-Å / eV-Å³ units.
  - New `EffectivePotential.evaluate_contributions(xcart, rprimd)` at the
    pure-Python backend layer (atomic units).
  - The harmonic IFC term is split into local (`harmonic_local`) and
    dipole-dipole (`dipdip`) parts via independently ASR-corrected force
    constant matrices; the two parts sum exactly to the combined harmonic.
  - Exported `Contributions` from the top-level `pymultibinit` package.
  - Only the pure-Python (`pyeffpot`) backend supports decomposition; the
    CFFI (Fortran) and spawned-process backends raise `NotImplementedError`.
- **Analytic phonopy construction**: Added
  `calculate_analytic_phonon(atoms, calculator, ...)`, which evaluates the
  pyeffpot Cartesian Hessian on phonopy's exact supercell atom order and
  installs it directly as full force constants without finite displacements.

### Changed
- **Dependency declaration**: `matplotlib` and `netCDF4` are now declared as
  required dependencies (they were imported but not listed). `jax` is now an
  optional extra (`pip install pymultibinit[jax]`) that accelerates the
  anharmonic backend; it falls back to an equivalent NumPy path when absent.

### Documentation
- New `docs/ENERGY_FORCE_STRESS_DECOMPOSITION.md` guide.
- New runnable example `examples/contributions_decomposition/`.
- README updated: backend/dependency notes, decomposition section, examples.

## [0.2.0] - 2025-12-07

### Added
- **Configuration file support**: New way to initialize potentials from config files
  - New `config` module with `MultibinitConfig` class for parsing configuration files
  - New class methods:
    - `MultibinitPotential.from_config_file()`: Initialize potential from config file
    - `MultibinitCalculator.from_config_file()`: Initialize ASE calculator from config file
  - Supports multiple formats:
    - Simple format: `key: value` pairs
    - INI format: Sections with `[section]` headers
    - Both `:` and `=` separators supported
  - Automatic path resolution: Relative paths resolved relative to config file directory
  - Comprehensive test suite with 14 tests covering all config scenarios
  - Example config files in `examples/` directory
  - Complete documentation in `docs/CONFIG_FILE_USAGE.md`

- **Supercell structure export**: Ability to extract and export the MULTIBINIT internal supercell structure
  - New methods on `MultibinitPotential`:
    - `get_supercell_structure()`: Get reference structure as numpy arrays
    - `export_supercell_to_ase()`: Export as ASE Atoms object
    - `export_supercell_to_file()`: Export to CIF, VASP, XYZ, etc.
  - Automatic unit conversion (Angstrom for ASE compatibility)
  - Test suite with 8 tests covering all export scenarios
  - Complete documentation in `docs/SUPERCELL_EXPORT.md`

- **Automatic atom matching with PBC handling**: `MultibinitPotential` now automatically matches atom ordering between input structures and MULTIBINIT's internal reference structure
  - New `atom_matching` module with core matching utilities
  - Enabled by default with `auto_match_atoms=True` parameter
  - Persistent mapping: computed once on first evaluation, reused for all subsequent calls
  - Automatic force remapping: forces always returned in input atom order
  - **Optimization**: Identity mapping detection to skip unnecessary force remapping when no reordering or PBC shifts needed
  - Full PBC support using minimum image convention
  - New utility functions:
    - `is_identity_mapping_no_pbc_shift()`: Check if mapping is trivial
  - New methods on `MultibinitPotential`:
    - `set_reference_structure()`: Explicitly set reference structure for matching
    - `compute_atom_mapping()`: Compute and store atom mapping
    - `get_atom_mapping()`: Retrieve stored mapping
    - `clear_atom_mapping()`: Reset stored mapping
  - Comprehensive test suite with 12 tests covering all matching scenarios (including PBC shift detection)
  - Complete documentation in `docs/ATOM_MATCHING_USAGE.md`

### Changed
- `MultibinitPotential.evaluate()` now automatically handles atom reordering when `auto_match_atoms=True` (default)
- Forces are now automatically mapped back to input atom order for user convenience
- **Performance improvement**: Force remapping is skipped when mapping is identity (no reordering) and no PBC shifts are needed

### Documentation
- Added `docs/SUPERCELL_EXPORT.md`: Complete guide for exporting supercell structures with 6 detailed examples
- Added `docs/CONFIG_FILE_USAGE.md`: Complete guide for configuration file usage
- Added `examples/multibinit_simple.conf`: Minimal simple format example
- Added `examples/multibinit_ini.conf`: Full INI format example
- Added `docs/ATOM_MATCHING_USAGE.md`: Complete usage guide with examples
- Added `docs/fortran/ATOM_ORDERING_AND_PBC.md`: Technical details on MULTIBINIT's internal atom handling
- Added `docs/fortran/DIAGRAMS.md`: Visual diagrams for supercell construction and displacement computation
- Added `docs/DEVELOPER_GUIDE.md`: Complete developer guide for extending pymultibinit
- Added `docs/api/API_REFERENCE.md`: Full API reference documentation

## [0.1.0] - Initial Release

### Added
- Initial Python wrapper for MULTIBINIT C API
- Basic `MultibinitPotential` class for energy and force evaluation
- ASE calculator interface
- Basic tests and examples
