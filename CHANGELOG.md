# Changelog

All notable changes to pymultibinit will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
