# PyMultibinit Unit Tests

This directory contains **unit tests** for the `pymultibinit` Python package. These tests validate individual components and APIs without requiring ABINIT/MULTIBINIT libraries.

## Test Files

### ✅ `test_atom_matching.py` - Atom Reordering Logic
Tests the atom matching/reordering algorithm with PBC handling.

**What it tests:**
- Identity mapping detection
- Reordered atoms matching
- PBC-wrapped positions
- Round-trip mapping consistency
- Small displacement tolerance
- Validation and error handling

**Run:**
```bash
pytest pymultibinit/tests/test_atom_matching.py -v
```

### ✅ `test_config.py` - Configuration File Parsing
Tests the configuration file parser with various formats.

**What it tests:**
- Simple format (DDB mode vs ABI mode)
- INI format with sections
- Mixed separators (`:` and `=`)
- Comma-separated triples
- Boolean value parsing (true/false/yes/no/1/0)
- Path resolution (relative/absolute)
- Error handling (missing params, invalid backend)

**Run:**
```bash
pytest pymultibinit/tests/test_config.py -v
```

### ✅ `test_supercell_export.py` - Export Functionality
Tests exporting MULTIBINIT supercell structures to ASE/files.

**What it tests:**
- Export to ASE Atoms objects
- Unit conversion (Bohr → Angstrom)
- Export to various file formats (XYZ, CIF, POSCAR)
- Returns copies (not references)
- Error handling (no reference structure)
- Chemical symbol handling

**Run:**
```bash
pytest pymultibinit/tests/test_supercell_export.py -v
```

### ✅ `test_api.py` - High-Level API Tests
Tests the main user-facing Python APIs (requires compiled ABINIT library).

**What it tests:**
- Low-level wrapper with .abi file
- Low-level wrapper with direct parameters
- High-level Potential API with unit conversions
- ASE Calculator interface
- Potential initialization from config file
- Calculator initialization from config file

**Run:**
```bash
cd pymultibinit_dev
PYTHONPATH=pymultibinit/src:$PYTHONPATH python pymultibinit/tests/test_api.py
```

**Note:** This test requires ABINIT/MULTIBINIT to be compiled and must run as a standalone script (MPI initialization constraint). Running all tests together may cause MPI conflicts/segfaults - this is a known ABINIT limitation, not a test failure.

### ⚠️ `test_cffi.py` - CFFI vs ctypes Comparison
Compares CFFI wrapper against deprecated ctypes wrapper (development tool).

**Run:**
```bash
python pymultibinit/tests/test_cffi.py
```

**Note:** Only useful if maintaining both backends.

---

## Running Tests

### Run All Unit Tests (No ABINIT Required)
```bash
cd pymultibinit_dev
pytest pymultibinit/tests/test_atom_matching.py -v
pytest pymultibinit/tests/test_config.py -v
pytest pymultibinit/tests/test_supercell_export.py -v
```

### Run All Tests (Including API Tests)
```bash
cd pymultibinit_dev
pytest pymultibinit/tests/ -v  # Unit tests only
python pymultibinit/tests/test_api.py  # API tests (requires ABINIT)
```

### Run Specific Test
```bash
pytest pymultibinit/tests/test_atom_matching.py::test_find_atom_mapping_identity -v
```

---

## Test vs `/tests/` Directory

This directory (`pymultibinit/tests/`) contains **unit tests** for the Python package itself:
- Test individual components (atom matching, config parsing, export)
- Mock or minimal dependencies
- Fast execution

The `/tests/` directory contains **integration/end-to-end tests**:
- Test full workflows (physics validation, ABI parsing)
- Require compiled ABINIT/MULTIBINIT libraries
- Test real physics (zero forces, supercells)

---

## Configuration

Tests use `pytest.ini` at the project root for configuration:
```ini
[pytest]
testpaths = pymultibinit/tests tests
pythonpath = pymultibinit/src
```

This ensures pytest can import `pymultibinit` correctly.

---

## Test Statistics

- **Total tests**: ~36 tests across 4 files
- **Atom matching**: 9 tests
- **Config parsing**: 17 tests
- **Supercell export**: 8 tests
- **API tests**: 6 tests (manual run - includes config file initialization)

---

## Adding New Tests

1. Create test file: `test_<feature>.py`
2. Use pytest conventions:
   - Test functions: `def test_<something>():`
   - Test classes: `class Test<Something>:`
   - Use `pytest` for assertions and fixtures
3. Run locally: `pytest pymultibinit/tests/test_<feature>.py -v`
4. Ensure tests are independent (no shared state)

---

## Requirements

- **Python**: >=3.8
- **Required**: `pytest`, `numpy`, `cffi`
- **Optional**: `ase` (for export tests and ASE calculator)
- **For API tests**: Compiled ABINIT/MULTIBINIT library

Install test dependencies:
```bash
pip install -e ".[test]"
# or
pip install pytest numpy cffi ase
```
