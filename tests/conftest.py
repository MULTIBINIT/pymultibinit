from pathlib import Path

import pytest


def _has_loadable_libabinit() -> bool:
    try:
        from cffi import FFI
        from pymultibinit.utils import find_library

        FFI().dlopen(find_library())
    except Exception:
        return False
    return True


def pytest_collection_modifyitems(config, items):
    root = Path(__file__).resolve().parents[2]
    bto_ddb = root / "abinit/tests/v9/Input/BTO.DDB"
    cffi_data = root / "tests/data/tmulti_l_8_1.abi"
    has_lib = _has_loadable_libabinit()

    skip_lib = pytest.mark.skip(reason="loadable libabinit is not available")
    skip_bto = pytest.mark.skip(reason=f"external ABINIT BTO.DDB fixture is not available: {bto_ddb}")
    skip_cffi_data = pytest.mark.skip(reason=f"external CFFI test data is not available: {cffi_data}")

    for item in items:
        path = Path(str(item.fspath)).name
        nodeid = item.nodeid

        if path in {"test_api.py", "test_api_old.py", "test_config_init.py"} and not has_lib:
            item.add_marker(skip_lib)

        if path == "test_libabinit_path.py" and item.name == "test_libabinit_path_priority" and not has_lib:
            item.add_marker(skip_lib)

        if path == "test_cli.py" and ("TestExportRef" in nodeid or "test_export_ref" in item.name):
            if not cffi_data.exists():
                item.add_marker(skip_cffi_data)
            elif not has_lib:
                item.add_marker(skip_lib)

        if not bto_ddb.exists():
            if path in {"test_pyeffpot_backend.py", "test_pyeffpot_vs_fortran.py"}:
                item.add_marker(skip_bto)
            elif path == "test_potential.py" and item.name == "test_from_files":
                item.add_marker(skip_bto)
            elif path == "test_phonon.py" and item.name in {"test_phonon_frequencies_bto", "test_ddb_parser_complete"}:
                item.add_marker(skip_bto)
            elif path == "test_supercell_builder.py" and item.name == "test_read_ddb_integrates_with_supercell_builder":
                item.add_marker(skip_bto)
