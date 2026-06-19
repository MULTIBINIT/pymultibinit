"""
Test supercell structure export functionality.

This module tests the ability to extract and export the MULTIBINIT
internal supercell structure.

How to run:
    pytest tests/test_supercell_export.py -v
"""
import pytest
import numpy as np
import tempfile
import os


def test_get_supercell_structure_before_evaluate():
    """Test that get_supercell_structure returns None before any evaluation."""
    from pymultibinit import MultibinitPotential
    
    pot = MultibinitPotential()
    
    # Should return None since no reference set
    result = pot.get_supercell_structure()
    assert result is None


def test_get_supercell_structure_after_set_reference():
    """Test get_supercell_structure after explicitly setting reference."""
    from pymultibinit import MultibinitPotential
    
    pot = MultibinitPotential()
    
    # Set reference structure
    positions = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 2.0, 2.0]
    ])
    lattice = np.array([
        [4.0, 0.0, 0.0],
        [0.0, 4.0, 0.0],
        [0.0, 0.0, 4.0]
    ])
    
    pot.set_reference_structure(positions, lattice)
    
    # Should return the structure
    result = pot.get_supercell_structure()
    assert result is not None
    
    positions_out, lattice_out, atomic_numbers = result
    
    # Check positions and lattice match
    assert np.allclose(positions_out, positions)
    assert np.allclose(lattice_out, lattice)
    
    # Atomic numbers should be None because no znucl/typat metadata is loaded
    assert atomic_numbers is None


def test_get_supercell_structure_returns_copy():
    """Test that get_supercell_structure returns copies, not references."""
    from pymultibinit import MultibinitPotential
    
    pot = MultibinitPotential()
    
    positions = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 2.0, 2.0]
    ])
    lattice = np.array([
        [4.0, 0.0, 0.0],
        [0.0, 4.0, 0.0],
        [0.0, 0.0, 4.0]
    ])
    
    pot.set_reference_structure(positions, lattice)
    
    # Get structure
    positions_out1, lattice_out1, _ = pot.get_supercell_structure()
    positions_out2, lattice_out2, _ = pot.get_supercell_structure()
    
    # Modify returned arrays
    positions_out1[0, 0] = 999.0
    lattice_out1[0, 0] = 999.0
    
    # Should not affect subsequent calls (returns copies)
    assert positions_out2[0, 0] == 0.0
    assert lattice_out2[0, 0] == 4.0
    
    # Should not affect internal reference
    positions_out3, lattice_out3, _ = pot.get_supercell_structure()
    assert positions_out3[0, 0] == 0.0
    assert lattice_out3[0, 0] == 4.0


def test_export_supercell_to_ase():
    """Test exporting supercell to ASE Atoms object."""
    pytest.importorskip("ase", reason="ASE required for this test")
    
    from pymultibinit import MultibinitPotential
    
    pot = MultibinitPotential(use_atomic_units=False)
    
    positions = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 2.0, 2.0],
        [1.0, 1.0, 0.0]
    ])
    lattice = np.array([
        [4.0, 0.0, 0.0],
        [0.0, 4.0, 0.0],
        [0.0, 0.0, 4.0]
    ])
    
    pot.set_reference_structure(positions, lattice)
    
    # Export to ASE
    atoms = pot.export_supercell_to_ase()
    
    # Check basic properties
    assert len(atoms) == 3
    assert np.allclose(atoms.get_positions(), positions)
    assert np.allclose(atoms.get_cell(), lattice)
    assert atoms.pbc.all()  # Should have PBC enabled
    
    # No DDB metadata loaded → symbols default to 'X'
    assert all(s == 'X' for s in atoms.get_chemical_symbols())


def test_export_supercell_to_ase_atomic_units():
    """Test exporting supercell with atomic units conversion."""
    pytest.importorskip("ase", reason="ASE required for this test")
    
    from pymultibinit import MultibinitPotential
    from pymultibinit.potential import BOHR_TO_ANGSTROM
    
    pot = MultibinitPotential(use_atomic_units=True)
    
    # Positions in Bohr
    positions_bohr = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 2.0, 2.0]
    ])
    lattice_bohr = np.array([
        [4.0, 0.0, 0.0],
        [0.0, 4.0, 0.0],
        [0.0, 0.0, 4.0]
    ])
    
    pot.set_reference_structure(positions_bohr, lattice_bohr)
    
    # Export to ASE (should be in Angstrom)
    atoms = pot.export_supercell_to_ase()
    
    # Check conversion to Angstrom
    positions_ang_expected = positions_bohr * BOHR_TO_ANGSTROM
    lattice_ang_expected = lattice_bohr * BOHR_TO_ANGSTROM
    
    assert np.allclose(atoms.get_positions(), positions_ang_expected)
    assert np.allclose(atoms.get_cell(), lattice_ang_expected)


def test_export_supercell_to_file():
    """Test exporting supercell to various file formats."""
    pytest.importorskip("ase", reason="ASE required for this test")
    
    from pymultibinit import MultibinitPotential
    
    pot = MultibinitPotential()
    
    positions = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 2.0, 2.0]
    ])
    lattice = np.array([
        [4.0, 0.0, 0.0],
        [0.0, 4.0, 0.0],
        [0.0, 0.0, 4.0]
    ])
    
    pot.set_reference_structure(positions, lattice)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test different formats
        formats = [
            ('supercell.xyz', None),
            ('supercell.cif', 'cif'),
            ('POSCAR', 'vasp'),
        ]
        
        for filename, fmt in formats:
            filepath = os.path.join(tmpdir, filename)
            
            # Export
            pot.export_supercell_to_file(filepath, format=fmt)
            
            # Check file was created
            assert os.path.exists(filepath), f"Failed to create {filename}"
            assert os.path.getsize(filepath) > 0, f"{filename} is empty"
            
            # Try to read it back
            from ase.io import read
            atoms_read = read(filepath)
            assert len(atoms_read) == 2


def test_export_before_reference_raises():
    """Test that export raises error if no reference structure."""
    pytest.importorskip("ase", reason="ASE required for this test")
    
    from pymultibinit import MultibinitPotential
    
    pot = MultibinitPotential()
    
    # Should raise RuntimeError
    with pytest.raises(RuntimeError, match="Reference structure not available"):
        pot.export_supercell_to_ase()
    
    with pytest.raises(RuntimeError, match="Reference structure not available"):
        pot.export_supercell_to_file('test.cif')


def test_export_supercell_with_symbol_setting():
    """Test setting correct chemical symbols after export."""
    pytest.importorskip("ase", reason="ASE required for this test")
    
    from pymultibinit import MultibinitPotential
    
    pot = MultibinitPotential()
    
    # 5-atom structure (e.g., BaTiO3 primitive cell)
    positions = np.array([
        [0.0, 0.0, 0.0],  # Ba
        [2.0, 2.0, 2.0],  # Ti
        [2.0, 2.0, 0.0],  # O
        [2.0, 0.0, 2.0],  # O
        [0.0, 2.0, 2.0],  # O
    ])
    lattice = np.array([
        [4.0, 0.0, 0.0],
        [0.0, 4.0, 0.0],
        [0.0, 0.0, 4.0]
    ])
    
    pot.set_reference_structure(positions, lattice)
    
    # Export to ASE
    atoms = pot.export_supercell_to_ase()
    
    # No DDB metadata loaded → symbols default to 'X'
    assert atoms.get_chemical_symbols() == ['X'] * 5
    
    # Set correct symbols
    atoms.set_chemical_symbols(['Ba', 'Ti', 'O', 'O', 'O'])
    
    # Verify
    assert atoms.get_chemical_symbols() == ['Ba', 'Ti', 'O', 'O', 'O']
    
    # Can now write with correct symbols
    with tempfile.NamedTemporaryFile(suffix='.cif', delete=False) as f:
        atoms.write(f.name)
        temp_file = f.name
    
    try:
        # Read back and check symbols preserved
        from ase.io import read
        atoms_read = read(temp_file)
        assert atoms_read.get_chemical_symbols() == ['Ba', 'Ti', 'O', 'O', 'O']
    finally:
        os.unlink(temp_file)


def test_get_supercell_structure_with_atomic_numbers():
    """get_supercell_structure returns resolved atomic numbers when znucl/typat are set."""
    from pymultibinit import MultibinitPotential
    pot = MultibinitPotential()
    pot._znucl = np.array([56, 22, 8])
    pot._typat_super = np.array([1, 2, 3, 3, 3])
    pot.set_reference_structure(
        positions=np.zeros((5, 3)),
        lattice=np.eye(3) * 4.0,
    )
    result = pot.get_supercell_structure()
    assert result is not None
    _, _, atomic_numbers = result
    assert atomic_numbers is not None
    assert list(atomic_numbers) == [56, 22, 8, 8, 8]


def test_export_supercell_to_ase_resolves_symbols():
    """export_supercell_to_ase returns correct symbols when znucl/typat are set."""
    pytest.importorskip("ase", reason="ASE required for this test")
    from pymultibinit import MultibinitPotential
    pot = MultibinitPotential()
    pot._znucl = np.array([56, 22, 8])
    pot._typat_super = np.array([1, 2, 3, 3, 3])
    pot.set_reference_structure(
        positions=np.array([
            [0.0, 0.0, 0.0],
            [2.0, 2.0, 2.0],
            [2.0, 2.0, 0.0],
            [2.0, 0.0, 2.0],
            [0.0, 2.0, 2.0],
        ]),
        lattice=np.eye(3) * 4.0,
    )
    atoms = pot.export_supercell_to_ase()
    assert atoms.get_chemical_symbols() == ['Ba', 'Ti', 'O', 'O', 'O']
    assert atoms.get_atomic_numbers().tolist() == [56, 22, 8, 8, 8]


def test_export_reference_to_ase_species_mapping():
    """export_reference_to_ase resolves species via znucl."""
    pytest.importorskip("ase", reason="ASE required for this test")
    from pymultibinit import MultibinitPotential
    from unittest.mock import MagicMock

    pot = MultibinitPotential()
    pot._initialized = True
    pot._znucl = np.array([56, 22, 8])

    mock_wrapper = MagicMock()
    mock_wrapper.get_reference_structure.return_value = (
        5,
        np.array([1, 2, 3, 3, 3], dtype=np.int32),
        np.array([
            [0.0, 0.0, 0.0],
            [3.7, 3.7, 3.7],
            [3.7, 3.7, 0.0],
            [3.7, 0.0, 3.7],
            [0.0, 3.7, 3.7],
        ]),
        np.eye(3) * 7.4,
    )
    pot.wrapper = mock_wrapper
    pot._ensure_cffi_wrapper = lambda: mock_wrapper

    atoms = pot.export_reference_to_ase()
    assert atoms.get_chemical_symbols() == ['Ba', 'Ti', 'O', 'O', 'O']


def test_from_pyeffpot_export_supercell_resolves_bahfo3_symbols():
    """from_pyeffpot exports real BaHfO3 symbols from DDB metadata."""
    pytest.importorskip("ase", reason="ASE required for this test")
    from ase.data import chemical_symbols
    from pymultibinit import MultibinitPotential

    ddb_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "examples",
        "BaHfO3_example",
        "BaHfO3_DDB",
    )
    if not os.path.exists(ddb_path):
        pytest.skip(f"BaHfO3 DDB fixture not found: {ddb_path}")

    pot = MultibinitPotential.from_pyeffpot(ddb_path, ncell=(2, 2, 2))
    atoms = pot.export_supercell_to_ase()
    symbols = atoms.get_chemical_symbols()
    expected_symbols = {chemical_symbols[int(z)] for z in pot._znucl}
    assert symbols
    assert 'X' not in symbols
    assert set(symbols) == expected_symbols
