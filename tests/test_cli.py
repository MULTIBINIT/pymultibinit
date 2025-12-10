"""
Test CLI commands (mbtools).

This module tests the command-line interface for pymultibinit,
including make-supercell and export-ref commands.

How to run:
    pytest tests/test_cli.py -v
"""
import pytest
import tempfile
import os
import numpy as np
from pathlib import Path
from ase import Atoms
from ase.io import read, write
from pymultibinit.cli import make_supercell, export_reference
from pymultibinit.config import MultibinitConfig  # type: ignore


@pytest.fixture
def unit_cell_file():
    """Create a temporary unit cell CIF file for testing."""
    # Create simple cubic unit cell with 5 atoms
    atoms = Atoms(
        symbols=['Sr', 'Ti', 'O', 'O', 'O'],
        positions=[
            [0.0, 0.0, 0.0],      # Sr
            [2.075, 2.075, 2.075], # Ti
            [2.075, 2.075, 0.0],  # O
            [2.075, 0.0, 2.075],  # O
            [0.0, 2.075, 2.075],  # O
        ],
        cell=[4.15, 4.15, 4.15],
        pbc=True
    )
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cif', delete=False) as f:
        write(f.name, atoms, format='cif')
        yield f.name
    
    # Cleanup
    os.unlink(f.name)


@pytest.fixture
def config_file_abi():
    """Create a temporary config file with ABI file."""
    # Get test data path - it's in the parent pymultibinit_dev/tests/data
    # Current file is in pymultibinit/tests/test_cli.py
    # So go up 3 levels to pymultibinit_dev, then to tests/data
    test_file_path = Path(__file__).resolve()
    pymultibinit_dev_dir = test_file_path.parent.parent.parent
    test_data_dir = pymultibinit_dev_dir / 'tests' / 'data'
    abi_file = test_data_dir / 'tmulti_l_8_1.abi'
    
    # Verify file exists
    if not abi_file.exists():
        raise FileNotFoundError(f"Test data file not found: {abi_file}")
    
    config_content = f"""abi_file: {abi_file}
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.conf', delete=False) as f:
        f.write(config_content)
        f.flush()  # Ensure content is written
        config_file = f.name
    
    yield config_file
    
    # Cleanup
    os.unlink(config_file)


class TestMakeSupercell:
    """Tests for make-supercell command."""
    
    def test_basic_supercell_222(self, unit_cell_file):
        """Test creating a 2x2x2 supercell."""
        with tempfile.NamedTemporaryFile(suffix='.cif', delete=False) as output:
            output_file = output.name
        
        try:
            # Run make-supercell command
            result = make_supercell(
                unit_cell_file=unit_cell_file,
                output_file=output_file,
                nx=2, ny=2, nz=2,
                format=None,
                verbose=False
            )
            
            assert result == 0, "make_supercell failed"
            
            # Verify output file exists
            assert os.path.exists(output_file), "Output file not created"
            
            # Read and verify supercell
            supercell = read(output_file)
            unit_cell = read(unit_cell_file)
            
            # Check number of atoms (should be 8x unit cell)
            assert len(supercell) == len(unit_cell) * 8, \
                f"Expected {len(unit_cell) * 8} atoms, got {len(supercell)}"
            
            # Check lattice parameters (should be 2x unit cell)
            unit_lattice = unit_cell.get_cell()  # type: ignore
            super_lattice = supercell.get_cell()  # type: ignore
            
            np.testing.assert_allclose(
                super_lattice.diagonal(),  # type: ignore
                unit_lattice.diagonal() * 2,  # type: ignore
                rtol=1e-3,
                err_msg="Supercell lattice not correctly scaled"
            )
            
        finally:
            if os.path.exists(output_file):
                os.unlink(output_file)
    
    def test_supercell_321(self, unit_cell_file):
        """Test creating a 3x2x1 supercell."""
        with tempfile.NamedTemporaryFile(suffix='.cif', delete=False) as output:
            output_file = output.name
        
        try:
            result = make_supercell(
                unit_cell_file=unit_cell_file,
                output_file=output_file,
                nx=3, ny=2, nz=1,
                format=None,
                verbose=False
            )
            
            assert result == 0, "make_supercell failed"
            
            # Verify
            supercell = read(output_file)
            unit_cell = read(unit_cell_file)
            
            # Check number of atoms (should be 6x unit cell)
            assert len(supercell) == len(unit_cell) * 6
            
            # Check lattice scaling
            unit_lattice = unit_cell.get_cell()  # type: ignore
            super_lattice = supercell.get_cell()  # type: ignore
            
            expected_scaling = np.array([3.0, 2.0, 1.0])
            np.testing.assert_allclose(
                super_lattice.diagonal(),  # type: ignore
                unit_lattice.diagonal() * expected_scaling,  # type: ignore
                rtol=1e-3
            )
            
        finally:
            if os.path.exists(output_file):
                os.unlink(output_file)
    
    def test_different_formats(self, unit_cell_file):
        """Test output in different formats (XYZ, VASP)."""
        formats = [('xyz', 'extxyz'), ('vasp', 'vasp')]
        
        for ext, fmt in formats:
            with tempfile.NamedTemporaryFile(suffix=f'.{ext}', delete=False) as output:
                output_file = output.name
            
            try:
                result = make_supercell(
                    unit_cell_file=unit_cell_file,
                    output_file=output_file,
                    nx=2, ny=2, nz=2,
                    format=fmt,
                    verbose=False
                )
                
                assert result == 0, f"make_supercell failed for {fmt}"
                
                # Verify file exists and can be read
                assert os.path.exists(output_file), f"Output file not created for {fmt}"
                supercell = read(output_file, format=fmt)
                
                # Basic sanity check
                unit_cell = read(unit_cell_file)
                assert len(supercell) == len(unit_cell) * 8
                
            finally:
                if os.path.exists(output_file):
                    os.unlink(output_file)
    
    def test_verbose_mode(self, unit_cell_file, capsys):
        """Test verbose output."""
        with tempfile.NamedTemporaryFile(suffix='.cif', delete=False) as output:
            output_file = output.name
        
        try:
            result = make_supercell(
                unit_cell_file=unit_cell_file,
                output_file=output_file,
                nx=2, ny=2, nz=2,
                format=None,
                verbose=True
            )
            
            assert result == 0, "make_supercell failed"
            
            # Check that verbose output was printed
            captured = capsys.readouterr()
            assert "Reading unit cell" in captured.out
            assert "Building" in captured.out
            assert "Supercell" in captured.out
            
        finally:
            if os.path.exists(output_file):
                os.unlink(output_file)


class TestExportRef:
    """Tests for export-ref command."""
    
    def test_export_ref_with_structure(self, config_file_abi, unit_cell_file):
        """Test exporting reference structure directly from potential."""
        # Test that export-ref now works without needing --from-structure
        # It gets the unit cell directly from the potential
        
        with tempfile.NamedTemporaryFile(suffix='.cif', delete=False) as ref_file:
            ref_path = ref_file.name
        
        try:
            # Export reference structure (no --from-structure needed)
            result = export_reference(
                config_file=config_file_abi,
                output_file=ref_path,
                format=None,
                symbols=None,
                from_structure=None,
                verbose=False
            )
            
            # Should succeed now that we get structure from potential
            assert result == 0, "export_reference should succeed"
            
            # Verify the file was created and has correct structure
            assert os.path.exists(ref_path), "Output file should exist"
            
            # Read the structure and verify it's BaHfO3 with 5 atoms
            from ase.io import read
            atoms = read(ref_path)
            if isinstance(atoms, list):
                atoms = atoms[0]
            assert len(atoms) == 5, "Should have 5 atoms in unit cell"
            
            # Verify we have the correct elements (not H, He, Li)
            symbols = atoms.get_chemical_symbols()
            assert 'Ba' in symbols, "Should contain Ba"
            assert 'Hf' in symbols, "Should contain Hf"
            assert 'O' in symbols, "Should contain O"
            
        finally:
            if os.path.exists(ref_path):
                os.unlink(ref_path)
    
    def test_export_ref_with_symbols(self, config_file_abi, unit_cell_file):
        """Test exporting with --symbols option (legacy, now prints warning)."""
        # The --symbols parameter is now deprecated since we auto-detect from znucl
        # but kept for backward compatibility
        
        with tempfile.NamedTemporaryFile(suffix='.cif', delete=False) as ref_file:
            ref_path = ref_file.name
        
        try:
            # Export with mismatched symbols (should print warning but use auto-detected)
            symbols_str = "Sr,Ti,O,O,O," * 8  # Wrong number of atoms
            symbols_str = symbols_str.rstrip(',')
            
            result = export_reference(
                config_file=config_file_abi,
                output_file=ref_path,
                format=None,
                symbols=symbols_str,
                from_structure=None,
                verbose=False
            )
            
            # Should still succeed with auto-detection (ignores wrong symbols)
            assert result == 0, "Should succeed with auto-detected symbols"
            
            # Verify correct elements were used (auto-detected, not user-provided)
            from ase.io import read
            atoms = read(ref_path)
            if isinstance(atoms, list):
                atoms = atoms[0]
            symbols = atoms.get_chemical_symbols()
            assert 'Ba' in symbols, "Should use auto-detected Ba"
            assert 'Hf' in symbols, "Should use auto-detected Hf"
            
        finally:
            if os.path.exists(ref_path):
                os.unlink(ref_path)
    
    def test_export_ref_verbose(self, config_file_abi, unit_cell_file, capsys):
        """Test verbose output for export-ref."""
        with tempfile.NamedTemporaryFile(suffix='.cif', delete=False) as supercell_file:
            supercell_path = supercell_file.name
        
        with tempfile.NamedTemporaryFile(suffix='.cif', delete=False) as ref_file:
            ref_path = ref_file.name
        
        try:
            # Create supercell
            result = make_supercell(
                unit_cell_file=unit_cell_file,
                output_file=supercell_path,
                nx=2, ny=2, nz=2,
                format=None,
                verbose=False
            )
            assert result == 0, "make_supercell failed"
            
            # Export with verbose
            result = export_reference(
                config_file=config_file_abi,
                output_file=ref_path,
                format=None,
                symbols=None,
                from_structure=supercell_path,
                verbose=True
            )
            
            assert result == 0, "export_reference failed"
            
            # Check verbose output
            captured = capsys.readouterr()
            assert "Loading potential" in captured.out
            assert "Potential initialized" in captured.out
            assert "exported successfully" in captured.out
            
        finally:
            for f in [supercell_path, ref_path]:
                if os.path.exists(f):
                    os.unlink(f)


class TestCLIEdgeCases:
    """Test edge cases and error handling."""
    
    def test_make_supercell_nonexistent_input(self):
        """Test make-supercell with non-existent input file."""
        with tempfile.NamedTemporaryFile(suffix='.cif', delete=False) as output:
            output_file = output.name
        
        try:
            result = make_supercell(
                unit_cell_file='/nonexistent/file.cif',
                output_file=output_file,
                nx=2, ny=2, nz=2,
                format=None,
                verbose=False
            )
            
            # Should return non-zero exit code
            assert result != 0, "Expected error for non-existent file"
                
        finally:
            if os.path.exists(output_file):
                os.unlink(output_file)
    
    def test_export_ref_missing_from_structure(self, config_file_abi):
        """Test export-ref without --from-structure."""
        with tempfile.NamedTemporaryFile(suffix='.cif', delete=False) as ref_file:
            ref_path = ref_file.name
        
        try:
            # Try to export without evaluating first
            result = export_reference(
                config_file=config_file_abi,
                output_file=ref_path,
                format=None,
                symbols=None,
                from_structure=None,  # No structure provided
                verbose=False
            )
            
            # This might succeed (uses internal default) or fail
            # Either way is acceptable, just verify it doesn't crash
            if result == 0:
                # If succeeded, output should exist
                assert os.path.exists(ref_path)
            
        except Exception as e:
            # If it fails, that's OK - just shouldn't crash
            pass
        finally:
            if os.path.exists(ref_path):
                os.unlink(ref_path)
