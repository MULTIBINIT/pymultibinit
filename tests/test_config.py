"""
Test configuration file loading.

This module tests the configuration file parser and initialization
from config files.

How to run:
    pytest tests/test_config.py -v
"""
import pytest
import tempfile
import os
import numpy as np
from pathlib import Path
from pymultibinit.config import MultibinitConfig


def test_simple_format_ddb_mode():
    """Test parsing simple format with DDB file."""
    config_content = """
# Comment line
ddb_file: test_DDB
sys_file: test.xml
ncell: 2 2 2
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.conf', delete=False) as f:
        f.write(config_content)
        config_file = f.name
    
    try:
        config = MultibinitConfig.from_file(config_file)
        
        # Check parsed values
        assert config.ddb_file.endswith('test_DDB')
        assert config.sys_file.endswith('test.xml')
        assert config.ncell == (2, 2, 2)
        assert not config.is_abi_mode()
    finally:
        os.unlink(config_file)


def test_simple_format_abi_mode():
    """Test parsing simple format with ABI file."""
    config_content = """
abi_file: test.abi
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.conf', delete=False) as f:
        f.write(config_content)
        config_file = f.name
    
    try:
        config = MultibinitConfig.from_file(config_file)
        
        assert config.abi_file.endswith('test.abi')
        assert config.is_abi_mode()
    finally:
        os.unlink(config_file)


def test_ini_format():
    """Test parsing INI-like format."""
    config_content = """
[files]
ddb_file = test_DDB
sys_file = test.xml
coeff_file = test_coeff.xml

[parameters]
ncell = 3 3 3
ngqpt = 4 4 4
dipdip = 0

[backend]
backend = cffi
use_atomic_units = true
auto_match_atoms = false
match_tolerance = 0.05
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.conf', delete=False) as f:
        f.write(config_content)
        config_file = f.name
    
    try:
        config = MultibinitConfig.from_file(config_file)
        
        assert config.ddb_file.endswith('test_DDB')
        assert config.sys_file.endswith('test.xml')
        assert config.coeff_file.endswith('test_coeff.xml')
        assert config.ncell == (3, 3, 3)
        assert config.ngqpt == (4, 4, 4)
        assert config.dipdip == 0
        assert config.backend == 'cffi'
        assert config.use_atomic_units is True
        assert config.auto_match_atoms is False
        assert config.match_tolerance == 0.05
    finally:
        os.unlink(config_file)


def test_mixed_separators():
    """Test parsing with both = and : separators."""
    config_content = """
ddb_file: test_DDB
sys_file = test.xml
ncell: 2 2 2
ngqpt = 4 4 4
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.conf', delete=False) as f:
        f.write(config_content)
        config_file = f.name
    
    try:
        config = MultibinitConfig.from_file(config_file)
        
        assert config.ddb_file.endswith('test_DDB')
        assert config.sys_file.endswith('test.xml')
        assert config.ncell == (2, 2, 2)
        assert config.ngqpt == (4, 4, 4)
    finally:
        os.unlink(config_file)


def test_comma_separated_triple():
    """Test parsing triple values with commas."""
    config_content = """
ddb_file: test_DDB
ncell: 2,2,2
ngqpt: 4, 4, 4
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.conf', delete=False) as f:
        f.write(config_content)
        config_file = f.name
    
    try:
        config = MultibinitConfig.from_file(config_file)
        
        assert config.ncell == (2, 2, 2)
        assert config.ngqpt == (4, 4, 4)
    finally:
        os.unlink(config_file)


def test_boolean_values():
    """Test parsing various boolean formats."""
    test_cases = [
        ('true', True),
        ('True', True),
        ('yes', True),
        ('1', True),
        ('on', True),
        ('false', False),
        ('False', False),
        ('no', False),
        ('0', False),
        ('off', False),
    ]
    
    for bool_str, expected in test_cases:
        config_content = f"""
ddb_file: test_DDB
auto_match_atoms: {bool_str}
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.conf', delete=False) as f:
            f.write(config_content)
            config_file = f.name
        
        try:
            config = MultibinitConfig.from_file(config_file)
            assert config.auto_match_atoms == expected, f"Failed for {bool_str}"
        finally:
            os.unlink(config_file)


def test_relative_path_resolution():
    """Test that relative paths are resolved relative to config file directory."""
    # Create temporary directory with config file
    with tempfile.TemporaryDirectory() as tmpdir:
        config_file = Path(tmpdir) / 'config.conf'
        
        config_content = """
ddb_file: subdir/test_DDB
sys_file: ./test.xml
"""
        config_file.write_text(config_content)
        
        config = MultibinitConfig.from_file(str(config_file))
        
        # Paths should be absolute and relative to config file directory
        assert os.path.isabs(config.ddb_file)
        assert os.path.isabs(config.sys_file)
        assert config.ddb_file == str(Path(tmpdir) / 'subdir' / 'test_DDB')
        assert config.sys_file == str(Path(tmpdir) / 'test.xml')


def test_absolute_path_unchanged():
    """Test that absolute paths are not modified."""
    abs_path = '/absolute/path/to/test_DDB'
    
    config_content = f"""
ddb_file: {abs_path}
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.conf', delete=False) as f:
        f.write(config_content)
        config_file = f.name
    
    try:
        config = MultibinitConfig.from_file(config_file)
        assert config.ddb_file == abs_path
    finally:
        os.unlink(config_file)


def test_missing_required_parameters():
    """Test that missing required parameters raise ValueError."""
    config_content = """
# No ddb_file or abi_file
ncell: 2 2 2
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.conf', delete=False) as f:
        f.write(config_content)
        config_file = f.name
    
    try:
        with pytest.raises(ValueError, match="must specify either"):
            MultibinitConfig.from_file(config_file)
    finally:
        os.unlink(config_file)


def test_invalid_backend():
    """Test that invalid backend raises ValueError."""
    config_content = """
ddb_file: test_DDB
backend: invalid_backend
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.conf', delete=False) as f:
        f.write(config_content)
        config_file = f.name
    
    try:
        with pytest.raises(ValueError, match="Invalid backend"):
            MultibinitConfig.from_file(config_file)
    finally:
        os.unlink(config_file)


def test_invalid_triple():
    """Test that invalid triple values raise ValueError."""
    config_content = """
ddb_file: test_DDB
ncell: 2 2
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.conf', delete=False) as f:
        f.write(config_content)
        config_file = f.name
    
    try:
        with pytest.raises(ValueError, match="Expected 3 integers"):
            MultibinitConfig.from_file(config_file)
    finally:
        os.unlink(config_file)


def test_nonexistent_file():
    """Test that nonexistent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        MultibinitConfig.from_file('/nonexistent/config.conf')


def test_to_dict():
    """Test conversion to dictionary."""
    config_content = """
ddb_file: test_DDB
sys_file: test.xml
ncell: 2 2 2
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.conf', delete=False) as f:
        f.write(config_content)
        config_file = f.name
    
    try:
        config = MultibinitConfig.from_file(config_file)
        config_dict = config.to_dict()
        
        assert 'ddb_file' in config_dict
        assert 'sys_file' in config_dict
        assert 'ncell' in config_dict
        assert config_dict['ncell'] == (2, 2, 2)
    finally:
        os.unlink(config_file)


def test_comments_and_blank_lines():
    """Test that comments and blank lines are properly ignored."""
    config_content = """
# This is a comment
ddb_file: test_DDB  # inline comment

# Another comment
sys_file: test.xml

ncell: 2 2 2  # with spaces
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.conf', delete=False) as f:
        f.write(config_content)
        config_file = f.name
    
    try:
        config = MultibinitConfig.from_file(config_file)
        
        assert config.ddb_file.endswith('test_DDB')
        assert config.sys_file.endswith('test.xml')
        assert config.ncell == (2, 2, 2)
    finally:
        os.unlink(config_file)


def test_identity_mapping_no_pbc_shift():
    """Test detection of identity mapping without PBC shifts."""
    from pymultibinit.atom_matching import is_identity_mapping_no_pbc_shift
    
    positions = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 2.0, 2.0],
        [0.0, 4.0, 0.0],
        [2.0, 6.0, 2.0]
    ])
    lattice = np.array([
        [4.0, 0.0, 0.0],
        [0.0, 8.0, 0.0],
        [0.0, 0.0, 4.0]
    ])
    
    # Identity mapping
    mapping = np.array([0, 1, 2, 3])
    
    # Same positions - should be identity
    assert is_identity_mapping_no_pbc_shift(positions, positions, mapping, lattice)
    
    # Slightly different positions (within tolerance) - should still be identity
    positions_close = positions + 1e-8
    assert is_identity_mapping_no_pbc_shift(positions_close, positions, mapping, lattice)
    
    # Different positions - should not be identity
    positions_different = positions + 0.1
    assert not is_identity_mapping_no_pbc_shift(positions_different, positions, mapping, lattice)


def test_identity_mapping_with_pbc_shift():
    """Test that PBC-shifted positions are not considered identity."""
    from pymultibinit.atom_matching import is_identity_mapping_no_pbc_shift
    
    positions_ref = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 2.0, 2.0]
    ])
    
    # Positions shifted by one lattice vector
    positions_shifted = np.array([
        [4.0, 0.0, 0.0],  # Shifted by lattice vector [4, 0, 0]
        [6.0, 2.0, 2.0]
    ])
    
    lattice = np.array([
        [4.0, 0.0, 0.0],
        [0.0, 8.0, 0.0],
        [0.0, 0.0, 4.0]
    ])
    
    mapping = np.array([0, 1])  # Identity mapping
    
    # Even though mapping is identity, positions are PBC-shifted
    # so we should NOT skip force remapping
    assert not is_identity_mapping_no_pbc_shift(positions_shifted, positions_ref, mapping, lattice)


def test_non_identity_mapping():
    """Test that reordered atoms are not considered identity."""
    from pymultibinit.atom_matching import is_identity_mapping_no_pbc_shift
    
    positions_ref = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 2.0, 2.0]
    ])
    
    # Same positions but will use non-identity mapping
    positions_input = positions_ref.copy()
    
    lattice = np.array([
        [4.0, 0.0, 0.0],
        [0.0, 8.0, 0.0],
        [0.0, 0.0, 4.0]
    ])
    
    # Non-identity mapping (reversed)
    mapping = np.array([1, 0])
    
    # Mapping is not identity, so should return False
    assert not is_identity_mapping_no_pbc_shift(positions_input, positions_ref, mapping, lattice)
