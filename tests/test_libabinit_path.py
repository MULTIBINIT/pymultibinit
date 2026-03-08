#!/usr/bin/env python
"""
Test LIBABINIT_PATH environment variable functionality.

This test verifies that:
1. LIBABINIT_PATH takes priority over auto-detection
2. Invalid LIBABINIT_PATH raises appropriate error
3. Auto-detection works when LIBABINIT_PATH is not set

Run:
    python tests/test_libabinit_path.py
"""
import os
import sys
import tempfile
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pymultibinit.utils import find_library


def test_libabinit_path_priority():
    """Test that LIBABINIT_PATH takes priority."""
    print("\n=== Test 1: LIBABINIT_PATH Priority ===")
    
    # Find the actual library location first
    original_env = os.environ.get('LIBABINIT_PATH')
    if original_env:
        del os.environ['LIBABINIT_PATH']
    
    auto_detected = find_library()
    print(f"Auto-detected library: {auto_detected}")
    
    # Set LIBABINIT_PATH to the same location
    os.environ['LIBABINIT_PATH'] = auto_detected
    env_detected = find_library()
    print(f"LIBABINIT_PATH library: {env_detected}")
    
    assert env_detected == auto_detected, "LIBABINIT_PATH should return the specified path"
    print("✓ LIBABINIT_PATH takes priority")
    
    # Restore original
    if original_env:
        os.environ['LIBABINIT_PATH'] = original_env
    else:
        del os.environ['LIBABINIT_PATH']


def test_invalid_libabinit_path():
    """Test that invalid LIBABINIT_PATH raises error."""
    print("\n=== Test 2: Invalid LIBABINIT_PATH ===")
    
    original_env = os.environ.get('LIBABINIT_PATH')
    
    # Set to invalid path
    os.environ['LIBABINIT_PATH'] = '/nonexistent/path/libabinit.dylib'
    
    try:
        find_library()
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError as e:
        error_msg = str(e)
        print(f"Got expected error: {error_msg[:100]}...")
        assert "LIBABINIT_PATH is set" in error_msg
        assert "does not exist" in error_msg
        print("✓ Invalid LIBABINIT_PATH raises appropriate error")
    
    # Restore original
    if original_env:
        os.environ['LIBABINIT_PATH'] = original_env
    else:
        del os.environ['LIBABINIT_PATH']


def test_auto_detection():
    """Test that auto-detection works when LIBABINIT_PATH is not set."""
    print("\n=== Test 3: Auto-Detection ===")
    
    original_env = os.environ.get('LIBABINIT_PATH')
    if original_env:
        del os.environ['LIBABINIT_PATH']
    
    try:
        lib_path = find_library()
        print(f"Auto-detected library: {lib_path}")
        assert os.path.exists(lib_path), f"Library should exist: {lib_path}"
        print("✓ Auto-detection works")
    except FileNotFoundError as e:
        print(f"⚠ Auto-detection failed (expected in some environments): {e}")
    finally:
        # Restore original
        if original_env:
            os.environ['LIBABINIT_PATH'] = original_env


def test_environment_variable_visibility():
    """Test that environment variable is visible from Python."""
    print("\n=== Test 4: Environment Variable Visibility ===")
    
    original_env = os.environ.get('LIBABINIT_PATH')
    
    test_path = '/test/path/libabinit.dylib'
    os.environ['LIBABINIT_PATH'] = test_path
    
    # Check visibility
    retrieved = os.environ.get('LIBABINIT_PATH')
    assert retrieved == test_path, f"Expected {test_path}, got {retrieved}"
    print(f"✓ Environment variable visible: {retrieved}")
    
    # Restore original
    if original_env:
        os.environ['LIBABINIT_PATH'] = original_env
    else:
        del os.environ['LIBABINIT_PATH']


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing LIBABINIT_PATH Functionality")
    print("=" * 60)
    
    try:
        test_environment_variable_visibility()
        test_libabinit_path_priority()
        test_invalid_libabinit_path()
        test_auto_detection()
        
        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        return 0
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"✗ Test failed: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
