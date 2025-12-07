#!/usr/bin/env python3
"""
Test script for CFFI wrapper.
"""
import os
import sys

print("Starting CFFI wrapper test...")

try:
    from pymultibinit.wrapper_cffi import MultibinitWrapperCFFI
    print("✅ Import successful")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

try:
    wrapper = MultibinitWrapperCFFI()
    print(f"✅ Wrapper created")
except Exception as e:
    print(f"❌ Wrapper creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Get data files
root_dir = os.path.abspath(os.getcwd())
test_data = os.path.join(root_dir, "tests", "data")
if not os.path.exists(test_data):
    # Try relative to this file
    test_data = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../tests/data")

ddb_file = os.path.join(test_data, "tmulti_l_6_DDB")
xml_file = os.path.join(test_data, "tmulti_l_8_1.xml")

print(f"\nData files:")
print(f"  DDB: {os.path.exists(ddb_file)}")
print(f"  XML: {os.path.exists(xml_file)}")

print("\nAttempting initialization...")
try:
    wrapper.init_from_params(
        ddb_file=ddb_file,
        sys_file=xml_file,
        coeff_file="",
        ncell=(2, 2, 2),
        ngqpt=(4, 4, 4),
        dipdip=1
    )
    print(f"✅ Initialization successful!")
    print(f"   Handle value: {wrapper.handle[0]}")
    
    # Check handle is not NULL
    if wrapper.handle[0] == wrapper.ffi.NULL:
        print("❌ Handle is NULL!")
        sys.exit(1)
        
    # Cleanup
    wrapper.free()
    print(f"✅ Cleanup successful")
    print("\n🎉 All tests passed!")
    
except Exception as e:
    print(f"❌ Initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
