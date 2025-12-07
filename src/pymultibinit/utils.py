"""
Utility functions for pymultibinit.

Includes library finding logic that works across different platforms
with various shared library extensions.
"""
import os
import platform
from typing import Optional, List


def get_library_extensions() -> List[str]:
    """
    Get list of possible shared library extensions for current platform.
    
    Returns:
        List of extensions to try, in order of preference
    """
    system = platform.system()
    
    if system == "Darwin":  # macOS
        # Try versioned first, then unversioned
        return [".dylib", ".10.dylib", ".10.5.8.dylib", ".so"]
    elif system == "Windows":
        return [".dll", ".lib"]
    elif system == "Linux":
        # Try versioned first, then unversioned
        return [".so", ".so.10", ".so.10.5.8"]
    else:
        # Generic Unix
        return [".so", ".dylib"]


def find_library(lib_name: str = "libabinit", search_paths: Optional[List[str]] = None) -> str:
    """
    Find the MULTIBINIT shared library across different platforms.
    
    Searches for library with various extensions (.dylib, .so, .dll, etc.)
    in standard locations.
    
    Args:
        lib_name: Base library name (default: "libabinit")
        search_paths: Additional paths to search (optional)
        
    Returns:
        Full path to found library
        
    Raises:
        FileNotFoundError: If library not found
    """
    # Get possible extensions for this platform
    extensions = get_library_extensions()
    
    # Build default search paths
    if search_paths is None:
        search_paths = []
    
    # Add relative paths from this file (development environment)
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    
    default_paths = [
        os.path.join(base_dir, "abinit_mb_clib", "build", "src", "98_main"),
        os.path.join(base_dir, "abinit_mb_clib", "build_so", "src", "98_main"),
        os.path.join(base_dir, "build", "src", "98_main"),
        os.path.join(base_dir, "lib"),
    ]
    
    # Combine search paths
    all_paths = search_paths + default_paths
    
    # Try each path with each extension
    for path in all_paths:
        if not os.path.isdir(path):
            continue
            
        for ext in extensions:
            lib_path = os.path.join(path, lib_name + ext)
            if os.path.exists(lib_path):
                return lib_path
            
            # Also try without 'lib' prefix on Windows
            if platform.system() == "Windows":
                alt_name = lib_name.replace("lib", "", 1) if lib_name.startswith("lib") else lib_name
                lib_path = os.path.join(path, alt_name + ext)
                if os.path.exists(lib_path):
                    return lib_path
    
    # Try finding in system library paths (LD_LIBRARY_PATH, etc.)
    # This is platform-specific
    if platform.system() == "Darwin":
        # macOS: Try common homebrew/macports locations
        system_paths = [
            "/usr/local/lib",
            "/opt/homebrew/lib",
            "/opt/local/lib",
        ]
        for path in system_paths:
            for ext in extensions:
                lib_path = os.path.join(path, lib_name + ext)
                if os.path.exists(lib_path):
                    return lib_path
    
    elif platform.system() == "Linux":
        # Linux: Try standard library paths
        system_paths = [
            "/usr/lib",
            "/usr/local/lib",
            "/usr/lib64",
            "/usr/local/lib64",
        ]
        for path in system_paths:
            for ext in extensions:
                lib_path = os.path.join(path, lib_name + ext)
                if os.path.exists(lib_path):
                    return lib_path
    
    # Not found - provide helpful error message
    extensions_str = ", ".join(extensions)
    searched_str = "\n  ".join(all_paths)
    raise FileNotFoundError(
        f"Could not find {lib_name} with extensions: {extensions_str}\n"
        f"Searched paths:\n  {searched_str}\n"
        f"Please ensure the library is built in abinit_mb_clib/build/src/98_main/\n"
        f"or provide lib_path explicitly."
    )


def get_library_info(lib_path: str) -> dict:
    """
    Get information about a library file.
    
    Args:
        lib_path: Path to library file
        
    Returns:
        Dictionary with library metadata
    """
    import os.path
    
    info = {
        "path": lib_path,
        "exists": os.path.exists(lib_path),
        "size": os.path.getsize(lib_path) if os.path.exists(lib_path) else 0,
        "extension": os.path.splitext(lib_path)[1],
        "platform": platform.system(),
    }
    
    return info
