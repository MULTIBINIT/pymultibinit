#!/usr/bin/env python3
"""
mbtools - MultiBinit Tools CLI

Command-line tools for working with MULTIBINIT potentials and structures.

Available commands:
  - export-ref: Export the MULTIBINIT internal reference structure
  - make-supercell: Create supercell from unit cell structure file

Usage:
    mbtools export-ref config.conf output.cif
    mbtools make-supercell unit_cell.cif supercell.cif 2 2 2
"""
import sys
import argparse
from pathlib import Path
from typing import Optional


def export_reference(config_file: str, output_file: str, 
                    format: Optional[str] = None,
                    symbols: Optional[str] = None,
                    from_structure: Optional[str] = None,  # Keep for backward compat, but ignore
                    verbose: bool = False) -> int:
    """
    Export MULTIBINIT reference unit cell structure to file.
    
    This exports the unit cell (reference structure) directly from the
    potential without requiring evaluation.
    
    Parameters
    ----------
    config_file : str
        Path to configuration file
    output_file : str
        Output file path
    format : str, optional
        Output format (auto-detected from extension if not provided)
    symbols : str, optional
        Chemical symbols as comma-separated list (e.g., "Ba,Ti,O,O,O")
        If not provided, symbols are auto-detected from atomic numbers
    from_structure : str, optional
        Deprecated - kept for backward compatibility but ignored
    verbose : bool
        Print detailed information
        
    Returns
    -------
    exit_code : int
        0 on success, non-zero on failure
    """
    try:
        from pymultibinit import MultibinitPotential  # type: ignore
        from ase.io import read, write
        
    except ImportError as e:
        print(f"Error: Required package not installed: {e}", file=sys.stderr)
        print("Please install: pip install pymultibinit ase", file=sys.stderr)
        return 1
    
    try:
        # Load potential
        if verbose:
            print(f"Loading potential from: {config_file}")
        pot = MultibinitPotential.from_config_file(config_file)
        
        if verbose:
            print("✓ Potential initialized")
        
        # Export the reference unit cell structure (no evaluation needed!)
        if verbose:
            print("Extracting reference unit cell structure...")
        
        atoms_ref = pot.export_reference_to_ase()
        
        if verbose:
            print(f"✓ Extracted reference structure: {len(atoms_ref)} atoms")
            cell = atoms_ref.get_cell()
            print(f"  Lattice parameters:")
            for i in range(3):
                vec = cell[i]
                print(f"    a{i+1} = [{vec[0]:10.6f}, {vec[1]:10.6f}, {vec[2]:10.6f}] Å")
        
        # Set chemical symbols if provided
        if symbols:
            symbol_list = [s.strip() for s in symbols.split(',')]
            if len(symbol_list) != len(atoms_ref):
                print(f"Warning: {len(symbol_list)} symbols provided but structure has {len(atoms_ref)} atoms",
                      file=sys.stderr)
                print(f"Using default 'X' symbols", file=sys.stderr)
            else:
                atoms_ref.set_chemical_symbols(symbol_list)
                if verbose:
                    print(f"✓ Set chemical symbols")
        
        # Write to file
        if verbose:
            print(f"Writing to: {output_file}")
            if format:
                print(f"Format: {format}")
        
        write(output_file, atoms_ref, format=format)  # type: ignore
        
        if verbose:
            print("✓ Structure exported successfully")
        
        pot.free()
        return 0
        
    except FileNotFoundError as e:
        print(f"Error: File not found: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if verbose:
            import traceback
            traceback.print_exc()
        return 1


def make_supercell(unit_cell_file: str, output_file: str,
                  nx: int, ny: int, nz: int,
                  format: Optional[str] = None,
                  verbose: bool = False) -> int:
    """
    Create supercell from unit cell structure file.
    
    Simple utility to create supercells matching the ncell parameter.
    
    Parameters
    ----------
    unit_cell_file : str
        Unit cell structure file
    output_file : str
        Output structure file (supercell)
    nx, ny, nz : int
        Supercell dimensions (matching ncell)
    format : str, optional
        Output format (auto-detected from extension if not provided)
    verbose : bool
        Print detailed information
        
    Returns
    -------
    exit_code : int
        0 on success, non-zero on failure
    """
    try:
        from ase.io import read, write
        
    except ImportError as e:
        print(f"Error: Required package not installed: {e}", file=sys.stderr)
        print("Please install: pip install ase", file=sys.stderr)
        return 1
    
    try:
        # Read unit cell
        if verbose:
            print(f"Reading unit cell from: {unit_cell_file}")
        
        unit_cell = read(unit_cell_file)
        
        natom_unit = len(unit_cell)
        
        if verbose:
            print(f"✓ Unit cell: {natom_unit} atoms")
            symbols = unit_cell.get_chemical_symbols()  # type: ignore
            print(f"  Chemical formula: {''.join(set(symbols))}")
            cell = unit_cell.get_cell()  # type: ignore
            print(f"  Lattice parameters:")
            for i in range(3):
                vec = cell[i]  # type: ignore
                print(f"    a{i+1} = [{vec[0]:10.6f}, {vec[1]:10.6f}, {vec[2]:10.6f}] Å")
        
        # Create supercell
        if verbose:
            print(f"Building {nx}×{ny}×{nz} supercell...")
        
        supercell = unit_cell * (nx, ny, nz)  # type: ignore
        natom_super = len(supercell)
        
        if verbose:
            print(f"✓ Supercell: {natom_super} atoms ({natom_unit} × {nx}×{ny}×{nz})")
            cell = supercell.get_cell()  # type: ignore
            print(f"  Lattice parameters:")
            for i in range(3):
                vec = cell[i]  # type: ignore
                print(f"    a{i+1} = [{vec[0]:10.6f}, {vec[1]:10.6f}, {vec[2]:10.6f}] Å")
        
        # Write supercell
        if verbose:
            print(f"Writing to: {output_file}")
            if format:
                print(f"Format: {format}")
        
        write(output_file, supercell, format=format)  # type: ignore
        
        if verbose:
            print("✓ Supercell structure created successfully")
            print(f"\nThis structure matches ncell={nx} {ny} {nz} and can be used for:")
            print(f"  - Evaluating the potential")
            print(f"  - As --from-structure input for 'mbtools export-ref'")
        
        return 0
        
    except FileNotFoundError as e:
        print(f"Error: File not found: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if verbose:
            import traceback
            traceback.print_exc()
        return 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog='mbtools',
        description='MultiBinit Tools - CLI for MULTIBINIT potential manipulation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build supercell from unit cell (matches ncell in config)
  mbtools make-supercell unit_cell.cif supercell.cif 2 2 2 --verbose
  mbtools make-supercell POSCAR POSCAR_222 2 2 2 -f vasp
  
  # Export MULTIBINIT internal reference structure  
  mbtools export-ref config.conf structure.cif --verbose
  mbtools export-ref config.conf structure.xyz -f xyz -s "Ba,Ti,O,O,O"
  mbtools export-ref config.conf ref.cif --from-structure supercell.cif

For more information, see: https://github.com/abinit/pymultibinit
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # export-ref subcommand
    export_parser = subparsers.add_parser(
        'export-ref',
        help='Export MULTIBINIT internal reference structure',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Export the MULTIBINIT internal reference structure (supercell) to file',
        epilog="""
The exported structure is the MULTIBINIT internal supercell (ncell × unit cell),
NOT the DDB unit cell.

Examples:
  mbtools export-ref config.conf structure.cif
  mbtools export-ref config.conf POSCAR -f vasp -s "Ba,Ti,O,O,O"
  mbtools export-ref config.conf ref.cif --from-structure test.cif -v
        """
    )
    export_parser.add_argument('config_file', type=str,
                              help='Configuration file path')
    export_parser.add_argument('output_file', type=str,
                              help='Output structure file path')
    export_parser.add_argument('--format', '-f', type=str, default=None,
                              help='Output format (cif, xyz, vasp, json, extxyz). '
                                   'Auto-detected from extension if not specified.')
    export_parser.add_argument('--symbols', '-s', type=str, default=None,
                              help='Chemical symbols as comma-separated list (e.g., "Ba,Hf,O,O,O")')
    export_parser.add_argument('--from-structure', type=str, default=None,
                              help='Evaluate with this structure file first (must match ncell). '
                                   'Recommended when C API does not expose structure directly.')
    export_parser.add_argument('--verbose', '-v', action='store_true',
                              help='Print detailed information')
    
    # make-supercell subcommand
    supercell_parser = subparsers.add_parser(
        'make-supercell',
        help='Build supercell from unit cell',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Build supercell structure by repeating unit cell',
        epilog="""
Read unit cell structure and create supercell by repeating it nx × ny × nz times.
Use this to create structures matching the ncell parameter in your config.

The output structure will have natom_unitcell × nx × ny × nz atoms.

Examples:
  mbtools make-supercell unit_cell.cif supercell.cif 2 2 2
  mbtools make-supercell POSCAR POSCAR_222 2 2 2 -f vasp -v
  mbtools make-supercell unit.xyz super.xyz 3 3 1 --verbose
        """
    )
    supercell_parser.add_argument('unit_cell_file', type=str,
                                 help='Unit cell structure file')
    supercell_parser.add_argument('output_file', type=str,
                                 help='Output structure file (supercell)')
    supercell_parser.add_argument('nx', type=int,
                                 help='Repetitions in x direction (matches ncell[0])')
    supercell_parser.add_argument('ny', type=int,
                                 help='Repetitions in y direction (matches ncell[1])')
    supercell_parser.add_argument('nz', type=int,
                                 help='Repetitions in z direction (matches ncell[2])')
    supercell_parser.add_argument('--format', '-f', type=str, default=None,
                                 help='Output format (cif, xyz, vasp, json, extxyz). '
                                      'Auto-detected from extension if not specified.')
    supercell_parser.add_argument('--verbose', '-v', action='store_true',
                                 help='Print detailed information')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Check if command was provided
    if not args.command:
        parser.print_help()
        return 1
    
    # Execute command
    if args.command == 'export-ref':
        # Check if config file exists
        if not Path(args.config_file).exists():
            print(f"Error: Config file not found: {args.config_file}", file=sys.stderr)
            return 1
        
        # Check if from_structure exists
        if args.from_structure and not Path(args.from_structure).exists():
            print(f"Error: Structure file not found: {args.from_structure}", file=sys.stderr)
            return 1
        
        return export_reference(
            args.config_file,
            args.output_file,
            format=args.format,
            symbols=args.symbols,
            from_structure=args.from_structure,
            verbose=args.verbose
        )
    
    elif args.command == 'make-supercell':
        # Check if unit cell file exists
        if not Path(args.unit_cell_file).exists():
            print(f"Error: Unit cell file not found: {args.unit_cell_file}", file=sys.stderr)
            return 1
        
        # Validate dimensions
        if args.nx <= 0 or args.ny <= 0 or args.nz <= 0:
            print(f"Error: Supercell dimensions must be positive integers", file=sys.stderr)
            return 1
        
        return make_supercell(
            args.unit_cell_file,
            args.output_file,
            args.nx, args.ny, args.nz,
            format=args.format,
            verbose=args.verbose
        )
    
    else:
        print(f"Error: Unknown command: {args.command}", file=sys.stderr)
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
