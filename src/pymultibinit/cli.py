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
import json
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


def ddb_to_phonopy(ddb: str, output_dir: str,
                   supercell: tuple[int, int, int] | None = None,
                   overwrite: bool = True,
                   verbose: bool = False) -> int:
    try:
        from pymultibinit.pyeffpot import write_phonopy_from_ddb
    except ImportError as e:
        print(f"Error: Required package not installed: {e}", file=sys.stderr)
        return 1

    try:
        result = write_phonopy_from_ddb(
            ddb,
            output_dir,
            supercell_matrix=supercell,
            overwrite=overwrite,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if verbose:
            import traceback
            traceback.print_exc()
        return 1

    if verbose:
        print("DDB-to-phonopy export completed")
        print(f"  Output directory: {result.output_dir}")
        print(f"  Supercell/q-grid: {result.supercell_matrix}")
        print(f"  phonopy_params.yaml: {result.phonopy_params_yaml}")
    return 0


def train_model(ddb: str, hist: str, config: Optional[str] = None,
                output_dir: str = "multibinit_training",
                executable: Optional[str] = None,
                extra_args: Optional[list[str]] = None,
                ifc_config: Optional[str] = None,
                verbose: bool = False) -> int:
    """Build a MULTIBINIT model by calling the multibinit binary."""
    if ifc_config:
        # FR-008 gate: the binary has no IFC fitting surface; fail loudly
        # instead of silently ignoring the channel.
        print(
            "Error: IFC targets are not supported by the multibinit binary "
            f"({ifc_config}); use 'mbtools train-python --ifc-target ...' instead",
            file=sys.stderr,
        )
        return 1
    try:
        from pymultibinit.training import train_multibinit_model
    except ImportError as e:
        print(f"Error: Required package not installed: {e}", file=sys.stderr)
        return 1

    try:
        result = train_multibinit_model(
            ddb=ddb,
            hist=hist,
            config=config,
            output_dir=output_dir,
            executable=executable,
            extra_args=extra_args,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if verbose:
            import traceback
            traceback.print_exc()
        return 1

    if verbose:
        print("MULTIBINIT training completed")
        print(f"  Output directory: {result.output_dir}")
        print(f"  Metadata: {result.metadata_file}")
        print(f"  stdout log: {result.log_file}")
        print(f"  stderr log: {result.stderr_file}")
        if result.model_config:
            print(f"  Model config: {result.model_config}")
    return 0


def train_model_python(ddb: str, hist: str, basis_xml: str, output_xml: str,
                       diagnostics_json: str, ncell: tuple[int, int, int],
                       selection: str = "all", ncoeff: Optional[int] = None,
                       regularization: float = 0.0, verbose: bool = False,
                       min_pure_strain_ratio: float = 0.05,
                       ifc_factor: Optional[float] = None,
                       ifc_targets: Optional[list[str]] = None) -> int:
    """Fit a MULTIBINIT XML model using the pure-Python pipeline."""
    try:
        from pymultibinit.training import PythonFitConfig, fit_multibinit_model_python
    except ImportError as e:
        print(f"Error: Required package not installed: {e}", file=sys.stderr)
        return 1

    try:
        config_kwargs = {
            "ncell": ncell,
            "selection": selection,
            "ncoeff": ncoeff,
            "regularization": regularization,
            "min_pure_strain_ratio": min_pure_strain_ratio,
        }
        if ifc_factor is not None:
            config_kwargs["ifc_factor"] = ifc_factor
        targets = None
        if ifc_targets:
            from pymultibinit.pyeffpot.ifc_targets import (
                IfcTargetSpec,
                load_ifc_target,
            )
            targets = [
                load_ifc_target(IfcTargetSpec(id=Path(t).stem, mode="import", fc_file=t))
                for t in ifc_targets
            ]
        config = PythonFitConfig(**config_kwargs)
        result = fit_multibinit_model_python(
            ddb=ddb,
            hist=hist,
            basis_xml=basis_xml,
            output_xml=output_xml,
            config=config,
            ifc_targets=targets,
        )
        diagnostics_path = Path(diagnostics_json)
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_path.write_text(json.dumps(_python_fit_diagnostics(result), indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if verbose:
            import traceback
            traceback.print_exc()
        return 1

    if verbose:
        print("Pure-Python training completed")
        print(f"  Output XML: {result.output_xml}")
        print(f"  Diagnostics: {diagnostics_json}")
        print(f"  Coefficients: {result.ncoeff}")
        print(f"  Frames: {result.nframes}")
    return 0


def _python_fit_diagnostics(result) -> dict:
    diag = result.diagnostics
    goal = diag.goal
    return {
        "coefficients": result.coefficients.tolist(),
        "output_xml": result.output_xml,
        "ncoeff": result.ncoeff,
        "nframes": result.nframes,
        "ddb": result.ddb,
        "hist": result.hist,
        "basis_xml": result.basis_xml,
        "goal": {
            "force_stress": _json_float(goal.force_stress),
            "force": _json_float(goal.force),
            "stress": _json_float(goal.stress),
            "energy": _json_float(goal.energy),
            "ifc": _json_float(getattr(goal, "ifc", 0.0)),
        },
        "residual_norm": _json_float(diag.residual_norm),
        "matrix_rank": diag.matrix_rank,
        "condition_number": _json_float(diag.condition_number),
        "regularization": _json_float(diag.regularization),
        "info": diag.info,
    }


def _json_float(value):
    import math

    value = float(value)
    return value if math.isfinite(value) else None


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

  # Build a model by delegating to the multibinit binary
  mbtools train system.ddb training_HIST.nc --config train.abi --output-dir model_out

  # Fit XML coefficients without invoking multibinit
  mbtools train-python system.ddb training_HIST.nc basis.xml --output-xml fitted.xml

  mbtools ddb-to-phonopy system_DDB phonopy_from_ddb --verbose

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

    ddb_phonopy_parser = subparsers.add_parser(
        'ddb-to-phonopy',
        help='Export ABINIT DDB harmonic data to phonopy files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Generate phonopy_params.yaml from a DDB file.',
        epilog="""
The default supercell is the DDB q-point grid. If --supercell is supplied, it
must match that grid.

Examples:
  mbtools ddb-to-phonopy system_DDB phonopy_from_ddb
  mbtools ddb-to-phonopy system_DDB phonopy_from_ddb --supercell 4 4 4
        """
    )
    ddb_phonopy_parser.add_argument('ddb', type=str, help='Input ABINIT DDB file')
    ddb_phonopy_parser.add_argument('output_dir', type=str, help='Output directory for phonopy_params.yaml')
    ddb_phonopy_parser.add_argument('--supercell', type=int, nargs=3, default=None, metavar=('NX', 'NY', 'NZ'), help='Diagonal supercell/q-grid dimensions')
    ddb_phonopy_parser.add_argument('--no-overwrite', action='store_true', help='Fail if output files already exist')
    ddb_phonopy_parser.add_argument('--verbose', '-v', action='store_true', help='Print detailed information')

    # train subcommand
    train_parser = subparsers.add_parser(
        'train',
        help='Build a MULTIBINIT model using the multibinit binary',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Invoke the multibinit executable for model building from DDB and HIST inputs.',
        epilog="""
The binary is resolved from --executable, MULTIBINIT_BINARY,
PYMULTIBINIT_MULTIBINIT_BINARY, or PATH. DDB/HIST paths are exposed to the
binary through PYMULTIBINIT_DDB and PYMULTIBINIT_HIST. If --config is supplied,
it is passed as the first positional argument to the binary.

Examples:
  mbtools train system.ddb training_HIST.nc --config train.abi --output-dir model_out
  mbtools train system.ddb training_HIST.nc --executable /path/to/multibinit --binary-arg=--dry-run
        """
    )
    train_parser.add_argument('ddb', type=str, help='Input DDB file')
    train_parser.add_argument('hist', type=str, help='Input HIST.nc file')
    train_parser.add_argument('--config', type=str, default=None,
                              help='MULTIBINIT training input/config file passed to the binary')
    train_parser.add_argument('--output-dir', type=str, default='multibinit_training',
                              help='Directory for logs, metadata, and generated model files')
    train_parser.add_argument('--executable', type=str, default=None,
                              help='Path to multibinit executable')
    train_parser.add_argument('--verbose', '-v', action='store_true',
                              help='Print detailed information')
    train_parser.add_argument('--binary-arg', dest='extra_args', action='append', default=[],
                              help='Additional argument passed to the multibinit binary. Repeat for multiple arguments.')
    train_parser.add_argument('--ifc-config', type=str, default=None,
                              help='Rejected: the multibinit binary has no IFC fitting surface (use train-python --ifc-target)')

    train_python_parser = subparsers.add_parser(
        'train-python',
        help='Fit a MULTIBINIT XML model using pure Python',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Fit XML coefficient values from DDB, HIST, and basis XML without invoking multibinit.',
    )
    train_python_parser.add_argument('ddb', type=str, help='Input DDB file')
    train_python_parser.add_argument('hist', type=str, help='Input HIST.nc file')
    train_python_parser.add_argument('basis_xml_pos', type=str, nargs='?', help='Input coefficient basis XML file')
    train_python_parser.add_argument('--basis-xml', dest='basis_xml_opt', type=str, default=None, help='Input coefficient basis XML file')
    train_python_parser.add_argument('--output-xml', type=str, default='fit_coeffs.xml', help='Output fitted coefficient XML file')
    train_python_parser.add_argument('--diagnostics-json', type=str, default='fit_diagnostics.json', help='Output diagnostics JSON file')
    train_python_parser.add_argument('--ncell', type=int, nargs=3, default=(1, 1, 1), metavar=('NX', 'NY', 'NZ'), help='Supercell dimensions used by HIST frames')
    train_python_parser.add_argument('--selection', choices=('all', 'greedy'), default='all', help='Coefficient selection mode')
    train_python_parser.add_argument('--ncoeff', type=int, default=None, help='Number of coefficients for greedy selection')
    train_python_parser.add_argument('--regularization', type=float, default=0.0, help='Ridge regularization strength')
    train_python_parser.add_argument(
        '--min-pure-strain-ratio',
        type=float,
        default=0.05,
        help='Minimum pure-strain fraction reserved by greedy selection',
    )
    train_python_parser.add_argument('--ifc-factor', type=float, default=None,
                                     help='Global weight of the IFC fitting channel (default 1.0 when targets are given)')
    train_python_parser.add_argument('--ifc-target', dest='ifc_targets', action='append', default=None,
                                     help='FORCE_CONSTANTS (or .hdf5) file of one canonical IFC target with its sidecar. Repeat for multiple targets.')
    train_python_parser.add_argument('--verbose', '-v', action='store_true', help='Print detailed information')

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

    elif args.command == 'ddb-to-phonopy':
        if not Path(args.ddb).exists():
            print(f"Error: DDB file not found: {args.ddb}", file=sys.stderr)
            return 1
        if args.supercell and any(value <= 0 for value in args.supercell):
            print("Error: --supercell values must be positive integers", file=sys.stderr)
            return 1
        export_supercell = None if args.supercell is None else (args.supercell[0], args.supercell[1], args.supercell[2])
        return ddb_to_phonopy(
            ddb=args.ddb,
            output_dir=args.output_dir,
            supercell=export_supercell,
            overwrite=not args.no_overwrite,
            verbose=args.verbose,
        )

    elif args.command == 'train':
        if not Path(args.ddb).exists():
            print(f"Error: DDB file not found: {args.ddb}", file=sys.stderr)
            return 1
        if not Path(args.hist).exists():
            print(f"Error: HIST file not found: {args.hist}", file=sys.stderr)
            return 1
        if args.config and not Path(args.config).exists():
            print(f"Error: Config file not found: {args.config}", file=sys.stderr)
            return 1
        return train_model(
            ddb=args.ddb,
            hist=args.hist,
            config=args.config,
            output_dir=args.output_dir,
            executable=args.executable,
            extra_args=args.extra_args,
            ifc_config=getattr(args, 'ifc_config', None),
            verbose=args.verbose,
        )

    elif args.command == 'train-python':
        basis_xml = args.basis_xml_opt or args.basis_xml_pos
        if basis_xml is None:
            print("Error: Basis XML file required as positional argument or --basis-xml", file=sys.stderr)
            return 1
        if not Path(args.ddb).exists():
            print(f"Error: DDB file not found: {args.ddb}", file=sys.stderr)
            return 1
        if not Path(args.hist).exists():
            print(f"Error: HIST file not found: {args.hist}", file=sys.stderr)
            return 1
        if not Path(basis_xml).exists():
            print(f"Error: Basis XML file not found: {basis_xml}", file=sys.stderr)
            return 1
        if any(value <= 0 for value in args.ncell):
            print("Error: --ncell values must be positive integers", file=sys.stderr)
            return 1
        return train_model_python(
            ddb=args.ddb,
            hist=args.hist,
            basis_xml=basis_xml,
            output_xml=args.output_xml,
            diagnostics_json=args.diagnostics_json,
            ncell=(args.ncell[0], args.ncell[1], args.ncell[2]),
            selection=args.selection,
            ncoeff=args.ncoeff,
            regularization=args.regularization,
            min_pure_strain_ratio=args.min_pure_strain_ratio,
            ifc_factor=args.ifc_factor,
            ifc_targets=args.ifc_targets,
            verbose=args.verbose,
        )

    else:
        print(f"Error: Unknown command: {args.command}", file=sys.stderr)
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
