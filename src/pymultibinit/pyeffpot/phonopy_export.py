"""Export ABINIT DDB harmonic data to phonopy files."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Sequence

import numpy as np

from .ddb_parser_complete import read_ddb
from .datastructures import UnitcellData
from .phonon import AMU_EMASS, HA_CMM1
from .symmetry import rotate_dynamical_matrix_full


@dataclass(frozen=True)
class PhonopyDdbExportResult:
    """Paths and metadata produced by :func:`write_phonopy_from_ddb`."""

    output_dir: Path
    phonopy_params_yaml: Path
    supercell_matrix: tuple[int, int, int]
    natom: int
    nqpoint: int


def write_phonopy_from_ddb(
    ddb_file: str | Path,
    output_dir: str | Path,
    *,
    supercell_matrix: Sequence[int] | np.ndarray | None = None,
    symprec: float = 1e-5,
    overwrite: bool = True,
) -> PhonopyDdbExportResult:
    """Generate a phonopy parameters file from an ABINIT DDB file.

    The written force constants use phonopy's standard VASP-compatible unit
    convention. Therefore plain ``phonopy.load("phonopy_params.yaml")`` reports
    frequencies in THz without requiring a custom ``factor`` argument.

    Parameters
    ----------
    ddb_file
        Input ABINIT text DDB file.
    output_dir
        Directory where ``phonopy_params.yaml`` will be written.
    supercell_matrix
        Diagonal supercell/q-grid dimensions. Defaults to the DDB ``ngqpt``.
        Only diagonal supercells matching the DDB q-grid are supported.
    symprec
        Symmetry tolerance passed to phonopy.
    overwrite
        If false, fail when an output file already exists.
    """

    _, Phonopy, DynmatToForceConstants, _, units = _import_phonopy()

    ddb_path = Path(ddb_file)
    out_dir = Path(output_dir)
    u = read_ddb(str(ddb_path))
    _validate_unitcell_for_export(u)

    supercell = _normalise_supercell_matrix(supercell_matrix, u)
    out_dir.mkdir(parents=True, exist_ok=True)

    phonopy_params_yaml = out_dir / "phonopy_params.yaml"
    planned = [phonopy_params_yaml]
    _check_overwrite(planned, overwrite)

    qgrid, dynmats = _expanded_dynamical_matrices(u, supercell)
    unitcell = _unitcell_to_phonopy_atoms(u, units.Bohr)
    phonon = Phonopy(
        unitcell,
        supercell_matrix=np.diag(supercell),
        primitive_matrix=np.eye(3),
        symprec=symprec,
        is_symmetry=True,
    )

    dynmat_to_fc = DynmatToForceConstants(phonon.primitive, phonon.supercell, is_full_fc=True)
    phonopy_dynmats = _ordered_phonopy_dynamical_matrices(
        dynmat_to_fc.commensurate_points,
        qgrid,
        dynmats,
        u,
        units.THzToCm,
    )
    dynmat_to_fc.dynamical_matrices = phonopy_dynmats
    dynmat_to_fc.run()

    # DynmatToForceConstants returns force constants consistent with factor=1.
    # Store in phonopy's standard VASP-unit convention so default phonopy.load()
    # applies VaspToTHz and recovers the same THz frequencies.
    phonon.force_constants = np.array(dynmat_to_fc.force_constants, dtype="double", order="C") / (units.DefaultToTHz**2)

    phonon.save(phonopy_params_yaml, settings={"force_constants": True})

    return PhonopyDdbExportResult(
        output_dir=out_dir,
        phonopy_params_yaml=phonopy_params_yaml,
        supercell_matrix=supercell,
        natom=u.natom,
        nqpoint=u.nqpt,
    )


def _import_phonopy():
    try:
        phonopy = import_module("phonopy")
        dynmat_to_fc = import_module("phonopy.harmonic.dynmat_to_fc")
        atoms = import_module("phonopy.structure.atoms")
        physical_units = import_module("phonopy.physical_units")
    except ImportError as exc:
        raise ImportError("phonopy is required for DDB-to-phonopy export. Install pymultibinit with phonopy support.") from exc
    return (
        phonopy,
        getattr(phonopy, "Phonopy"),
        getattr(dynmat_to_fc, "DynmatToForceConstants"),
        getattr(atoms, "PhonopyAtoms"),
        physical_units.get_physical_units(),
    )


def _validate_unitcell_for_export(u: UnitcellData) -> None:
    if u.qpoints is None or u.dynmat is None:
        raise ValueError("DDB does not contain q-point dynamical matrices")
    if u.ngqpt is None:
        raise ValueError("Could not infer DDB q-point grid (ngqpt)")
    if u.symrel is None and len(u.qpoints) != int(np.prod(u.ngqpt)):
        raise ValueError("DDB has irreducible q-points but no symmetry operations for expansion")


def _normalise_supercell_matrix(supercell_matrix: Sequence[int] | np.ndarray | None, u: UnitcellData) -> tuple[int, int, int]:
    ngqpt = np.asarray(u.ngqpt, dtype=int).reshape(3)
    grid = (int(ngqpt[0]), int(ngqpt[1]), int(ngqpt[2]))
    if supercell_matrix is None:
        return grid
    arr = np.asarray(supercell_matrix, dtype=int)
    if arr.shape == (3, 3):
        if not np.all(arr == np.diag(np.diag(arr))):
            raise ValueError("Only diagonal supercell matrices are supported")
        arr = np.diag(arr)
    if arr.shape != (3,):
        raise ValueError("supercell_matrix must be length 3 or a 3x3 diagonal matrix")
    result = (int(arr[0]), int(arr[1]), int(arr[2]))
    if any(x <= 0 for x in result):
        raise ValueError("supercell_matrix entries must be positive integers")
    if result != grid:
        raise ValueError(f"supercell_matrix {result} must match DDB q-grid {grid}")
    return result


def _check_overwrite(paths: Sequence[Path], overwrite: bool) -> None:
    if overwrite:
        return
    for path in paths:
        if path.exists():
            raise FileExistsError(f"Output file already exists: {path}")


def _unitcell_to_phonopy_atoms(u: UnitcellData, bohr_to_angstrom: float):
    _, _, _, PhonopyAtoms, _ = _import_phonopy()
    numbers = [int(u.znucl[int(typ) - 1]) for typ in u.typat]
    masses = [float(u.amu[int(typ) - 1]) for typ in u.typat]
    return PhonopyAtoms(
        numbers=numbers,
        masses=masses,
        scaled_positions=np.asarray(u.xred, dtype=float),
        cell=np.asarray(u.rprimd, dtype=float) * bohr_to_angstrom,
    )


def _expanded_dynamical_matrices(u: UnitcellData, grid: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    qgrid = _full_qgrid(grid)
    qpoints = np.asarray(u.qpoints, dtype=float)
    dynmat = np.asarray(u.dynmat, dtype=float)
    if len(qpoints) == len(qgrid):
        ordered = _order_existing_full_grid(qpoints, dynmat, qgrid)
        return qgrid, ordered

    if u.symrel is None:
        raise ValueError("Cannot expand irreducible DDB q-points without symrel")

    symrel = np.asarray(u.symrel, dtype=int)
    tnons = np.zeros((len(symrel), 3), dtype=float) if u.tnons is None else np.asarray(u.tnons, dtype=float)
    indsym = _build_inverse_atom_mapping(np.asarray(u.xred, dtype=float), symrel, tnons)
    symrec = np.array([np.linalg.inv(operation).T.round().astype(int) for operation in symrel])
    dynmats = np.zeros((len(qgrid), u.natom, 3, u.natom, 3, 2), dtype=float)

    for iq, q_target in enumerate(qgrid):
        found = False
        for iq_ibz, q_ibz in enumerate(qpoints):
            for isym, rec in enumerate(symrec):
                q_rot = rec @ q_ibz
                for time_reversal, sign in ((False, 1), (True, -1)):
                    diff = q_target - sign * q_rot
                    diff -= np.round(diff)
                    if np.max(np.abs(diff)) < 1e-8:
                        dynmats[iq] = rotate_dynamical_matrix_full(
                            dynmat[iq_ibz],
                            q_ibz,
                            symrel[isym],
                            tnons[isym],
                            indsym[:, isym, :],
                            u.rprimd,
                            time_reversal=time_reversal,
                            q_target=q_target,
                        )
                        found = True
                        break
                if found:
                    break
            if found:
                break
        if not found:
            raise ValueError(f"Could not expand DDB dynamical matrix to q={q_target}")
    return qgrid, dynmats


def _full_qgrid(grid: tuple[int, int, int]) -> np.ndarray:
    return np.array(
        [[i / grid[0], j / grid[1], k / grid[2]] for i in range(grid[0]) for j in range(grid[1]) for k in range(grid[2])],
        dtype=float,
    )


def _order_existing_full_grid(qpoints: np.ndarray, dynmats: np.ndarray, qgrid: np.ndarray) -> np.ndarray:
    ordered = np.zeros_like(dynmats)
    for iq, q in enumerate(qgrid):
        diff = qpoints - q
        diff -= np.round(diff)
        matches = np.where(np.max(np.abs(diff), axis=1) < 1e-8)[0]
        if len(matches) != 1:
            raise ValueError(f"Could not uniquely match full-grid DDB q-point {q}")
        ordered[iq] = dynmats[int(matches[0])]
    return ordered


def _build_inverse_atom_mapping(xred: np.ndarray, rotations: np.ndarray, translations: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    indsym = np.zeros((4, len(rotations), len(xred)), dtype=int)
    for isym, (rotation, tau) in enumerate(zip(rotations, translations, strict=True)):
        inverse = np.linalg.inv(rotation).round().astype(int)
        for iatom, position in enumerate(xred):
            target = inverse @ (position - tau)
            mapped_atom, lattice_translation = _find_equivalent_atom(target, xred, tol)
            indsym[3, isym, iatom] = mapped_atom
            indsym[:3, isym, iatom] = lattice_translation
    return indsym


def _find_equivalent_atom(target: np.ndarray, xred: np.ndarray, tol: float) -> tuple[int, np.ndarray]:
    diff = target[np.newaxis, :] - xred
    wrapped = diff - np.round(diff)
    distances = np.linalg.norm(wrapped, axis=1)
    index = int(np.argmin(distances))
    if distances[index] > tol:
        raise ValueError(f"Could not find equivalent atom for position {target}. Minimum distance: {distances[index]}")
    return index, np.round(target - xred[index]).astype(int)


def _ordered_phonopy_dynamical_matrices(
    commensurate_points: np.ndarray,
    qgrid: np.ndarray,
    dynmats: np.ndarray,
    u: UnitcellData,
    thz_to_cm: float,
) -> np.ndarray:
    matrices = []
    for q in np.asarray(commensurate_points, dtype=float):
        folded = q - np.floor(q)
        diff = qgrid - folded
        diff -= np.round(diff)
        matches = np.where(np.max(np.abs(diff), axis=1) < 1e-8)[0]
        if len(matches) != 1:
            raise ValueError(f"Could not map phonopy commensurate q-point {q} to DDB q-grid")
        ddb_dynmat = dynmats[int(matches[0]), ..., 0] + 1j * dynmats[int(matches[0]), ..., 1]
        matrices.append(_ddb_dynmat_to_phonopy_dynmat(ddb_dynmat, q, u, thz_to_cm))
    return np.array(matrices)


def _ddb_dynmat_to_phonopy_dynmat(ddb_dynmat: np.ndarray, q: np.ndarray, u: UnitcellData, thz_to_cm: float) -> np.ndarray:
    natom = u.natom
    matrix = ddb_dynmat.reshape(3 * natom, 3 * natom)
    masses = np.array([u.amu[int(typ) - 1] for typ in u.typat], dtype=float) * AMU_EMASS
    mass_factors = 1.0 / np.sqrt(np.outer(np.repeat(masses, 3), np.repeat(masses, 3)))
    phonopy_matrix = matrix * mass_factors * (HA_CMM1 / thz_to_cm) ** 2

    blocks = phonopy_matrix.reshape(natom, 3, natom, 3)
    xred = np.asarray(u.xred, dtype=float)
    atom_diff = xred[:, np.newaxis, :] - xred[np.newaxis, :, :]
    phase = np.exp(-2j * np.pi * np.einsum("i,abi->ab", q, atom_diff))
    return (blocks * phase[:, np.newaxis, :, np.newaxis]).reshape(3 * natom, 3 * natom)
