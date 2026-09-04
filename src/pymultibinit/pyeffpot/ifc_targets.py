"""Canonical interatomic-force-constant (IFC) fitting targets.

Story 2 of ``specs/ifc-aware-model-fitting`` (memnotes vault); the fitting
equations live in ``docs/derivations/ifc_fitting_derivation.md``.

Canonical target form
---------------------
Dense ``(3N_sc, 3N_sc)`` force-constant matrix in eV/Angstrom^2, atom-major
flat order ``3*atom + direction``, sign convention
``K = -dF/du = d2E/du2`` (the phonopy ``FORCE_CONSTANTS`` convention).
No reshaping, permutation, or unit guessing ever happens silently.

Import (ADR-2)
--------------
``load_ifc_target`` reads a phonopy force-constants file (``FORCE_CONSTANTS``
text or ``force_constants.hdf5``) through phonopy's own ``file_IO`` parsers,
plus a mandatory sidecar JSON declaring supercell/primitive matrices, the
unit cell, atom-order convention, units, semantics, and ASR/dipdip policy.
Validation is strict (D-RQ2): shape, finiteness, reciprocity, and ASR row
sums; every failure names the target ID. The single documented unit
conversion is Ha/Bohr^2 -> eV/Angstrom^2 (CODATA-2014 module constants).

Generation (ADR-5)
------------------
``generate_ifc_target`` builds the Phonopy skeleton via
``pymultibinit.phonon.build_phonopy`` — the identical construction used by
``calculate_analytic_phonon``, hence the identical supercell atom order —
and assembles force constants by explicit finite displacements of a
reference force evaluator (``stencil`` ``'central'`` or ``'central5'``).
Drift correction (ASR projection) is off by default; symmetrization is
applied and recorded by default. The result is written as the canonical
artifact (FORCE_CONSTANTS + sidecar) and re-read through the import path.
Content-hash fingerprint caching: any change to structure content,
calculator config, displacement, stencil, or matrices misses the cache.

K_fixed (ADR-3)
---------------
``fixed_ifc`` returns ``analytic_blocks().ifc`` of the potential with all
fitted (anharmonic polynomial) coefficient values zeroed on a deep copy —
exact by coefficient linearity (derivation D3).
"""
from __future__ import annotations
import copy
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Tuple

import numpy as np

from ..phonon import VaspToTHz, build_phonopy
from ..potential import BOHR_TO_ANGSTROM, HARTREE_TO_EV
from .potential import EffectivePotential

#: Sidecar schema tag (bumped on incompatible changes).
SIDECAR_SCHEMA = "ifc-target-sidecar/1"

UNITS_EV_ANGSTROM = "eV/angstrom^2"
UNITS_HA_BOHR = "ha/bohr^2"
ATOM_ORDER_PHONOPY = "phonopy"
SEMANTICS_TOTAL = "total"
SEMANTICS_SHORT_RANGE = "short_range"

_SIDECAR_REQUIRED = (
    "supercell_matrix",
    "primitive_matrix",
    "atom_order",
    "units",
    "semantics",
    "asr_applied",
    "dipdip_removed",
)

#: Ha/Bohr^2 -> eV/Angstrom^2 (CODATA-2014 module constants; derivation D5).
HA_BOHR2_TO_EV_ANGSTROM2 = HARTREE_TO_EV / BOHR_TO_ANGSTROM**2

#: Default validation tolerances (D-RQ2 strict validation, no projection).
RECIPROCITY_TOL = 1e-6   # relative: max|K-K^T| / max|K|
ASR_TOL = 1e-2           # absolute eV/Angstrom^2 row-sum tolerance

_STENCILS = {
    # offset multipliers (in units of `displacement`), derivative weights
    "central": ((+1.0, 0.5), (-1.0, -0.5)),
    "central5": ((+2.0, -1.0 / 12.0), (+1.0, 8.0 / 12.0),
                 (-1.0, -8.0 / 12.0), (-2.0, 1.0 / 12.0)),
}


class IfcTargetError(ValueError):
    """Named-target IFC import/generation/validation failure."""


def _canonical_json(obj: Any) -> str:
    """Deterministic JSON used for content hashing and sidecar writing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=float)


def _sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class IfcUnitCell:
    """Unit-cell content of an IFC target (Angstrom / fractional)."""

    cell: np.ndarray                 # (3, 3) rows, Angstrom
    symbols: Tuple[str, ...]
    scaled_positions: np.ndarray     # (natom, 3)

    def __post_init__(self) -> None:
        cell = np.asarray(self.cell, dtype=float)
        pos = np.asarray(self.scaled_positions, dtype=float)
        symbols = tuple(str(s) for s in self.symbols)
        if cell.shape != (3, 3) or not np.isfinite(cell).all():
            raise IfcTargetError("unit cell must be a finite (3, 3) array")
        if pos.shape != (len(symbols), 3) or not np.isfinite(pos).all():
            raise IfcTargetError(
                "scaled_positions must be finite with shape (natom, 3) "
                "matching symbols")
        object.__setattr__(self, "cell", cell)
        object.__setattr__(self, "symbols", symbols)
        object.__setattr__(self, "scaled_positions", pos)

    @property
    def natom(self) -> int:
        return len(self.symbols)

    def to_dict(self) -> dict:
        return {
            "cell": self.cell.tolist(),
            "symbols": list(self.symbols),
            "scaled_positions": self.scaled_positions.tolist(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "IfcUnitCell":
        try:
            return cls(cell=data["cell"], symbols=tuple(data["symbols"]),
                       scaled_positions=data["scaled_positions"])
        except KeyError as err:
            raise IfcTargetError(f"unitcell missing key {err}") from err

    @classmethod
    def from_atoms(cls, atoms) -> "IfcUnitCell":
        """From ASE atoms (cell rows in Angstrom, scaled positions)."""
        return cls(cell=atoms.get_cell().array,
                   symbols=tuple(atoms.get_chemical_symbols()),
                   scaled_positions=atoms.get_scaled_positions())

    def content_bytes(self) -> bytes:
        """Canonical structure-content bytes for fingerprinting (FR-011)."""
        return _canonical_json(self.to_dict()).encode("utf-8")

    def content_hash(self) -> str:
        return _sha256_hex(self.content_bytes())


@dataclass(frozen=True)
class IfcTargetSpec:
    """Declaration of one IFC fitting target (import or generate)."""

    id: str
    mode: str                                   # 'import' | 'generate'
    weight: float = 1.0
    # --- import mode ---
    fc_file: Optional[str] = None               # FORCE_CONSTANTS | *.hdf5
    sidecar_file: Optional[str] = None          # JSON; default <fc>.sidecar.json
    # --- generate mode ---
    structure: Optional[IfcUnitCell] = None
    supercell_matrix: Any = None                # default 3x3 identity
    primitive_matrix: Any = None                # default 3x3 identity
    symprec: float = 1e-5
    displacement: float = 0.01                  # Angstrom
    stencil: str = "central"                    # 'central' | 'central5'
    symmetrize: bool = True
    drift_correction: bool = False              # off by default (D-RQ5)
    calculator_config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode not in ("import", "generate"):
            raise IfcTargetError(
                f"target '{self.id}': mode must be 'import' or 'generate', "
                f"got {self.mode!r}")
        if not np.isfinite(self.weight) or self.weight < 0.0:
            raise IfcTargetError(
                f"target '{self.id}': weight must be finite and non-negative")
        if self.stencil not in _STENCILS:
            raise IfcTargetError(
                f"target '{self.id}': stencil must be one of "
                f"{sorted(_STENCILS)}, got {self.stencil!r}")
        if self.mode == "import" and not self.fc_file:
            raise IfcTargetError(f"target '{self.id}': fc_file is required")
        if self.mode == "generate":
            if self.structure is None:
                raise IfcTargetError(
                    f"target '{self.id}': structure is required for generate")
            if not np.isfinite(self.displacement) or self.displacement <= 0.0:
                raise IfcTargetError(
                    f"target '{self.id}': displacement must be positive")
            object.__setattr__(self, "supercell_matrix",
                               _int_matrix(self.supercell_matrix, "supercell_matrix")
                               if self.supercell_matrix is not None else np.eye(3, dtype=int))
            object.__setattr__(self, "primitive_matrix",
                               _int_matrix(self.primitive_matrix, "primitive_matrix")
                               if self.primitive_matrix is not None else np.eye(3, dtype=int))
        object.__setattr__(self, "weight", float(self.weight))


@dataclass(frozen=True)
class IfcTarget:
    """A validated canonical IFC target."""

    id: str
    weight: float
    ifc: np.ndarray                  # (3N, 3N) eV/Angstrom^2, atom-major flat
    supercell_matrix: np.ndarray     # (3, 3) int
    primitive_matrix: np.ndarray     # (3, 3)
    unitcell: IfcUnitCell
    content_hash: str                # fingerprint of the generating content
    metadata: Mapping[str, Any]

    @property
    def natsuper(self) -> int:
        return self.ifc.shape[0] // 3


# ----------------------------------------------------------------------
# Import path
# ----------------------------------------------------------------------

def load_ifc_target(spec: IfcTargetSpec, *,
                    reciprocity_tol: float = RECIPROCITY_TOL,
                    asr_tol: float = ASR_TOL) -> IfcTarget:
    """Import and strictly validate one IFC target (AC-2/AC-3).

    Uses phonopy ``file_IO`` parsers only. Raises :class:`IfcTargetError`
    naming ``spec.id`` on any missing sidecar field or invalid matrix.
    """
    if spec.mode != "import":
        raise IfcTargetError(f"target '{spec.id}': load_ifc_target needs mode 'import'")

    fc_path = Path(spec.fc_file)
    if not fc_path.exists():
        raise IfcTargetError(f"target '{spec.id}': force-constants file not found: {fc_path}")
    sidecar_path = Path(spec.sidecar_file) if spec.sidecar_file else \
        fc_path.parent / (fc_path.name + ".sidecar.json")
    if not sidecar_path.exists():
        raise IfcTargetError(
            f"target '{spec.id}': mandatory sidecar not found: {sidecar_path}")

    sidecar = _read_sidecar(spec.id, sidecar_path)
    unitcell = _unitcell_from_sidecar(spec.id, sidecar)
    supercell_matrix = _int_matrix(sidecar["supercell_matrix"], "supercell_matrix")
    primitive_matrix = np.asarray(sidecar["primitive_matrix"], dtype=float)

    n_uc = unitcell.natom
    det = float(np.linalg.det(supercell_matrix))
    n_sc = n_uc * int(round(abs(det)))
    if n_sc <= 0 or abs(abs(det) - round(abs(det))) > 1e-8:
        raise IfcTargetError(
            f"target '{spec.id}': supercell_matrix determinant {det} does not "
            "define an integer cell multiplication")

    fc = _parse_fc_file(spec.id, fc_path)
    if (fc.ndim != 4 or fc.shape[0] != fc.shape[1]
            or fc.shape[2:] != (3, 3)):
        raise IfcTargetError(
            f"target '{spec.id}': force constants must be a full square "
            f"(N, N, 3, 3) array, got shape {fc.shape} "
            "(compact/p2s form is not supported)")
    if fc.shape[0] != n_sc:
        raise IfcTargetError(
            f"target '{spec.id}': force constants have {fc.shape[0]} supercell "
            f"atoms but the sidecar defines {n_sc} "
            f"({n_uc} unit-cell atoms x |det(supercell_matrix)|={int(round(abs(det)))})")

    units = str(sidecar["units"]).strip().lower()
    converted = False
    if units == UNITS_HA_BOHR.lower():
        fc = fc * HA_BOHR2_TO_EV_ANGSTROM2
        converted = True
    elif units != UNITS_EV_ANGSTROM.lower():
        raise IfcTargetError(
            f"target '{spec.id}': units must be '{UNITS_EV_ANGSTROM}' or "
            f"'{UNITS_HA_BOHR}' (one documented conversion), got "
            f"{sidecar['units']!r}")

    # phonopy (i, j, 3, 3) layout -> atom-major flat (3i+a, 3j+b)
    ifc = fc.transpose(0, 2, 1, 3).reshape(3 * n_sc, 3 * n_sc)
    _validate_ifc(spec.id, ifc, sidecar, reciprocity_tol=reciprocity_tol,
                  asr_tol=asr_tol)

    metadata = {
        "source_mode": "import",
        "fc_file": str(fc_path),
        "sidecar_file": str(sidecar_path),
        "units": UNITS_EV_ANGSTROM,
        "units_converted_from_ha_bohr": converted,
        "atom_order": str(sidecar["atom_order"]),
        "semantics": str(sidecar["semantics"]),
        "asr_applied": bool(sidecar["asr_applied"]),
        "dipdip_removed": bool(sidecar["dipdip_removed"]),
        "sidecar_schema": str(sidecar.get("schema", "legacy")),
        "generator": sidecar.get("generator"),
    }
    return IfcTarget(
        id=spec.id, weight=spec.weight, ifc=ifc,
        supercell_matrix=supercell_matrix, primitive_matrix=primitive_matrix,
        unitcell=unitcell,
        content_hash=_import_content_hash(fc_path, sidecar_path),
        metadata=metadata,
    )


def _read_sidecar(target_id: str, path: Path) -> dict:
    try:
        sidecar = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as err:
        raise IfcTargetError(
            f"target '{target_id}': cannot read sidecar {path}: {err}") from err
    if not isinstance(sidecar, dict):
        raise IfcTargetError(
            f"target '{target_id}': sidecar {path} must be a JSON object")
    missing = [key for key in _SIDECAR_REQUIRED if key not in sidecar]
    if missing:
        raise IfcTargetError(
            f"target '{target_id}': sidecar {path} is missing mandatory "
            f"field(s): {', '.join(sorted(missing))}")
    order = str(sidecar["atom_order"]).strip().lower()
    if order != ATOM_ORDER_PHONOPY:
        raise IfcTargetError(
            f"target '{target_id}': unsupported atom_order "
            f"{sidecar['atom_order']!r}; only '{ATOM_ORDER_PHONOPY}' is supported")
    semantics = str(sidecar["semantics"]).strip().lower()
    if semantics not in (SEMANTICS_TOTAL, SEMANTICS_SHORT_RANGE):
        raise IfcTargetError(
            f"target '{target_id}': semantics must be "
            f"'{SEMANTICS_TOTAL}' or '{SEMANTICS_SHORT_RANGE}', got "
            f"{sidecar['semantics']!r}")
    sidecar["atom_order"] = order
    sidecar["semantics"] = semantics
    return sidecar


def _unitcell_from_sidecar(target_id: str, sidecar: Mapping[str, Any]) -> IfcUnitCell:
    unitcell = sidecar.get("unitcell")
    if isinstance(unitcell, Mapping):
        return IfcUnitCell.from_dict(unitcell)
    ref = sidecar.get("structure_ref")
    if isinstance(ref, Mapping) and "path" in ref:
        path = Path(ref["path"])
        if not path.exists():
            raise IfcTargetError(
                f"target '{target_id}': structure_ref file not found: {path}")
        digest = _sha256_hex(path.read_bytes())
        if "sha256" in ref and digest != ref["sha256"]:
            raise IfcTargetError(
                f"target '{target_id}': structure_ref content hash mismatch "
                f"(sidecar {ref['sha256']}, file {digest})")
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as err:
            raise IfcTargetError(
                f"target '{target_id}': structure_ref {path} must be a JSON "
                f"unit-cell descriptor: {err}") from err
        return IfcUnitCell.from_dict(data)
    raise IfcTargetError(
        f"target '{target_id}': sidecar needs an inline 'unitcell' object or "
        "a 'structure_ref' {'path', 'sha256'} entry")


def _parse_fc_file(target_id: str, path: Path) -> np.ndarray:
    from phonopy import file_IO
    try:
        if path.suffix == ".hdf5":
            out = file_IO.read_force_constants_hdf5(filename=str(path))
            if isinstance(out, tuple):  # return_physical_unit=True form
                out = out[0]
        else:
            out = file_IO.parse_FORCE_CONSTANTS(filename=str(path))
    except Exception as err:  # phonopy raises assorted errors; name the target
        raise IfcTargetError(
            f"target '{target_id}': phonopy could not parse {path}: {err}") from err
    return np.asarray(out, dtype=float)


def _validate_ifc(target_id: str, ifc: np.ndarray, sidecar: Mapping[str, Any],
                  *, reciprocity_tol: float, asr_tol: float) -> None:
    if not np.isfinite(ifc).all():
        raise IfcTargetError(f"target '{target_id}': force constants contain non-finite values")
    scale = float(np.abs(ifc).max()) if ifc.size else 0.0
    if reciprocity_tol > 0.0:
        asym = float(np.abs(ifc - ifc.T).max()) if ifc.size else 0.0
        if asym > reciprocity_tol * max(scale, 1e-30):
            raise IfcTargetError(
                f"target '{target_id}': force constants violate reciprocity: "
                f"max|K-K^T|={asym:.3e} > {reciprocity_tol:.0e} * max|K|={scale:.3e}")
    if str(sidecar.get("semantics")) == SEMANTICS_TOTAL and asr_tol > 0.0:
        row_sums = np.abs(ifc.sum(axis=1)).max() if ifc.size else 0.0
        if row_sums > asr_tol:
            raise IfcTargetError(
                f"target '{target_id}': force constants violate the acoustic "
                f"sum rule: max |sum_j K_ij| = {row_sums:.3e} eV/Angstrom^2 "
                f"> {asr_tol:.0e} (declare semantics 'short_range' for "
                "dipole-dipole-removed data)")


def _import_content_hash(fc_path: Path, sidecar_path: Path) -> str:
    h = hashlib.sha256()
    h.update(str(fc_path.name).encode("utf-8"))
    h.update(b"\0")
    h.update(fc_path.read_bytes())
    h.update(b"\0")
    h.update(sidecar_path.read_bytes())
    return h.hexdigest()


# ----------------------------------------------------------------------
# Generation path
# ----------------------------------------------------------------------

def _generation_fingerprint(spec: IfcTargetSpec) -> str:
    payload = {
        "schema": SIDECAR_SCHEMA,
        "structure": spec.structure.content_hash(),
        "supercell_matrix": np.asarray(spec.supercell_matrix).tolist(),
        "primitive_matrix": np.asarray(spec.primitive_matrix, dtype=float).tolist(),
        "symprec": float(spec.symprec),
        "displacement": float(spec.displacement),
        "stencil": spec.stencil,
        "symmetrize": bool(spec.symmetrize),
        "drift_correction": bool(spec.drift_correction),
        "calculator": dict(spec.calculator_config),
        "units": UNITS_EV_ANGSTROM,
        "atom_order": ATOM_ORDER_PHONOPY,
    }
    return _sha256_hex(_canonical_json(payload).encode("utf-8"))


def generate_ifc_target(spec: IfcTargetSpec,
                        get_forces: Callable[[Any], np.ndarray],
                        cache_dir: Optional[str] = None, *,
                        reciprocity_tol: float = RECIPROCITY_TOL,
                        asr_tol: float = ASR_TOL) -> IfcTarget:
    """Generate one reference IFC target by finite displacements (AC-4/AC-5).

    ``get_forces(atoms) -> (natom, 3)`` must return forces in eV/Angstrom
    for ASE atoms in the phonopy supercell order it is given. The Phonopy
    skeleton is built by ``pymultibinit.phonon.build_phonopy`` — the same
    construction as ``calculate_analytic_phonon`` — so the supercell atom
    order matches a phonopy run on the same structure and matrices exactly.

    Writes the canonical artifact (FORCE_CONSTANTS + sidecar) into
    ``cache_dir/<fingerprint>/`` when ``cache_dir`` is given (reusing a hit
    verbatim), always returns the target as loaded through the import path.
    """
    if spec.mode != "generate":
        raise IfcTargetError(
            f"target '{spec.id}': generate_ifc_target needs mode 'generate'")

    fingerprint = _generation_fingerprint(spec)
    write_dir = (Path(cache_dir) / fingerprint) if cache_dir else None
    if write_dir is not None:
        fc_path = write_dir / "FORCE_CONSTANTS"
        sidecar_path = write_dir / "sidecar.json"
        if fc_path.exists() and sidecar_path.exists():
            loaded = load_ifc_target(
                IfcTargetSpec(id=spec.id, mode="import", weight=spec.weight,
                              fc_file=str(fc_path), sidecar_file=str(sidecar_path)),
                reciprocity_tol=reciprocity_tol, asr_tol=asr_tol)
            metadata = dict(loaded.metadata)
            metadata["cache"] = "hit"
            metadata["fingerprint"] = fingerprint
            metadata["source_mode"] = "generate"
            return IfcTarget(id=loaded.id, weight=loaded.weight, ifc=loaded.ifc,
                             supercell_matrix=loaded.supercell_matrix,
                             primitive_matrix=loaded.primitive_matrix,
                             unitcell=loaded.unitcell,
                             content_hash=fingerprint, metadata=metadata)

    ifc = _finite_difference_ifc(spec, get_forces)

    if write_dir is not None:
        write_dir.mkdir(parents=True, exist_ok=True)
        fc_path = write_dir / "FORCE_CONSTANTS"
        sidecar_path = write_dir / "sidecar.json"
    else:
        import tempfile
        write_dir = Path(tempfile.mkdtemp(prefix="ifc-target-"))
        fc_path = write_dir / "FORCE_CONSTANTS"
        sidecar_path = write_dir / "sidecar.json"
    _write_artifact(spec, ifc, fc_path, sidecar_path, fingerprint)

    loaded = load_ifc_target(
        IfcTargetSpec(id=spec.id, mode="import", weight=spec.weight,
                      fc_file=str(fc_path), sidecar_file=str(sidecar_path)),
        reciprocity_tol=reciprocity_tol, asr_tol=asr_tol)
    metadata = dict(loaded.metadata)
    metadata["cache"] = "miss" if cache_dir else "write-only"
    metadata["fingerprint"] = fingerprint
    metadata["source_mode"] = "generate"
    return IfcTarget(id=loaded.id, weight=loaded.weight, ifc=loaded.ifc,
                     supercell_matrix=loaded.supercell_matrix,
                     primitive_matrix=loaded.primitive_matrix,
                     unitcell=loaded.unitcell,
                     content_hash=fingerprint, metadata=metadata)


def _finite_difference_ifc(spec: IfcTargetSpec,
                           get_forces: Callable[[Any], np.ndarray]) -> np.ndarray:
    """FD assembly in the phonopy supercell atom order (no projection)."""
    from ase import Atoms

    atoms = Atoms(symbols=list(spec.structure.symbols),
                  cell=np.asarray(spec.structure.cell, dtype=float),
                  scaled_positions=np.asarray(spec.structure.scaled_positions,
                                              dtype=float),
                  pbc=True)
    phonon, supercell_atoms = build_phonopy(
        atoms,
        supercell_matrix=spec.supercell_matrix,
        primitive_matrix=spec.primitive_matrix,
        factor=VaspToTHz,
        symprec=spec.symprec,
    )
    natom = len(supercell_atoms)
    reference_positions = supercell_atoms.get_positions()
    stencil = _STENCILS[spec.stencil]
    h = float(spec.displacement)

    ifc = np.zeros((3 * natom, 3 * natom), dtype=float)
    for i in range(natom):
        for mu in range(3):
            derivative = np.zeros(natom * 3, dtype=float)
            for mult, weight in stencil:
                displaced = supercell_atoms.copy()
                positions = reference_positions.copy()
                positions[i, mu] += mult * h
                displaced.set_positions(positions)
                forces = np.asarray(get_forces(displaced), dtype=float).reshape(-1)
                derivative += weight * forces
            # K = -dF/du; divide once by h (weights are 1/(coeff*h) factors)
            ifc[:, 3 * i + mu] = -derivative / h

    if spec.drift_correction:
        # explicit, recorded ASR row-sum projection (off by default)
        ifc -= np.repeat(ifc.sum(axis=1, keepdims=True) / (3 * natom), 3 * natom, axis=1)
    if spec.symmetrize:
        ifc = 0.5 * (ifc + ifc.T)
    return ifc


def _write_artifact(spec: IfcTargetSpec, ifc: np.ndarray, fc_path: Path,
                    sidecar_path: Path, fingerprint: str) -> None:
    from phonopy import file_IO

    n3 = ifc.shape[0]
    n = n3 // 3
    fc = ifc.reshape(n, 3, n, 3).transpose(0, 2, 1, 3)
    file_IO.write_FORCE_CONSTANTS(fc, filename=str(fc_path))

    sidecar = {
        "schema": SIDECAR_SCHEMA,
        "supercell_matrix": np.asarray(spec.supercell_matrix).tolist(),
        "primitive_matrix": np.asarray(spec.primitive_matrix, dtype=float).tolist(),
        "unitcell": spec.structure.to_dict(),
        "atom_order": ATOM_ORDER_PHONOPY,
        "units": UNITS_EV_ANGSTROM,
        "semantics": SEMANTICS_TOTAL,
        "asr_applied": bool(spec.drift_correction),
        "dipdip_removed": False,
        "generator": {
            "tool": "pymultibinit.pyeffpot.ifc_targets",
            "fingerprint": fingerprint,
            "stencil": spec.stencil,
            "displacement": float(spec.displacement),
            "symmetrized": bool(spec.symmetrize),
            "drift_correction": bool(spec.drift_correction),
            "symprec": float(spec.symprec),
            "calculator": dict(spec.calculator_config),
        },
    }
    sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n")


# ----------------------------------------------------------------------
# K_fixed (ADR-3)
# ----------------------------------------------------------------------

def with_fitted_values(potential, values) -> "EffectivePotential":
    """Deep copy of an EffectivePotential with fitted values replaced.

    ``values`` aligns with ``potential.supercell.anharmonic_coeffs``; all
    other channels (harmonic, elastic, internal strain, phonon-strain) are
    untouched. Recompiled so ``_anharmonic_compiled`` carries the new values.
    """
    eff = _as_pyeffpot(potential)
    zeroed = copy.deepcopy(eff)
    coeffs = list(getattr(zeroed.supercell, "anharmonic_coeffs", None) or [])
    values = list(values)
    if len(values) != len(coeffs):
        raise IfcTargetError(
            f"with_fitted_values: got {len(values)} values for "
            f"{len(coeffs)} fitted coefficients")
    for coeff, value in zip(coeffs, values):
        coeff.value = float(value)
    zeroed._anharmonic_compiled = None
    zeroed._compile_anharmonic_terms()
    zeroed._jax_compiled = None
    zeroed._use_jax = False
    return zeroed


def fixed_ifc(potential, xcart: np.ndarray, rprimd: np.ndarray) -> np.ndarray:
    """K_fixed: analytic IFC with fitted coefficient values zeroed (AC-6).

    ``potential`` is an ``EffectivePotential`` (pyeffpot backend); ``xcart``
    and ``rprimd`` follow the ``EffectivePotential.evaluate`` convention
    (Bohr). Returns the dense ``(3N, 3N)`` matrix in eV/Angstrom^2 with the
    same unit boundary as ``MultibinitPotential.analytic_blocks``.
    Exact by coefficient linearity (derivation D3).
    """
    from .second_derivatives import analytic_blocks

    eff = _as_pyeffpot(potential)
    coeffs = list(getattr(eff.supercell, "anharmonic_coeffs", None) or [])
    zeroed = with_fitted_values(eff, [0.0] * len(coeffs))

    xcart = np.asarray(xcart, dtype=float)
    rprimd = np.asarray(rprimd, dtype=float)
    u = zeroed._compute_displacements(xcart, rprimd)
    eta = zeroed._compute_strain(rprimd)
    blocks = analytic_blocks(zeroed, u, eta)
    return np.asarray(blocks.ifc) * HA_BOHR2_TO_EV_ANGSTROM2


def _as_pyeffpot(potential) -> "EffectivePotential":
    eff = getattr(potential, "_pyeffpot_potential", None)
    if eff is not None:
        return eff
    if isinstance(potential, EffectivePotential):
        return potential
    raise IfcTargetError(
        "potential must be an EffectivePotential or a MultibinitPotential "
        "with the pyeffpot backend")


def _int_matrix(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (3, 3):
        raise IfcTargetError(f"{name} must have shape (3, 3), got {array.shape}")
    rounded = np.rint(array).astype(int)
    if not np.allclose(array, rounded, atol=1e-8):
        raise IfcTargetError(f"{name} must be integer-valued")
    return rounded
