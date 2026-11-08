"""Optional downstream consistency checks against the ``longrangefy`` package.

Dependency direction is one-way: pymultibinit may import longrangefy, never
the reverse.  Importing this module is safe when the optional package is
absent; only the comparison functions import it, and they raise an explanatory
``ImportError`` when it is missing.

This module is a *measurement harness*, not a parity certificate.  It aligns

* units (Hartree/Bohr <-> eV/Angstrom),
* atom and periodic-image order (image-major, atom-minor),
* the Fourier synthesis sign,
* Wigner-Seitz interpolation weights,
* the acoustic-sum-rule policy, and
* the conducting Gamma boundary,

then reports component-level IFC, energy and force differences with their
scales.  No acceptance tolerance is declared here; a caller deciding parity
must compare the reported numbers to its own predeclared bound.

The compared upstream route is the q-grid Fourier builder
``pyeffpot.supercell_builder.build_supercell(..., dipdip=True/False)``.  The
dense finite-difference IFC route of :mod:`pymultibinit.pyeffpot.ifc_targets`
is a *different* construction and is not mapped here; comparing it requires
separate ordering/extension glue and is deliberately not guessed.
"""

from __future__ import annotations

import copy
import importlib
import importlib.util
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from .potential import BOHR_TO_ANGSTROM, HARTREE_TO_EV
from .pyeffpot.datastructures import UnitcellData
from .pyeffpot.supercell_builder import (
    _bigbx9_rpoints,
    _canonical_coordinates,
    _compute_rpoint_weights,
    _expand_dynmat_to_full_bz,
    _generate_full_bz_qpoints,
    build_supercell,
)

#: Ha/Bohr**2 -> eV/Angstrom**2 (same CODATA constants as ifc_targets).
HA_BOHR2_TO_EV_ANGSTROM2 = HARTREE_TO_EV / BOHR_TO_ANGSTROM**2

#: Cosine tolerance used when recognizing that a q-point belongs to a grid.
_GRID_TOL = 1e-8
#: Default ceiling for the canonical-phase alignment; larger translations
#: change the target image assignment and are not silently ignored.
_PHASE_TOL_BOHR = 1e-8

_OPTIONAL_PACKAGE = "longrangefy"


def longrangefy_available() -> bool:
    """Whether the optional downstream package is importable.

    This never imports longrangefy itself.  A ``None`` entry in ``sys.modules``
    (used by tests and by deliberately disabled environments) is treated as
    absent even when a distribution remains on disk.
    """
    try:
        return importlib.util.find_spec(_OPTIONAL_PACKAGE) is not None
    except (ImportError, ValueError):
        return False


def _import_longrangefy():
    try:
        matched = importlib.import_module("longrangefy.matched")
        dipole = importlib.import_module("longrangefy.dipole")
        ifc = importlib.import_module("longrangefy.ifc")
    except ImportError as exc:  # pragma: no cover - exercised by the absence test
        raise ImportError(
            "pymultibinit.longrangefy_parity requires the optional "
            f"'{_OPTIONAL_PACKAGE}' package. Install it with "
            "'pip install pymultibinit[longrangefy]' (or install longrangefy "
            "directly). pymultibinit has no runtime dependency on it otherwise."
        ) from exc
    return matched, dipole, ifc


def _require_longrangefy():
    matched, dipole, ifc = _import_longrangefy()
    build = getattr(matched, "build_correction", None)
    if build is None:
        raise ImportError(
            "the installed longrangefy exposes no longrangefy.matched.build_correction; "
            "install a version providing the matched fixed-cell correction"
        )
    return matched, dipole, ifc, build


# ----------------------------------------------------------------------
# Data containers
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class MatrixDifference:
    """Quantitative difference between two matrices in eV/Angstrom**2."""

    label: str
    reference_scale: float
    max_abs: float
    rms: float

    @property
    def relative_max(self) -> float:
        if self.reference_scale == 0.0:
            return float("nan")
        return self.max_abs / self.reference_scale

    def as_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "reference_scale_eV_A2": self.reference_scale,
            "max_abs_eV_A2": self.max_abs,
            "rms_eV_A2": self.rms,
            "relative_max": self.relative_max,
        }


@dataclass(frozen=True)
class BranchMatrices:
    """Dense target-cell harmonic matrices of one construction route.

    All arrays are ``(3*N_L, 3*N_L)`` eV/Angstrom**2 in the documented
    image-major, atom-minor order.  ``split = short_range + ewald``; the
    upstream builder stores short-range blocks in ``atmfrc`` and the Ewald
    blocks separately, so the sum is required before comparing to the
    downstream split branch.
    """

    split: np.ndarray
    unsplit: np.ndarray
    correction: np.ndarray
    short_range: Optional[np.ndarray]
    ewald: Optional[np.ndarray]
    asr: bool

    @property
    def natom_target(self) -> int:
        return self.split.shape[0] // 3


@dataclass(frozen=True)
class DownstreamCorrection:
    """Aligned inputs and the standalone matched correction."""

    correction: Any
    unitcell_full: UnitcellData
    qbz: np.ndarray
    grid_h: np.ndarray
    phi0_grid: np.ndarray
    positions: np.ndarray
    cell: np.ndarray
    born: np.ndarray
    dielectric: np.ndarray
    matching_reps: Tuple[int, int, int]
    target_reps: Tuple[int, int, int]
    eta_angstrom2: float
    eta_bohr2: float
    nreal: int
    nrecip: int
    ft_sign: int
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class ParityReport:
    """Measurements from one upstream/downstream comparison."""

    alignment: Dict[str, Any]
    matrices: Dict[str, MatrixDifference]
    components: Dict[str, MatrixDifference]
    energy_force: Dict[str, Any]
    weights: Dict[str, Any]
    dipole_kernel: Dict[str, Any]
    offgrid: Dict[str, Any]
    asr: Dict[str, Any]
    convergence: Dict[str, Any]
    metadata: Dict[str, Any]
    upstream: BranchMatrices
    downstream: DownstreamCorrection

    def as_dict(self) -> Dict[str, Any]:
        """JSON-friendly metrics (large matrices are excluded)."""
        return _jsonable(
            {
                "alignment": self.alignment,
                "matrices": {key: value.as_dict() for key, value in self.matrices.items()},
                "components": {key: value.as_dict() for key, value in self.components.items()},
                "energy_force": self.energy_force,
                "weights": self.weights,
                "dipole_kernel": self.dipole_kernel,
                "offgrid": self.offgrid,
                "asr": self.asr,
                "convergence": self.convergence,
                "metadata": self.metadata,
            }
        )


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


# ----------------------------------------------------------------------
# Small numerical helpers
# ----------------------------------------------------------------------


def _as_reps(reps: Sequence[int], name: str) -> Tuple[int, int, int]:
    array = np.asarray(reps)
    if array.shape != (3,) or not np.all(np.isfinite(array.astype(float))):
        raise ValueError(f"{name} must be three positive integers, got {reps!r}")
    rounded = np.rint(array.astype(float)).astype(int)
    if np.any(rounded < 1) or not np.allclose(array.astype(float), rounded):
        raise ValueError(f"{name} must be three positive integers, got {reps!r}")
    return int(rounded[0]), int(rounded[1]), int(rounded[2])


def _tiled_grid(reps: Tuple[int, int, int]) -> np.ndarray:
    """Integer translations in tiled C order, matching the dense site order."""
    return np.stack(
        np.meshgrid(*[np.arange(k) for k in reps], indexing="ij"), axis=-1
    ).reshape(-1, 3).astype(int)


def _matrix_difference(
    label: str, reference: np.ndarray, candidate: np.ndarray
) -> MatrixDifference:
    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    if reference.shape != candidate.shape:
        raise ValueError(
            f"{label}: shapes differ ({reference.shape} vs {candidate.shape})"
        )
    delta = candidate - reference
    scale = float(np.abs(reference).max()) if reference.size else 0.0
    return MatrixDifference(
        label=label,
        reference_scale=scale,
        max_abs=float(np.abs(delta).max()) if delta.size else 0.0,
        rms=float(np.sqrt(np.mean(delta**2))) if delta.size else 0.0,
    )


def _dense_from_qmatrices(qmatrices: np.ndarray, reps: Tuple[int, int, int], ifc) -> np.ndarray:
    blocks = ifc.qmatrices_to_blocks(np.asarray(qmatrices, dtype=complex), reps)
    return np.asarray(ifc.blocks_to_dense(blocks, reps), dtype=float)


def _target_dipole_dense(
    dipole,
    ifc,
    cell: np.ndarray,
    positions: np.ndarray,
    born: np.ndarray,
    dielectric: np.ndarray,
    reps: Tuple[int, int, int],
    eta: float,
    nreal: int,
    nrecip: int,
) -> np.ndarray:
    """Analytic target-periodic dipole matrix from public longrangefy pieces."""
    qgrid = _tiled_grid(reps) / np.asarray(reps, dtype=float)[None, :]
    samples = np.stack(
        [
            dipole.dipole_ifc(
                cell, positions, born, dielectric, q,
                eta=eta, nreal=nreal, nrecip=nrecip,
            )
            for q in qgrid
        ]
    )
    return _dense_from_qmatrices(samples, reps, ifc)


def _target_matrix_at_q(
    dense: np.ndarray,
    q: np.ndarray,
    reps: Tuple[int, int, int],
    natom: int,
    phase_sign: int = 1,
) -> Tuple[np.ndarray, float]:
    """Synthesize the primitive-cell block at ``q`` from a dense target matrix.

    Reads the home-cell (0,0,0) row block as the translation-resolved kernel
    and evaluates ``sum_T block(T) exp(phase_sign * 2 pi i q.T)``.  The second
    return value is the translation-invariance residual of the dense matrix.
    """
    l1, l2, l3 = reps
    dense = np.asarray(dense, dtype=float)
    expected = (3 * l1 * l2 * l3 * natom, 3 * l1 * l2 * l3 * natom)
    if dense.shape != expected:
        raise ValueError(
            f"dense matrix shape {dense.shape} does not match reps {reps} "
            f"and natom {natom}"
        )
    view = dense.reshape(l1, l2, l3, natom, 3, l1, l2, l3, natom, 3)
    home = view[0, 0, 0]  # (natom, 3, l1, l2, l3, natom, 3)
    translation_residual = 0.0
    for ix in range(l1):
        for iy in range(l2):
            for iz in range(l3):
                other = view[ix, iy, iz]
                rolled = np.roll(
                    np.roll(np.roll(home, ix, axis=2), iy, axis=3),
                    iz, axis=4,
                )
                translation_residual = max(
                    translation_residual,
                    float(np.abs(other - rolled).max()),
                )
    translations = _tiled_grid(reps).astype(float)
    phase = np.exp(int(phase_sign) * 2j * np.pi * translations @ np.asarray(q, dtype=float))
    blocks = np.stack(
        [
            home[:, :, tx, ty, tz, :, :]
            for tx, ty, tz in translations.astype(int)
        ]
    )
    block = np.einsum("tijkl,t->ijkl", blocks, phase)
    return block, translation_residual


def _select_phase_sign(
    dense: np.ndarray,
    q: np.ndarray,
    reps: Tuple[int, int, int],
    natom: int,
    reference: np.ndarray,
) -> Tuple[int, Dict[str, float]]:
    reports: Dict[str, Any] = {}
    blocks = {}
    for sign in (1, -1):
        block, residual = _target_matrix_at_q(dense, q, reps, natom, sign)
        blocks[sign] = block
        reports[f"sign_{sign:+d}_max_abs"] = float(np.abs(block - reference).max())
        reports[f"sign_{sign:+d}_translation_residual"] = residual
    plus = reports["sign_+1_max_abs"]
    minus = reports["sign_-1_max_abs"]
    scale = float(np.abs(reference).max()) if reference.size else 0.0
    reports["degenerate"] = bool(abs(plus - minus) <= 1e-12 * max(scale, 1.0))
    sign = min((1, -1), key=lambda s: reports[f"sign_{s:+d}_max_abs"])
    return sign, reports


def measure_phi0_symmetry_violation(
    phi0_grid: np.ndarray,
    grid_h: np.ndarray,
    matching_reps: Sequence[int],
) -> Dict[str, float]:
    """Reciprocal-kernel and Hermiticity residuals of matching-grid samples.

    The definition is the one used by ``longrangefy.matched`` before its
    explicit projection, so a measured value can be declared as
    ``phi0_symmetry_tol`` with full provenance instead of hard-coding a
    material guess.
    """
    reps = _as_reps(matching_reps, "matching_reps")
    phi0 = np.asarray(phi0_grid, dtype=complex)
    m1, m2, m3 = reps
    nq = m1 * m2 * m3
    if phi0.shape[0] != nq:
        raise ValueError(f"phi0_grid must carry {nq} samples, got {phi0.shape[0]}")
    grid_h = np.asarray(grid_h, dtype=int)
    if grid_h.shape != (nq, 3):
        raise ValueError(f"grid_h must have shape ({nq}, 3), got {grid_h.shape}")
    lookup = {tuple(int(v) for v in h): k for k, h in enumerate(grid_h.tolist())}
    if len(lookup) != nq:
        raise ValueError("grid_h must cover each matching-grid residue class once")
    tiled = _tiled_grid(reps)
    try:
        order = [lookup[tuple(int(v) for v in h)] for h in tiled]
    except KeyError as exc:
        raise ValueError(f"grid_h is missing a matching-grid residue class: {exc}") from exc
    grid = phi0[order].reshape(m1, m2, m3, *phi0.shape[1:])
    negative = [np.mod(-np.arange(k), k) for k in reps]
    flipped = grid[np.ix_(negative[0], negative[1], negative[2])]
    real_residual = float(np.abs(grid - np.conj(flipped)).max())
    hermitian = grid.transpose(0, 1, 2, 5, 6, 3, 4).conj()
    herm_residual = float(np.abs(grid - hermitian).max())
    scale = float(np.abs(grid).max()) if grid.size else 0.0
    return {
        "real_kernel_residual": real_residual,
        "hermiticity_residual": herm_residual,
        "max_abs": max(real_residual, herm_residual),
        "scale": scale,
        "relative_max": (
            max(real_residual, herm_residual) / scale if scale > 0.0 else float("nan")
        ),
    }


# ----------------------------------------------------------------------
# Input normalization
# ----------------------------------------------------------------------


def _full_grid_data(
    unitcell: UnitcellData, matching_reps: Tuple[int, int, int]
) -> Tuple[UnitcellData, np.ndarray, np.ndarray, np.ndarray]:
    """Return a copy whose q-point data is exactly the full matching grid."""
    if unitcell.qpoints is None or unitcell.dynmat is None:
        raise ValueError(
            "the q-grid route requires unitcell.qpoints and unitcell.dynmat; "
            "dense IFC-only unitcells use a different construction and are "
            "not mapped by this harness"
        )
    if unitcell.ngqpt is not None:
        declared = _as_reps(unitcell.ngqpt, "unitcell.ngqpt")
        if declared != matching_reps:
            raise ValueError(
                f"unitcell.ngqpt {declared} does not match matching_reps "
                f"{matching_reps}; the q-point data and matching grid must "
                "describe the same full grid"
            )
    qbz = _generate_full_bz_qpoints(np.asarray(matching_reps), 1, None)
    qpoints = np.asarray(unitcell.qpoints, dtype=float)
    dynmat = np.asarray(unitcell.dynmat, dtype=float)
    if dynmat.shape[0] != len(qpoints):
        raise ValueError("unitcell.qpoints and unitcell.dynmat disagree in length")
    nq = len(qbz)

    def _reorder() -> Optional[np.ndarray]:
        if len(qpoints) != nq:
            return None
        used = np.zeros(nq, dtype=bool)
        order = np.empty(nq, dtype=int)
        for k, q in enumerate(qbz):
            diff = qpoints - q
            diff -= np.round(diff)
            hits = np.where(np.max(np.abs(diff), axis=1) < _GRID_TOL)[0]
            hits = [int(h) for h in hits if not used[h]]
            if len(hits) != 1:
                return None
            order[k] = hits[0]
            used[hits[0]] = True
        return dynmat[order]

    reordered = _reorder()
    if reordered is not None:
        # The caller already supplied the full matching grid; reordering is
        # exact and avoids a symmetry expansion overwriting samples.
        dynmat_bz = reordered
    elif unitcell.symrel is not None:
        tnons = unitcell.tnons
        if tnons is None:
            tnons = np.zeros((len(unitcell.symrel), 3))
        dynmat_bz = _expand_dynmat_to_full_bz(
            qpoints, dynmat, qbz, unitcell.symrel, unitcell.rprimd,
            unitcell.xred, tnons, use_rotation=True,
        )
    else:
        raise ValueError(
            f"unitcell has {len(qpoints)} q-points but the full matching "
            f"grid {matching_reps} needs {nq}, and no symmetry operations are "
            "available to expand an irreducible set"
        )

    if np.iscomplexobj(dynmat_bz):
        complex_dynmat = np.asarray(dynmat_bz, dtype=complex)
        dynmat_stack = np.stack([complex_dynmat.real, complex_dynmat.imag], axis=-1)
    else:
        dynmat_stack = np.asarray(dynmat_bz, dtype=float)
        complex_dynmat = dynmat_stack[..., 0] + 1j * dynmat_stack[..., 1]
    grid_h = np.rint(qbz * np.asarray(matching_reps, dtype=float)).astype(int)
    grid_h %= np.asarray(matching_reps, dtype=int)
    if len({tuple(int(v) for v in h) for h in grid_h.tolist()}) != len(grid_h):
        raise ValueError("normalized matching grid contains duplicate residue classes")

    aligned = copy.deepcopy(unitcell)
    aligned.qpoints = qbz.copy()
    aligned.dynmat = dynmat_stack.copy()
    aligned.ngqpt = np.asarray(matching_reps, dtype=int)
    aligned.nqshft = 1
    aligned.q1shft = np.zeros((1, 3))
    aligned.symrel = None
    aligned.tnons = None
    return aligned, complex_dynmat, qbz, grid_h


def _default_eta(cell: np.ndarray) -> float:
    invcell = np.linalg.inv(cell)
    return float(
        np.pi * 100.0 / 33.0
        * np.sqrt(1.69 * np.sum(invcell @ invcell.T) / np.sum(cell @ cell.T))
    )


def _default_eta_angstrom2(cell: np.ndarray) -> float:
    return _default_eta(cell)


def _geometry_alignment(unitcell: UnitcellData) -> Dict[str, Any]:
    rcan, translation = _canonical_coordinates(unitcell.xred, unitcell.rprimd)
    phase_max = float(np.abs(translation).max()) if translation.size else 0.0
    if phase_max > _PHASE_TOL_BOHR:
        raise ValueError(
            "canonical phase alignment is not supported by this harness: "
            f"max translation {phase_max:.3e} Bohr exceeds {_PHASE_TOL_BOHR:.1e}; "
            "the upstream builder phase-shifts by these translations while the "
            "downstream site order does not, so target image assignments differ"
        )
    transposed = np.asarray(unitcell.xred) @ np.asarray(unitcell.rprimd)
    transpose_residual = float(
        np.abs(np.asarray(unitcell.xcart) - transposed).max()
    ) * BOHR_TO_ANGSTROM
    return {
        "canonical_translation_max_bohr": phase_max,
        "canonical_translation_max_angstrom": phase_max * BOHR_TO_ANGSTROM,
        "xcart_transpose_residual_angstrom": transpose_residual,
        "positions_source": "unitcell.xcart (the positions used by dipdip)",
        "cell_source": "unitcell.rprimd rows",
    }


def _ordering_residual(
    unitcell: UnitcellData,
    target_reps: Tuple[int, int, int],
    crystal_sc,
) -> float:
    """Max distance between upstream target atoms and the expected tiled order."""
    natom = unitcell.natom
    positions = np.asarray(unitcell.xcart, dtype=float)
    cell = np.asarray(unitcell.rprimd, dtype=float)
    l1, l2, l3 = target_reps
    expected = np.empty((natom * l1 * l2 * l3, 3), dtype=float)
    index = 0
    for ix in range(l1):
        for iy in range(l2):
            for iz in range(l3):
                shift = np.array([ix, iy, iz], dtype=float) @ cell.T
                for atom in range(natom):
                    expected[index] = positions[atom] + shift
                    index += 1
    actual = np.asarray(crystal_sc.xcart, dtype=float)
    # xcart = xred @ rprimd.T, so the position-space lattice matrix is rprimd.T.
    cell_sc = np.diag(target_reps).astype(float) @ cell
    cell_pos = cell_sc.T
    delta = expected - actual
    frac = delta @ np.linalg.inv(cell_pos)
    delta_mic = (frac - np.round(frac)) @ cell_pos
    return float(np.sqrt(np.einsum("ni,ni->n", delta_mic, delta_mic)).max()) * BOHR_TO_ANGSTROM


# ----------------------------------------------------------------------
# Route builders
# ----------------------------------------------------------------------


def build_downstream_correction(
    unitcell: UnitcellData,
    matching_reps: Optional[Sequence[int]] = None,
    target_reps: Optional[Sequence[int]] = None,
    *,
    ft_sign: int = 1,
    eta: Optional[float] = None,
    nreal: int = 5,
    nrecip: int = 8,
    neutrality_tolerance: float = 1e-4,
    phi0_symmetry_tol: Optional[float] = None,
    accept_measured_symmetry_violation: bool = False,
    response_provenance: Optional[Mapping[str, Any]] = None,
) -> DownstreamCorrection:
    """Build the standalone matched correction from upstream q-grid data.

    ``eta`` is in Angstrom**-2 (the downstream convention).  When omitted, the
    primitive-cell default formula is used and converted to Bohr**-2 for the
    upstream Ewald calls, so both routes share one numerical split.
    """
    matched, dipole, ifc, build_correction = _require_longrangefy()
    del matched, dipole, ifc  # only build_correction is needed here
    if matching_reps is None:
        if unitcell.ngqpt is None:
            raise ValueError("matching_reps must be supplied when unitcell.ngqpt is None")
        matching_reps = unitcell.ngqpt
    reps_m = _as_reps(matching_reps, "matching_reps")
    reps_l = _as_reps(target_reps if target_reps is not None else reps_m, "target_reps")

    aligned, dynmat, qbz, grid_h = _full_grid_data(unitcell, reps_m)
    geometry = _geometry_alignment(unitcell)

    cell = np.asarray(unitcell.rprimd, dtype=float) * BOHR_TO_ANGSTROM
    positions = np.asarray(unitcell.xcart, dtype=float) * BOHR_TO_ANGSTROM
    born = np.asarray(unitcell.zeff, dtype=float)
    dielectric = np.asarray(unitcell.epsilon_inf, dtype=float)
    if born.shape != (unitcell.natom, 3, 3):
        raise ValueError("unitcell.zeff must have shape (natom, 3, 3)")
    if dielectric.shape != (3, 3):
        raise ValueError("unitcell.epsilon_inf must have shape (3, 3)")

    if eta is None:
        eta_angstrom2 = _default_eta_angstrom2(cell)
        eta_source = "primitive-cell default formula (shared)"
    else:
        eta_angstrom2 = float(eta)
        eta_source = "caller supplied (shared)"
    if not np.isfinite(eta_angstrom2) or eta_angstrom2 <= 0.0:
        raise ValueError("eta must be a positive finite value in Angstrom**-2")
    eta_bohr2 = eta_angstrom2 * BOHR_TO_ANGSTROM**2

    phi0 = dynmat * HA_BOHR2_TO_EV_ANGSTROM2
    symmetry = measure_phi0_symmetry_violation(phi0, grid_h, reps_m)
    tolerance_source = "caller supplied" if phi0_symmetry_tol is not None else None
    if phi0_symmetry_tol is None and accept_measured_symmetry_violation:
        # Round the measured value up by a relative 1e-12 so the same
        # projection the measurement described is accepted.  An exactly zero
        # violation stays on the strict roundoff path (a zero tolerance is
        # rejected by build_correction).
        measured = float(symmetry["max_abs"])
        if measured > 0.0:
            phi0_symmetry_tol = measured * (1.0 + 1e-12)
            tolerance_source = "measured upstream violation (explicitly accepted)"
        else:
            tolerance_source = "measured upstream violation is zero; strict roundoff"

    correction = build_correction(
        cell,
        positions,
        born,
        dielectric,
        reps_m,
        phi0,
        grid_h,
        reps_l,
        ft_sign=int(ft_sign),
        eta=eta_angstrom2,
        nreal=int(nreal),
        nrecip=int(nrecip),
        neutrality_tolerance=float(neutrality_tolerance),
        phi0_symmetry_tol=phi0_symmetry_tol,
        response_provenance=dict(response_provenance) if response_provenance else None,
    )

    metadata = {
        "units": "Angstrom, eV, eV/Angstrom, eV/Angstrom**2",
        "ifc_axes": "(atom, displacement, atom, displacement), unmassweighted",
        "site_order": "image-major atom-minor: atom + natom*(iz + nz*(iy + ny*ix))",
        "fourier_convention": f"Phi(q) = sum_T kappa(T) exp({'+' if ft_sign == 1 else '-'}2i pi q.T)",
        "boundary": "conducting (macroscopic Gamma zero mode omitted)",
        "matching_reps": reps_m,
        "target_reps": reps_l,
        "eta_angstrom2": eta_angstrom2,
        "eta_bohr2": eta_bohr2,
        "eta_source": eta_source,
        "ewald_shells": {"nreal": int(nreal), "nrecip": int(nrecip)},
        "neutrality_tolerance": float(neutrality_tolerance),
        "phi0_symmetry": symmetry,
        "phi0_symmetry_tol": phi0_symmetry_tol,
        "phi0_symmetry_tolerance_source": tolerance_source,
        "n_qpoints_matching": int(len(qbz)),
        "geometry": geometry,
        "longrangefy_version": getattr(
            __import__("longrangefy"), "__version__", "unknown"
        ),
    }
    return DownstreamCorrection(
        correction=correction,
        unitcell_full=aligned,
        qbz=qbz,
        grid_h=grid_h,
        phi0_grid=phi0,
        positions=positions,
        cell=cell,
        born=born,
        dielectric=dielectric,
        matching_reps=reps_m,
        target_reps=reps_l,
        eta_angstrom2=eta_angstrom2,
        eta_bohr2=eta_bohr2,
        nreal=int(nreal),
        nrecip=int(nrecip),
        ft_sign=int(ft_sign),
        metadata=metadata,
    )


def _dense_ifc(ifc, unitcell: UnitcellData, target_reps: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    natom = unitcell.natom * int(np.prod(target_reps))
    atmfrc = np.asarray(ifc.atmfrc, dtype=float)
    short = np.asarray(ifc.short_atmfrc, dtype=float) if ifc.short_atmfrc is not None else np.zeros_like(atmfrc)
    ewald = np.asarray(ifc.ewald_atmfrc, dtype=float) if ifc.ewald_atmfrc is not None else np.zeros_like(atmfrc)
    shape = (natom, 3, natom, 3)
    if atmfrc.shape[:4] != shape:
        # The legacy (3, n, 3, n, nrpt) layout is rejected rather than guessed.
        raise ValueError(
            f"unsupported upstream IFC layout {atmfrc.shape}; expected (natom, 3, natom, 3, nrpt)"
        )
    factor = HA_BOHR2_TO_EV_ANGSTROM2
    short_dense = short.sum(axis=4).reshape(3 * natom, 3 * natom) * factor
    ewald_dense = ewald.sum(axis=4).reshape(3 * natom, 3 * natom) * factor
    total_dense = atmfrc.sum(axis=4).reshape(3 * natom, 3 * natom) * factor
    return short_dense, ewald_dense, total_dense


def _accepted_phi0_unitcell(downstream: DownstreamCorrection) -> UnitcellData:
    """Upstream-unitcell copy carrying the downstream-accepted phi0 samples.

    Both harmonic representations must be built from the same reference data.
    ``MatchedCorrection.phi0_samples`` is the Hermitian projection actually
    used downstream; feeding it back to the upstream builder removes the
    raw-vs-projected sample difference from the matrix comparison.
    """
    samples = downstream.correction.phi0_samples
    if samples is None:
        raise ValueError("longrangefy correction exposes no accepted phi0 samples")
    tiled = _tiled_grid(downstream.matching_reps)
    lookup = {tuple(int(v) for v in h): k for k, h in enumerate(tiled.tolist())}
    order = [lookup[tuple(int(v) for v in h)] for h in downstream.grid_h.tolist()]
    accepted = np.asarray(samples, dtype=complex)[order]
    unitcell = copy.deepcopy(downstream.unitcell_full)
    unitcell.qpoints = downstream.qbz.copy()
    unitcell.dynmat = np.stack(
        [accepted.real, accepted.imag], axis=-1
    ) / HA_BOHR2_TO_EV_ANGSTROM2
    return unitcell


def build_upstream_branches(
    unitcell: UnitcellData,
    target_reps: Sequence[int],
    *,
    asr: bool = False,
    eta_bohr2: Optional[float] = None,
    nreal: int = 5,
    nrecip: int = 8,
) -> Tuple[BranchMatrices, UnitcellData, Any]:
    """Build the existing split and unsplit supercell routes from one q-grid.

    The unitcell is deep-copied for each call because the Fourier builder can
    otherwise mutate ``unitcell.dynmat`` in place when ``symrel`` is absent.
    The returned split matrix is ``short_range + ewald``; ``atmfrc`` alone is
    only the short-range part on the ``dipdip=True`` route.
    """
    reps = _as_reps(target_reps, "target_reps")
    split_sc = build_supercell(
        copy.deepcopy(unitcell), reps, dipdip=True, asr=asr,
        ewald_eta=eta_bohr2, ewald_nreal=nreal, ewald_nrecip=nrecip,
    )
    unsplit_sc = build_supercell(
        copy.deepcopy(unitcell), reps, dipdip=False, asr=asr,
        ewald_eta=eta_bohr2, ewald_nreal=nreal, ewald_nrecip=nrecip,
    )
    if split_sc.natom_sc != unsplit_sc.natom_sc:
        raise ValueError("split and unsplit supercells disagree in atom count")
    short, ewald, atmfrc = _dense_ifc(split_sc.ifcs_sc, unitcell, reps)
    _, _, unsplit = _dense_ifc(unsplit_sc.ifcs_sc, unitcell, reps)
    if not np.allclose(atmfrc, short):
        raise ValueError(
            "upstream dipdip=True atmfrc differs from short_atmfrc; this route "
            "is not the expected short-range storage and cannot be aligned"
        )
    split = short + ewald
    branches = BranchMatrices(
        split=split,
        unsplit=unsplit,
        correction=split - unsplit,
        short_range=short,
        ewald=ewald,
        asr=bool(asr),
    )
    return branches, split_sc.crystal_sc, split_sc.ifcs_sc


# ----------------------------------------------------------------------
# Comparison
# ----------------------------------------------------------------------


def _dipole_kernel_comparison(
    unitcell: UnitcellData,
    downstream: DownstreamCorrection,
    dipole,
) -> Dict[str, Any]:
    from .pyeffpot.dipdip import compute_dipdip_dynmat

    q_list = [
        np.zeros(3),
        np.asarray(downstream.grid_h[1], dtype=float) / np.asarray(downstream.matching_reps),
        np.asarray(downstream.grid_h[-1], dtype=float) / np.asarray(downstream.matching_reps),
    ]
    entries = []
    for q in q_list:
        upstream = compute_dipdip_dynmat(
            q, unitcell, eta=downstream.eta_bohr2,
            nreal=downstream.nreal, nrecip=downstream.nrecip, sumg0=1,
        )
        downstream_ha = dipole.dipole_ifc(
            downstream.cell, downstream.positions, downstream.born,
            downstream.dielectric, q, eta=downstream.eta_angstrom2,
            nreal=downstream.nreal, nrecip=downstream.nrecip,
        ) / HA_BOHR2_TO_EV_ANGSTROM2
        scale = float(np.abs(upstream).max()) if upstream.size else 0.0
        delta = downstream_ha - upstream
        entries.append(
            {
                "q_fractional": np.asarray(q, dtype=float),
                "reference_scale_Ha_Bohr2": scale,
                "max_abs_Ha_Bohr2": float(np.abs(delta).max()),
                "relative_max": float(np.abs(delta).max() / scale) if scale else float("nan"),
            }
        )
    return {
        "boundary": "conducting Gamma (G=0 macroscopic term omitted by both)",
        "eta_policy": "same primitive-cell eta on both sides (direct kernel check)",
        "q_points": entries,
    }


def _weight_comparison(
    unitcell: UnitcellData,
    downstream: DownstreamCorrection,
    ifc,
    matched,
) -> Dict[str, Any]:
    reps = downstream.matching_reps
    rpoints_cols = _bigbx9_rpoints(np.asarray(reps), 1)
    rpoints = rpoints_cols.T
    nqbz = int(np.prod(reps))
    rcan, _ = _canonical_coordinates(unitcell.xred, unitcell.rprimd)
    upstream = _compute_rpoint_weights(
        rpoints_cols, unitcell.rprimd, rcan, np.asarray(reps), nqbz,
    )
    reference = ifc.wigner_seitz_weights(
        rpoints, downstream.cell, rcan * BOHR_TO_ANGSTROM, reps,
    )
    positions_reference = ifc.wigner_seitz_weights(
        rpoints, downstream.cell, downstream.positions, reps,
    )
    sums = upstream.sum(axis=2)
    report: Dict[str, Any] = {
        "rpoint_count": int(rpoints.shape[0]),
        "weight_sum_rule": {
            "expected_per_pair": nqbz,
            "max_abs_residual": float(np.abs(sums - nqbz).max()),
        },
        "upstream_vs_reference_max_abs": float(np.abs(upstream - reference).max()),
        "upstream_vs_positions_reference_max_abs": float(
            np.abs(upstream - positions_reference).max()
        ),
        "reference": "longrangefy.ifc.wigner_seitz_weights (independent adapted wght9)",
    }
    pair_ws = getattr(matched, "_pair_ws_weights", None)
    if callable(pair_ws):
        try:
            _, _, _, class_residual = pair_ws(
                downstream.cell, downstream.positions, reps,
            )
            report["matched_candidate_weights"] = {
                "available": True,
                "class_sum_residual": float(class_residual),
                "note": (
                    "the matched path folds per-class candidates onto the target "
                    "residues; that operator is not the same array as the upstream "
                    "per-R weights, so only its class-sum residual is compared here"
                ),
            }
        except Exception as exc:  # pragma: no cover - diagnostic only
            report["matched_candidate_weights"] = {
                "available": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
    else:
        report["matched_candidate_weights"] = {
            "available": False,
            "error": "longrangefy.matched does not expose _pair_ws_weights",
        }
    return report


def _offgrid_comparison(
    downstream: DownstreamCorrection,
    upstream: BranchMatrices,
) -> Dict[str, Any]:
    mc = downstream.correction
    reps_m = np.asarray(downstream.matching_reps)
    reps_l = np.asarray(downstream.target_reps)
    natom = upstream.natom_target // int(np.prod(reps_l))

    # A shared wavevector is target-periodic and lies on the matching grid.
    shared = [
        h for h in _tiled_grid(downstream.matching_reps)
        if np.all((reps_l[None, :] * np.asarray(h)[None, :]) % reps_m[None, :] == 0)
    ]
    new = []
    for h in _tiled_grid(downstream.target_reps):
        q = np.asarray(h, dtype=float) / reps_l
        on_matching = np.asarray(q * reps_m)
        if np.max(np.abs(on_matching - np.round(on_matching))) > _GRID_TOL:
            new.append(q)
    report: Dict[str, Any] = {
        "target_grid_point_count": int(np.prod(reps_l)),
        "shared_wavevector_count": len(shared),
        "new_wavevector_count": len(new),
    }

    selected_sign = downstream.ft_sign

    def compare_at(q: np.ndarray) -> Dict[str, Any]:
        reference_split_q, trans_res = _target_matrix_at_q(
            upstream.split, q, downstream.target_reps, natom, selected_sign,
        )
        reference_unsplit_q, _ = _target_matrix_at_q(
            upstream.unsplit, q, downstream.target_reps, natom, selected_sign,
        )
        reference_corr_q = reference_split_q - reference_unsplit_q
        downstream_split_q = mc.branch_dynamical(q, "split")
        downstream_unsplit_q = mc.branch_dynamical(q, "unsplit")
        downstream_corr_q = mc.branch_dynamical(q, "difference")
        return {
            "q_fractional": np.asarray(q, dtype=float),
            "translation_invariance_residual": trans_res,
            "split_max_abs": float(np.abs(downstream_split_q - reference_split_q).max()),
            "unsplit_max_abs": float(np.abs(downstream_unsplit_q - reference_unsplit_q).max()),
            "correction_max_abs": float(np.abs(downstream_corr_q - reference_corr_q).max()),
            "reference_correction_scale": float(np.abs(reference_corr_q).max()),
        }

    if shared:
        # Choose a nonzero shared wavevector so the phase-sign test is not
        # degenerate at Gamma.
        shared_probe = max(
            shared,
            key=lambda h: float(np.linalg.norm(
                np.asarray(h, dtype=float) / reps_m
            )),
        )
        q_shared = np.asarray(shared_probe, dtype=float) / reps_m
        if mc.phi0_samples is not None:
            # The matching-grid block at this shared point is the accepted sample.
            index = int(
                np.where(
                    np.all(
                        _tiled_grid(downstream.matching_reps) == np.asarray(shared_probe),
                        axis=1,
                    )
                )[0][0]
            )
            reference_block = np.asarray(mc.phi0_samples[index], dtype=complex).real
            selected_sign, sign_report = _select_phase_sign(
                upstream.unsplit, q_shared, downstream.target_reps, natom, reference_block,
            )
            report["phase_sign_selection"] = {
                "selected": selected_sign,
                "residuals": sign_report,
                "reference": "accepted matching-grid sample at a shared wavevector",
            }
        else:
            report["phase_sign_selection"] = {"selected": selected_sign, "residuals": {}}
        report["shared_point"] = compare_at(q_shared)
    else:
        report["shared_point"] = None
        report["phase_sign_selection"] = {"selected": downstream.ft_sign, "residuals": {}}

    if new:
        # Prefer the smallest new wavevector for a reproducible probe.
        q_new = min(new, key=lambda q: tuple(np.round(q, 12)))
        report["new_point"] = compare_at(q_new)
    else:
        report["new_point"] = None
    return report


def _energy_force_comparison(
    upstream: BranchMatrices,
    downstream: DownstreamCorrection,
    displacements: Optional[Iterable[np.ndarray]],
) -> Dict[str, Any]:
    natom = upstream.natom_target
    if displacements is None:
        rng = np.random.default_rng(0)
        patterns = [rng.normal(size=3 * natom) * 0.01]
        source = "normal(seed=0) * 0.01 Angstrom"
    else:
        patterns = [np.asarray(value, dtype=float).reshape(-1) for value in displacements]
        source = "caller supplied"
    mc = downstream.correction
    entries = []
    for index, u in enumerate(patterns):
        if u.shape[0] != 3 * natom:
            raise ValueError(
                f"displacement {index} carries {u.shape[0]} components; "
                f"expected {3 * natom} in target-atom order"
            )
        up_energy = {}
        up_force = {}
        for name, matrix in (
            ("split", upstream.split), ("unsplit", upstream.unsplit),
            ("correction", upstream.correction),
        ):
            up_energy[name] = float(0.5 * u @ matrix @ u)
            up_force[name] = -(matrix @ u)
        down_energy = {
            "split": float(0.5 * u @ downstream.correction.split_matrix @ u),
            "unsplit": float(0.5 * u @ downstream.correction.unsplit_matrix @ u),
            "correction": float(mc.energy(u)),
        }
        down_force = {
            "split": -(downstream.correction.split_matrix @ u),
            "unsplit": -(downstream.correction.unsplit_matrix @ u),
            "correction": mc.forces(u),
        }
        entry = {"index": index, "u_norm_angstrom": float(np.linalg.norm(u))}
        for name in ("split", "unsplit", "correction"):
            energy_scale = max(abs(up_energy[name]), abs(down_energy[name]), 1e-30)
            force_scale = max(
                float(np.linalg.norm(up_force[name])),
                float(np.linalg.norm(down_force[name])),
                1e-30,
            )
            entry[name] = {
                "upstream_energy_eV": up_energy[name],
                "downstream_energy_eV": down_energy[name],
                "energy_abs_difference_eV": abs(up_energy[name] - down_energy[name]),
                "energy_relative_difference": abs(up_energy[name] - down_energy[name]) / energy_scale,
                "force_max_abs_difference_eV_A": float(
                    np.abs(up_force[name] - down_force[name]).max()
                ),
                "force_rms_difference_eV_A": float(
                    np.sqrt(np.mean((up_force[name] - down_force[name]) ** 2))
                ),
                "force_scale_eV_A": force_scale,
            }
        entries.append(entry)
    return {"pattern_source": source, "entries": entries}


def _difference_floors(
    downstream: DownstreamCorrection, dipole, use_accepted_phi0: bool
) -> Dict[str, Any]:
    """Known numerical floors that explain part of the reported differences."""
    coulomb = getattr(dipole, "_COULOMB", None)
    hartree_bohr = HARTREE_TO_EV * BOHR_TO_ANGSTROM
    relative = None
    if coulomb is not None and coulomb != 0.0:
        relative = float((hartree_bohr - coulomb) / coulomb)
    symmetrization = downstream.correction.metadata.get("phi0_symmetrization", {})
    return {
        "unit_constants": {
            "hartree_to_eV": HARTREE_TO_EV,
            "bohr_to_angstrom": BOHR_TO_ANGSTROM,
            "ha_bohr2_to_eV_A2": HA_BOHR2_TO_EV_ANGSTROM2,
            "longrangefy_coulomb_constant_eV_A": coulomb,
            "hartree_times_bohr_eV_A": hartree_bohr,
            "relative_coulomb_difference": relative,
        },
        "phi0_symmetrization_applied_max_abs_eV_A2": symmetrization.get(
            "applied_max_abs_change"
        ),
        "note": (
            "The upstream conversion uses the module Hartree/Bohr constants; "
            "longrangefy's dipole kernel uses its own Coulomb constant. Their "
            "measured relative difference is a floor on direct dipole, "
            "correction and force differences, not an interpolation error. "
            + (
                "Both representations are built from the same accepted samples "
                "here; the recorded projection scale is reported for provenance, "
                "not as an expected branch difference."
                if use_accepted_phi0 else
                "The upstream builder uses the raw q-grid samples while the "
                "downstream construction applies the recorded Hermitian "
                "projection, so split/unsplit matrices can additionally differ "
                "at the recorded projection scale."
            )
        ),
    }


def compare_matched_correction(
    unitcell: UnitcellData,
    matching_reps: Optional[Sequence[int]] = None,
    target_reps: Optional[Sequence[int]] = None,
    *,
    asr: bool = False,
    ft_sign: int = 1,
    eta: Optional[float] = None,
    nreal: int = 5,
    nrecip: int = 8,
    neutrality_tolerance: float = 1e-4,
    displacements: Optional[Iterable[np.ndarray]] = None,
    phi0_symmetry_tol: Optional[float] = None,
    accept_measured_symmetry_violation: bool = False,
    align_ewald_eta: bool = False,
    use_accepted_phi0: bool = True,
    convergence_eta_factors: Sequence[float] = (),
    response_provenance: Optional[Mapping[str, Any]] = None,
) -> ParityReport:
    """Compare the q-grid upstream route with the downstream matched correction.

    ``asr=True`` is allowed but is *not* aligned: the downstream construction
    applies no acoustic projection, so the ASR effect is measured and reported
    instead of being treated as parity.  The main matrix comparison is therefore
    like-for-like only with ``asr=False``.

    ``align_ewald_eta=False`` lets each route use its own per-cell Ewald
    splitting parameter (the physically equivalent, numerically balanced
    choice) and records both.  ``align_ewald_eta=True`` forces the downstream
    primitive-cell value on the upstream target-cell Ewald; this can increase
    truncation error and is intended for convergence studies, not as a
    presumed improvement.

    ``use_accepted_phi0=True`` (default) feeds the upstream builder the same
    Hermitian-projected matching-grid samples the downstream construction uses.
    Set it to ``False`` to compare raw upstream samples against the downstream
    projection; the recorded ``phi0_symmetrization`` diagnostics then explain
    the additional branch difference.
    """
    matched, dipole, ifc, _ = _require_longrangefy()
    if matching_reps is None and unitcell.ngqpt is None:
        raise ValueError("matching_reps must be supplied when unitcell.ngqpt is None")
    reps_m = _as_reps(
        matching_reps if matching_reps is not None else unitcell.ngqpt,
        "matching_reps",
    )
    reps_l = _as_reps(
        target_reps if target_reps is not None else reps_m,
        "target_reps",
    )
    downstream = build_downstream_correction(
        unitcell, reps_m, reps_l, ft_sign=ft_sign, eta=eta,
        nreal=nreal, nrecip=nrecip,
        neutrality_tolerance=neutrality_tolerance,
        phi0_symmetry_tol=phi0_symmetry_tol,
        accept_measured_symmetry_violation=accept_measured_symmetry_violation,
        response_provenance=response_provenance,
    )
    upstream_eta = downstream.eta_bohr2 if align_ewald_eta else None
    upstream_input = (
        _accepted_phi0_unitcell(downstream) if use_accepted_phi0
        else downstream.unitcell_full
    )
    upstream, crystal_sc, _ = build_upstream_branches(
        upstream_input, reps_l, asr=asr,
        eta_bohr2=upstream_eta, nreal=nreal, nrecip=nrecip,
    )
    ordering_residual = _ordering_residual(
        downstream.unitcell_full, reps_l, crystal_sc,
    )
    if ordering_residual > 1e-6:
        raise ValueError(
            "upstream target atom order/positions differ from the declared "
            f"image-major order by {ordering_residual:.3e} Angstrom; refusing "
            "to compare misaligned matrices"
        )

    matrices = {
        "split": _matrix_difference("split", upstream.split, downstream.correction.split_matrix),
        "unsplit": _matrix_difference("unsplit", upstream.unsplit, downstream.correction.unsplit_matrix),
        "correction": _matrix_difference(
            "correction", upstream.correction, downstream.correction.correction_matrix
        ),
    }
    target_dipole = _target_dipole_dense(
        dipole, ifc, downstream.cell, downstream.positions, downstream.born,
        downstream.dielectric, reps_l, downstream.eta_angstrom2,
        nreal, nrecip,
    )
    downstream_short = downstream.correction.split_matrix - target_dipole
    components = {
        "ewald": _matrix_difference("ewald", upstream.ewald, target_dipole),
        "short_range": _matrix_difference(
            "short_range", upstream.short_range, downstream_short
        ),
        "counterterm": _matrix_difference(
            "counterterm", upstream.correction,
            downstream.correction.counterterm_matrix(),
        ),
    }

    asr_report: Dict[str, Any] = {
        "upstream_asr": "weighted wght9 asrif9 + unweighted union diagonal" if asr else "none",
        "downstream_asr": "none (build_correction uses the supplied data exactly)",
        "aligned": not asr,
    }
    if asr:
        upstream_no_asr, _, _ = build_upstream_branches(
            upstream_input, reps_l, asr=False,
            eta_bohr2=upstream_eta, nreal=nreal, nrecip=nrecip,
        )
        asr_report["effect"] = {
            name: _matrix_difference(
                f"asr_effect_{name}",
                getattr(upstream_no_asr, name),
                getattr(upstream, name),
            ).as_dict()
            for name in ("split", "unsplit", "correction")
        }
        asr_report["note"] = (
            "ASR is an upstream-only projection here; matrix differences are "
            "expected and are not evidence of an interpolation error"
        )

    convergence: Dict[str, Any] = {
        "downstream_eta_sensitivity": downstream.correction.eta_sensitivity((0.5, 2.0)),
        "ewald": downstream.correction.metadata.get("ewald"),
        "upstream_ewald_shells": {"nreal": int(nreal), "nrecip": int(nrecip)},
        "upstream_eta_bohr2": upstream_eta,
        "upstream_eta_selection": (
            "shared downstream primitive-cell eta" if align_ewald_eta
            else "per-cell dipdip defaults"
        ),
        "upstream_default_etas_bohr2": {
            "q_grid_subtraction": _default_eta(np.asarray(unitcell.rprimd, dtype=float)),
            "target_ewald": _default_eta(
                np.diag(reps_l).astype(float) @ np.asarray(unitcell.rprimd, dtype=float)
            ),
        },
        "downstream_eta_angstrom2": downstream.eta_angstrom2,
        "downstream_eta_bohr2": downstream.eta_bohr2,
        "phi0_symmetry": downstream.metadata["phi0_symmetry"],
        "phi0_symmetrization": downstream.correction.metadata.get("phi0_symmetrization"),
        "dipole_reciprocity_residual": downstream.correction.metadata.get(
            "dipole_reciprocity_residual"
        ),
        "weight_class_sum_residual": downstream.correction.metadata.get(
            "weight_class_sum_residual"
        ),
    }
    if convergence_eta_factors:
        scaled = {}
        for factor in convergence_eta_factors:
            factor = float(factor)
            alternative, _, _ = build_upstream_branches(
                upstream_input, reps_l, asr=asr,
                eta_bohr2=downstream.eta_bohr2 * factor,
                nreal=nreal, nrecip=nrecip,
            )
            scaled[f"eta_x{factor:g}"] = _matrix_difference(
                f"upstream_eta_x{factor:g}", upstream.correction, alternative.correction
            ).as_dict()
        convergence["upstream_eta_sensitivity"] = scaled
    else:
        convergence["upstream_eta_sensitivity"] = {
            "available": False,
            "reason": "pass convergence_eta_factors to rebuild the upstream path",
        }

    alignment = {
        "units": "upstream Hartree/Bohr converted to eV/Angstrom with the module constants",
        "conversion_factor_Ha_Bohr2_to_eV_A2": HA_BOHR2_TO_EV_ANGSTROM2,
        "atom_order": "image-major atom-minor (validated to 1e-6 Angstrom)",
        "ordering_residual_angstrom": ordering_residual,
        "fourier_sign": ft_sign,
        "phase": downstream.metadata["geometry"],
        "eta_shared": bool(align_ewald_eta),
        "upstream_phi0": (
            "downstream-accepted Hermitian-projected samples"
            if use_accepted_phi0 else "raw upstream samples"
        ),
        "eta_note": (
            "the physical Ewald result is eta independent; per-cell defaults "
            "keep each route's sums balanced, while align_ewald_eta forces one "
            "numerical split and may increase truncation error"
        ),
        "boundary": "conducting Gamma (both)",
        "asr_aligned": not asr,
        "expected_difference_floors": _difference_floors(
            downstream, dipole, use_accepted_phi0
        ),
        "parity_claim": None,
        "acceptance_note": (
            "measurements only; no tolerance is declared by this harness"
        ),
    }
    metadata = dict(downstream.metadata)
    metadata.update(
        {
            "upstream_route": "build_supercell(q-grid Fourier, dipdip=True/False)",
            "downstream_route": "longrangefy.matched.build_correction",
            "upstream_phi0_source": (
                "downstream-accepted Hermitian projection"
                if use_accepted_phi0 else "raw upstream q-grid samples"
            ),
            "split_storage_note": (
                "upstream atmfrc is short-range on the dipdip=True route; "
                "split = atmfrc + ewald_atmfrc"
            ),
            "ifc_layout_validation": "atmfrc == short_atmfrc on dipdip=True",
        }
    )
    return ParityReport(
        alignment=alignment,
        matrices=matrices,
        components=components,
        energy_force=_energy_force_comparison(upstream, downstream, displacements),
        weights=_weight_comparison(unitcell, downstream, ifc, matched),
        dipole_kernel=_dipole_kernel_comparison(unitcell, downstream, dipole),
        offgrid=_offgrid_comparison(downstream, upstream),
        asr=asr_report,
        convergence=convergence,
        metadata=metadata,
        upstream=upstream,
        downstream=downstream,
    )


__all__ = [
    "HA_BOHR2_TO_EV_ANGSTROM2",
    "BranchMatrices",
    "DownstreamCorrection",
    "MatrixDifference",
    "ParityReport",
    "build_downstream_correction",
    "build_upstream_branches",
    "compare_matched_correction",
    "longrangefy_available",
    "measure_phi0_symmetry_violation",
]
