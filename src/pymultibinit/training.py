"""MULTIBINIT model-building and pure-Python fitting helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from itertools import combinations_with_replacement
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Mapping, Optional, Sequence

import numpy as np


HIST_UNITS = {
    "rprimd": "Bohr",
    "xcart": "Bohr",
    "xred": "fractional",
    "energy": "Hartree",
    "forces": "Hartree/Bohr",
    "stress": "Hartree/Bohr^3",
}


@dataclass(frozen=True)
class PythonFitConfig:
    """Configuration for pure-Python coefficient fitting.

    The target ordering for ``fit_on`` and ``fit_factors`` is forces, stress,
    then energy, matching the fitting architecture documents.
    """

    ncell: tuple[int, int, int]
    fit_on: tuple[bool, bool, bool] = (True, True, True)
    fit_factors: tuple[float, float, float] = (1.0, 1.0, 1.0)
    regularization: float = 0.0
    selection: str = "all"
    ncoeff: Optional[int] = None
    cutoff: Optional[float] = None
    power_range: tuple[int, int] = (3, 4)
    feature_backend: str = "auto"
    feature_memmap_dir: Optional[str] = None
    candidate_pool_size: Optional[int] = None
    feature_chunk_size: int = 512
    screening_frame_count: Optional[int] = None

    def __post_init__(self) -> None:
        ncell = _tuple3_int(self.ncell, "ncell")
        fit_on = _tuple3_bool(self.fit_on, "fit_on")
        fit_factors = _tuple3_float(self.fit_factors, "fit_factors")
        power_range = _tuple2_int(self.power_range, "power_range")

        if any(value <= 0 for value in ncell):
            raise ValueError("ncell values must be positive")
        if any(not np.isfinite(value) for value in fit_factors):
            raise ValueError("fit_factors values must be finite")
        if any(value < 0.0 for value in fit_factors):
            raise ValueError("fit_factors values must be non-negative")
        if not np.isfinite(self.regularization):
            raise ValueError("regularization must be finite")
        if self.regularization < 0.0:
            raise ValueError("regularization must be non-negative")
        if self.selection not in {"all", "greedy", "screened_greedy", "lasso"}:
            raise ValueError("selection must be one of: all, greedy, screened_greedy, lasso")
        if self.feature_backend not in {"auto", "numpy", "jax", "legacy", "memmap"}:
            raise ValueError("feature_backend must be one of: auto, numpy, jax, legacy, memmap")
        if self.ncoeff is not None and self.ncoeff <= 0:
            raise ValueError("ncoeff must be positive when provided")
        if self.candidate_pool_size is not None and self.candidate_pool_size <= 0:
            raise ValueError("candidate_pool_size must be positive when provided")
        if self.feature_chunk_size <= 0:
            raise ValueError("feature_chunk_size must be positive")
        if self.screening_frame_count is not None and self.screening_frame_count <= 0:
            raise ValueError("screening_frame_count must be positive when provided")
        if self.cutoff is not None and not np.isfinite(self.cutoff):
            raise ValueError("cutoff must be finite when provided")
        if self.cutoff is not None and self.cutoff <= 0.0:
            raise ValueError("cutoff must be positive when provided")
        if power_range[0] > power_range[1]:
            raise ValueError("power_range must be ordered as (min_power, max_power)")
        if any(value < 0 for value in power_range):
            raise ValueError("power_range values must be non-negative")

        object.__setattr__(self, "ncell", ncell)
        object.__setattr__(self, "fit_on", fit_on)
        object.__setattr__(self, "fit_factors", fit_factors)
        object.__setattr__(self, "regularization", float(self.regularization))
        object.__setattr__(self, "ncoeff", int(self.ncoeff) if self.ncoeff is not None else None)
        object.__setattr__(self, "candidate_pool_size", int(self.candidate_pool_size) if self.candidate_pool_size is not None else None)
        object.__setattr__(self, "feature_chunk_size", int(self.feature_chunk_size))
        object.__setattr__(self, "screening_frame_count", int(self.screening_frame_count) if self.screening_frame_count is not None else None)
        object.__setattr__(self, "cutoff", float(self.cutoff) if self.cutoff is not None else None)
        object.__setattr__(self, "power_range", power_range)


@dataclass(frozen=True)
class TrainingFrame:
    """One ABINIT HIST frame in raw ABINIT units.

    Units are exposed via ``units`` and are: ``rprimd``/``xcart`` in Bohr,
    ``xred`` fractional, ``energy`` in Hartree, ``forces`` in Hartree/Bohr,
    and ``stress`` in Hartree/Bohr^3 with ABINIT Voigt ordering.
    """

    rprimd: np.ndarray
    xred: np.ndarray
    xcart: np.ndarray
    energy: float
    forces: np.ndarray
    stress: np.ndarray
    units: Mapping[str, str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        object.__setattr__(self, "rprimd", _array_shape(self.rprimd, (3, 3), "rprimd"))
        xred = np.asarray(self.xred, dtype=float)
        xcart = np.asarray(self.xcart, dtype=float)
        forces = np.asarray(self.forces, dtype=float)
        stress = _array_shape(self.stress, (6,), "stress")
        if xred.ndim != 2 or xred.shape[1] != 3:
            raise ValueError("xred must have shape (natom, 3)")
        if xcart.shape != xred.shape:
            raise ValueError("xcart must have the same shape as xred")
        if forces.shape != xred.shape:
            raise ValueError("forces must have the same shape as xred")
        object.__setattr__(self, "xred", xred)
        object.__setattr__(self, "xcart", xcart)
        object.__setattr__(self, "forces", forces)
        object.__setattr__(self, "stress", stress)
        object.__setattr__(self, "energy", float(self.energy))
        object.__setattr__(self, "units", dict(HIST_UNITS))


@dataclass(frozen=True)
class TrainingDataset:
    """Mapped fitting dataset in Python array order ``(time, ...)``."""

    displacement: np.ndarray
    du_delta: np.ndarray
    strain: np.ndarray
    ucvol: np.ndarray
    sqomega: np.ndarray
    energy_diff: np.ndarray
    force_diff: np.ndarray
    stress_diff: np.ndarray


@dataclass(frozen=True)
class FitFeatureMatrices:
    """Per-coefficient linear features in Python array order."""

    energy: np.ndarray
    forces: np.ndarray
    stress: np.ndarray

    def __post_init__(self) -> None:
        energy = np.asarray(self.energy, dtype=float)
        forces = np.asarray(self.forces, dtype=float)
        stress = np.asarray(self.stress, dtype=float)
        if energy.ndim != 2:
            raise ValueError("energy features must have shape (time, ncoeff)")
        if forces.ndim != 4 or forces.shape[2] != 3:
            raise ValueError("forces features must have shape (time, natom, 3, ncoeff)")
        if stress.ndim != 3 or stress.shape[1] != 6:
            raise ValueError("stress features must have shape (time, 6, ncoeff)")
        ntime, ncoeff = energy.shape
        if forces.shape[0] != ntime or stress.shape[0] != ntime:
            raise ValueError("feature time dimensions must match")
        if forces.shape[3] != ncoeff or stress.shape[2] != ncoeff:
            raise ValueError("feature coefficient dimensions must match")
        if not (np.isfinite(energy).all() and np.isfinite(forces).all() and np.isfinite(stress).all()):
            raise ValueError("feature arrays must be finite")
        object.__setattr__(self, "energy", energy)
        object.__setattr__(self, "forces", forces)
        object.__setattr__(self, "stress", stress)


@dataclass(frozen=True)
class GoalFunctionComponents:
    """MULTIBINIT-style goal-function components."""

    force_stress: float
    force: float
    stress: float
    energy: float


@dataclass(frozen=True)
class FitDiagnostics:
    """Linear solve diagnostics."""

    goal: GoalFunctionComponents
    residual_norm: float
    matrix_rank: int
    condition_number: float
    regularization: float
    info: int


@dataclass(frozen=True)
class LinearFitResult:
    """Weighted least-squares solve result."""

    coefficients: np.ndarray
    diagnostics: FitDiagnostics


@dataclass(frozen=True)
class PythonFitResult:
    """Public result for pure-Python model fitting."""

    coefficients: np.ndarray
    diagnostics: FitDiagnostics
    output_xml: Optional[str]
    ncoeff: int
    nframes: int
    ddb: str
    hist: str
    basis_xml: str
    selection_steps: tuple[Mapping[str, object], ...] = ()

    def to_dict(self) -> dict:
        data = asdict(self)
        data["coefficients"] = self.coefficients.tolist()
        return data


@dataclass(frozen=True)
class GreedySelectionResult:
    """Greedy coefficient selection result."""

    selected: tuple[int, ...]
    coefficients: np.ndarray
    diagnostics: FitDiagnostics
    steps: tuple[Mapping[str, object], ...]


@dataclass(frozen=True, order=True)
class PairKey:
    """Canonical displacement-difference factor key."""

    direction: int
    atom_a: int
    atom_b: int
    cell_b: tuple[int, int, int] = (0, 0, 0)


@dataclass(frozen=True, order=True)
class MonomialKey:
    """Canonical monomial key as sorted ``(factor, power)`` entries."""

    factors: tuple[tuple[PairKey, int], ...]


@dataclass(frozen=True)
class FortranPairList:
    """Fortran-style anchored displacement-pair list diagnostics."""

    ncoeff_sym: int
    ncoeff_total: int
    irreducible: tuple[PairKey, ...]
    factors: tuple[PairKey, ...]
    sym_factors: tuple[tuple[PairKey | None, ...], ...]
    sym_indices: tuple[tuple[int, ...], ...]
    sym_signs: tuple[tuple[int, ...], ...]
    cells: tuple[tuple[int, int, int], ...]
    dist: np.ndarray


@dataclass(frozen=True)
class XmlBasisFunction:
    """Internal immutable representation of one XML-loaded coefficient basis."""

    number: int
    value: float
    text: str
    terms: tuple[Mapping[str, object], ...]


def read_hist_frames(filename) -> list[TrainingFrame]:
    """Read ABINIT ``HIST.nc`` frames for pure-Python fitting.

    The returned arrays keep ABINIT HIST units without conversion: Bohr,
    Hartree, Hartree/Bohr, and Hartree/Bohr^3.
    """
    from scipy.io import netcdf_file

    path = _existing_path(filename, "HIST file")
    with netcdf_file(str(path), "r", mmap=False) as nc:
        rprimd = _required_var(nc, "rprimd", ndim=3)
        xred = _required_var(nc, "xred", ndim=3)
        xcart = _required_var(nc, "xcart", ndim=3)
        etotal = _required_var(nc, "etotal", ndim=1)
        fcart = _required_var(nc, "fcart", ndim=3)
        strten = _required_var(nc, "strten", ndim=2)

    ntime = rprimd.shape[0]
    if ntime == 0:
        raise ValueError(f"HIST file contains no frames: {path}")
    expected_time = {
        "xred": xred.shape[0],
        "xcart": xcart.shape[0],
        "etotal": etotal.shape[0],
        "fcart": fcart.shape[0],
        "strten": strten.shape[0],
    }
    mismatched = {name: size for name, size in expected_time.items() if size != ntime}
    if mismatched:
        raise ValueError(f"HIST variable time dimensions do not match rprimd={ntime}: {mismatched}")
    if xred.shape[1:] != xcart.shape[1:] or xred.shape[1:] != fcart.shape[1:]:
        raise ValueError("HIST xred, xcart, and fcart atom dimensions must match")
    if rprimd.shape[1:] != (3, 3):
        raise ValueError("HIST rprimd must have shape (time, 3, 3)")
    if xred.shape[2] != 3:
        raise ValueError("HIST xred must have shape (time, natom, 3)")
    if strten.shape[1] != 6:
        raise ValueError("HIST strten must have shape (time, 6)")

    return [
        TrainingFrame(
            rprimd=rprimd[iframe],
            xred=xred[iframe],
            xcart=xcart[iframe],
            energy=float(etotal[iframe]),
            forces=fcart[iframe],
            stress=strten[iframe],
        )
        for iframe in range(ntime)
    ]


def build_training_dataset(reference, frames, fixed_model=None) -> TrainingDataset:
    """Map HIST frames to a reference structure and compute fit residuals.

    ``fixed_model`` may be any object with ``evaluate(xcart, rprimd)`` returning
    ``(energy, forces, stress)`` in ABINIT units. Stress may be Voigt-6 or a
    symmetric 3x3 tensor.
    """
    frame_list = list(frames)
    if not frame_list:
        raise ValueError("Cannot build training dataset with no frames")

    ref_rprimd, ref_xcart, ref_xred = _reference_lattice_positions(reference)
    natom = ref_xcart.shape[0]
    ntime = len(frame_list)

    displacement = np.zeros((ntime, natom, 3), dtype=float)
    du_delta = np.zeros((ntime, 6, natom, 3), dtype=float)
    strain = np.zeros((ntime, 6), dtype=float)
    ucvol = np.zeros(ntime, dtype=float)
    sqomega = np.zeros(ntime, dtype=float)
    energy_diff = np.zeros(ntime, dtype=float)
    force_diff = np.zeros((ntime, natom, 3), dtype=float)
    stress_diff = np.zeros((ntime, 6), dtype=float)

    inv_ref_rprimd = np.linalg.inv(ref_rprimd)
    for iframe, frame in enumerate(frame_list):
        if frame.xcart.shape[0] != natom:
            raise ValueError(f"Frame {iframe} atom count mismatch: expected {natom}, got {frame.xcart.shape[0]}")
        if frame.xcart.shape != ref_xcart.shape:
            raise ValueError(f"Frame {iframe} xcart shape mismatch: expected {ref_xcart.shape}, got {frame.xcart.shape}")
        if frame.forces.shape != ref_xcart.shape:
            raise ValueError(f"Frame {iframe} forces shape mismatch: expected {ref_xcart.shape}, got {frame.forces.shape}")

        frame = _align_frame_to_reference(frame, ref_xred)

        strain_tensor = _engineering_strain(frame.rprimd, inv_ref_rprimd)
        ref_xcart_in_frame = ref_xred @ frame.rprimd.T
        displacement[iframe] = frame.xcart - ref_xcart_in_frame
        strain[iframe] = _strain_tensor_to_voigt(strain_tensor)
        du_delta[iframe] = _compute_du_delta(displacement[iframe], strain_tensor)
        ucvol[iframe] = abs(float(np.linalg.det(frame.rprimd)))
        sqomega[iframe] = ucvol[iframe] ** (4.0 / 3.0) / natom ** (1.0 / 3.0)

        model_energy, model_forces, model_stress = _evaluate_fixed_model(fixed_model, frame)
        if model_forces.shape != ref_xcart.shape:
            raise ValueError(
                f"Fixed model forces shape mismatch for frame {iframe}: expected {ref_xcart.shape}, got {model_forces.shape}"
            )
        energy_diff[iframe] = frame.energy - model_energy
        force_diff[iframe] = frame.forces - model_forces
        stress_diff[iframe] = frame.stress - model_stress

    return TrainingDataset(
        displacement=displacement,
        du_delta=du_delta,
        strain=strain,
        ucvol=ucvol,
        sqomega=sqomega,
        energy_diff=energy_diff,
        force_diff=force_diff,
        stress_diff=stress_diff,
    )


def _align_frame_to_reference(frame: TrainingFrame, ref_xred: np.ndarray) -> TrainingFrame:
    """Return a frame reordered to the reference atom order using PBC xred matching."""
    cost = _fractional_distance_matrix(ref_xred, frame.xred)
    if np.array_equal(np.argmin(cost, axis=1), np.arange(ref_xred.shape[0])) and np.all(np.min(cost, axis=1) < 1e-10):
        return frame
    try:
        from scipy.optimize import linear_sum_assignment

        row, col = linear_sum_assignment(cost)
        order = col[np.argsort(row)]
    except Exception:
        order = _greedy_assignment(cost)
    return TrainingFrame(
        rprimd=frame.rprimd,
        xred=frame.xred[order],
        xcart=frame.xcart[order],
        energy=frame.energy,
        forces=frame.forces[order],
        stress=frame.stress,
    )


def _fractional_distance_matrix(ref_xred: np.ndarray, frame_xred: np.ndarray) -> np.ndarray:
    diff = ref_xred[:, None, :] - frame_xred[None, :, :]
    diff -= np.round(diff)
    return np.linalg.norm(diff, axis=2)


def _greedy_assignment(cost: np.ndarray) -> np.ndarray:
    remaining = set(range(cost.shape[1]))
    order = np.empty(cost.shape[0], dtype=int)
    for irow in range(cost.shape[0]):
        best = min(remaining, key=lambda icol: cost[irow, icol])
        order[irow] = best
        remaining.remove(best)
    return order


def solve_weighted_least_squares(
    features: FitFeatureMatrices,
    dataset,
    config: Optional[PythonFitConfig] = None,
    weights=None,
    regularization: Optional[float] = None,
) -> LinearFitResult:
    """Solve MULTIBINIT-style weighted normal equations with optional ridge."""
    cfg = config or PythonFitConfig(ncell=(1, 1, 1))
    weights_array = _weights(features.energy.shape[0], weights)
    ridge = cfg.regularization if regularization is None else float(regularization)
    if not np.isfinite(ridge):
        raise ValueError("regularization must be finite")
    if ridge < 0.0:
        raise ValueError("regularization must be non-negative")

    normal, rhs = _normal_equations(features, dataset, cfg, weights_array)
    if ridge:
        normal = normal + ridge * np.eye(normal.shape[0])
    try:
        coefficients = np.linalg.solve(normal, rhs)
        info = 0
    except np.linalg.LinAlgError:
        coefficients, *_ = np.linalg.lstsq(normal, rhs, rcond=None)
        info = 1
    if not np.isfinite(coefficients).all() or np.any(np.abs(coefficients) > 1.0e10):
        coefficients = np.zeros(normal.shape[0], dtype=float)
        info = 2

    goal = compute_goal_function(coefficients, features, dataset, weights_array)
    residual_norm = _weighted_residual_norm(coefficients, features, dataset, weights_array, cfg)
    rank = int(np.linalg.matrix_rank(normal))
    condition = float(np.linalg.cond(normal)) if normal.size else 0.0
    return LinearFitResult(
        coefficients=coefficients,
        diagnostics=FitDiagnostics(
            goal=goal,
            residual_norm=residual_norm,
            matrix_rank=rank,
            condition_number=condition,
            regularization=ridge,
            info=info,
        ),
    )


def compute_goal_function(coefficients, features: FitFeatureMatrices, dataset, weights=None) -> GoalFunctionComponents:
    """Compute MULTIBINIT ``computeGF`` components for predictions and residuals."""
    coeffs = np.asarray(coefficients, dtype=float)
    ntime = features.energy.shape[0]
    natom = features.forces.shape[1]
    ncoeff = features.energy.shape[1]
    if coeffs.shape != (ncoeff,):
        raise ValueError(f"coefficients must have shape ({ncoeff},), got {coeffs.shape}")
    weights_array = _weights(ntime, weights)
    energy_diff, force_diff, stress_diff, sqomega = _dataset_arrays(dataset, ntime, natom)
    energy_pred = features.energy @ coeffs
    force_pred = np.tensordot(features.forces, coeffs, axes=([3], [0]))
    stress_pred = np.tensordot(features.stress, coeffs, axes=([2], [0]))

    energy_resid = energy_diff - energy_pred
    force_resid = force_diff - force_pred
    stress_resid = stress_diff - stress_pred

    force_raw = float(np.sum(force_resid**2 * weights_array[:, None, None]))
    stress_raw = float(np.sum(stress_resid**2 * sqomega[:, None] * weights_array[:, None]))
    energy_raw = float(np.sum(energy_resid**2 / np.sqrt(sqomega) * weights_array))
    force = force_raw / (3 * natom * ntime)
    stress = stress_raw / (6 * ntime)
    energy = energy_raw / ntime
    return GoalFunctionComponents(force_stress=force + stress, force=force, stress=stress, energy=energy)


def load_xml_basis(filename) -> list[XmlBasisFunction]:
    """Load MULTIBINIT coefficient XML as immutable basis functions."""
    from pymultibinit.pyeffpot.xml_parser import read_coefficient_xml

    return [
        XmlBasisFunction(
            number=int(coeff.number),
            value=float(coeff.value),
            text=coeff.text,
            terms=tuple(_basis_term(term) for term in coeff.terms),
        )
        for coeff in read_coefficient_xml(str(filename))
    ]


def basis_to_coefficients(basis, fitted_values=None):
    """Convert XML basis functions back to parser coefficients."""
    from pymultibinit.pyeffpot.xml_parser import PolynomialCoefficient, PolynomialTerm

    basis_list = list(basis)
    values = _basis_values(basis_list, fitted_values)
    coefficients = []
    for item, value in zip(basis_list, values):
        coeff = PolynomialCoefficient(number=int(item.number), value=float(value), text=item.text)
        for term in item.terms:
            poly_term = PolynomialTerm(weight=float(term["weight"]))
            poly_term.displacements.extend(_dict_list(term["displacements"]))
            poly_term.strains.extend(_dict_list(term["strains"]))
            coeff.terms.append(poly_term)
        coefficients.append(coeff)
    return coefficients


def write_fitted_xml(filename, basis, fitted_values=None):
    """Write XML basis functions with optional fitted coefficient values."""
    from pymultibinit.pyeffpot.xml_parser import write_coefficient_xml

    coefficients = basis_to_coefficients(basis, fitted_values=fitted_values)
    write_coefficient_xml(str(filename), coefficients)
    return Path(filename)


_DIR_TO_INT = {"x": 0, "y": 1, "z": 2}
_INT_TO_DIR = {0: "x", 1: "y", 2: "z"}


def write_basis_netcdf(filename, basis, fitted_values=None) -> Path:
    from netCDF4 import Dataset

    basis_list = list(basis)
    values = _basis_values(basis_list, fitted_values)
    n = len(basis_list)
    if n == 0:
        raise ValueError("Cannot write empty basis")

    max_terms = max(1, max(len(item.terms) for item in basis_list))
    max_disps = 1
    max_strains = 1
    for item in basis_list:
        for t in item.terms:
            max_disps = max(max_disps, len(t["displacements"]))
            max_strains = max(max_strains, len(t.get("strains", ())))

    with Dataset(str(filename), "w", format="NETCDF4") as nc:
        nc.createDimension("ncoeff", n)
        nc.createDimension("max_terms", max_terms)
        nc.createDimension("max_disps", max_disps)
        nc.createDimension("max_strains", max_strains)
        nc.createDimension("three", 3)

        v_value = nc.createVariable("value", "f8", ("ncoeff",))
        v_nterms = nc.createVariable("nterms", "i4", ("ncoeff",))
        v_weight = nc.createVariable("weight", "f8", ("ncoeff", "max_terms"))
        v_ndisps = nc.createVariable("ndisps", "i4", ("ncoeff", "max_terms"))
        v_nstrains = nc.createVariable("nstrains", "i4", ("ncoeff", "max_terms"))
        v_atom_a = nc.createVariable("atom_a", "i4", ("ncoeff", "max_terms", "max_disps"))
        v_atom_b = nc.createVariable("atom_b", "i4", ("ncoeff", "max_terms", "max_disps"))
        v_dir = nc.createVariable("direction", "i4", ("ncoeff", "max_terms", "max_disps"))
        v_power = nc.createVariable("power", "i4", ("ncoeff", "max_terms", "max_disps"))
        v_cell_a = nc.createVariable("cell_a", "i4", ("ncoeff", "max_terms", "max_disps", "three"))
        v_cell_b = nc.createVariable("cell_b", "i4", ("ncoeff", "max_terms", "max_disps", "three"))
        v_svoigt = nc.createVariable("strain_voigt", "i4", ("ncoeff", "max_terms", "max_strains"))
        v_spower = nc.createVariable("strain_power", "i4", ("ncoeff", "max_terms", "max_strains"))

        v_value[:] = values
        for i, item in enumerate(basis_list):
            v_nterms[i] = len(item.terms)
            for j, term in enumerate(item.terms):
                v_weight[i, j] = float(term["weight"])
                disps = term["displacements"]
                v_ndisps[i, j] = len(disps)
                for k, d in enumerate(disps):
                    v_atom_a[i, j, k] = d["atom_a"]
                    v_atom_b[i, j, k] = d["atom_b"]
                    v_dir[i, j, k] = _DIR_TO_INT.get(d["direction"], 0)
                    v_power[i, j, k] = d["power"]
                    ca = d.get("cell_a", (0, 0, 0))
                    cb = d.get("cell_b", (0, 0, 0))
                    v_cell_a[i, j, k, :] = ca
                    v_cell_b[i, j, k, :] = cb
                strains = term.get("strains", ())
                v_nstrains[i, j] = len(strains)
                for k, s in enumerate(strains):
                    v_svoigt[i, j, k] = s.get("voigt", s.get("voigt_index", 0))
                    v_spower[i, j, k] = s.get("power", 1)

    return Path(filename)


def read_basis_netcdf(filename) -> list[XmlBasisFunction]:
    from netCDF4 import Dataset

    with Dataset(str(filename), "r") as nc:
        n = nc.dimensions["ncoeff"].size
        values = nc.variables["value"][:]
        nterms = nc.variables["nterms"][:]
        weight = nc.variables["weight"][:]
        ndisps = nc.variables["ndisps"][:]
        nstrains = nc.variables["nstrains"][:]
        atom_a = nc.variables["atom_a"][:]
        atom_b = nc.variables["atom_b"][:]
        direction = nc.variables["direction"][:]
        power = nc.variables["power"][:]
        cell_a = nc.variables["cell_a"][:]
        cell_b = nc.variables["cell_b"][:]
        svoigt = nc.variables["strain_voigt"][:]
        spower = nc.variables["strain_power"][:]

    basis = []
    for i in range(n):
        terms = []
        for j in range(int(nterms[i])):
            disps = []
            for k in range(int(ndisps[i, j])):
                disps.append({
                    "atom_a": int(atom_a[i, j, k]),
                    "atom_b": int(atom_b[i, j, k]),
                    "direction": _INT_TO_DIR[int(direction[i, j, k])],
                    "power": int(power[i, j, k]),
                    "cell_a": tuple(int(x) for x in cell_a[i, j, k, :]),
                    "cell_b": tuple(int(x) for x in cell_b[i, j, k, :]),
                })
            strains = []
            for k in range(int(nstrains[i, j])):
                strains.append({
                    "voigt": int(svoigt[i, j, k]),
                    "power": int(spower[i, j, k]),
                })
            terms.append({
                "weight": float(weight[i, j]),
                "displacements": tuple(disps),
                "strains": tuple(strains),
            })
        basis.append(XmlBasisFunction(
            number=i + 1,
            value=float(values[i]),
            text="",
            terms=tuple(terms),
        ))
    return basis


def write_fitted(filename, basis, fitted_values=None) -> Path:
    """Auto-select NetCDF (.nc) or XML (.xml) based on extension."""
    filename = Path(filename)
    if filename.suffix == ".nc":
        return _write_fitted_netcdf(filename, basis, fitted_values=fitted_values)
    return write_fitted_xml(filename, basis, fitted_values=fitted_values)


def _write_fitted_netcdf(filename, basis, fitted_values=None) -> Path:
    basis_list = list(basis)
    values = _basis_values(basis_list, fitted_values)
    nonzero = [i for i, v in enumerate(values) if abs(float(v)) > 1e-15]
    if not nonzero:
        raise ValueError("No nonzero coefficients to write")
    selected_basis = [basis_list[i] for i in nonzero]
    selected_values = [values[i] for i in nonzero]
    for new_num, (item, orig_idx) in enumerate(zip(selected_basis, nonzero), start=1):
        orig_text = item.text or ""
        object.__setattr__(item, "number", new_num)
        object.__setattr__(item, "text", f"{orig_text}  ; orig_idx={orig_idx}")
    return write_basis_netcdf(filename, selected_basis, fitted_values=selected_values)


def load_basis(filename) -> list[XmlBasisFunction]:
    """Auto-select NetCDF (.nc) or XML (.xml) based on extension."""
    filename = Path(filename)
    if filename.suffix == ".nc":
        return read_basis_netcdf(filename)
    return load_xml_basis(filename)


def with_fortran_text_labels(basis, symbols) -> list[XmlBasisFunction]:
    """Return basis functions with MULTIBINIT/Fortran-style text labels.

    The XML parser/evaluator use the structured ``term`` contents; the ``text``
    attribute is primarily a human-readable MULTIBINIT convention. Fortran uses
    labels such as ``(Ti_z-O1_z)^4`` rather than the internal Python shorthand
    ``u2_1_4^4``.
    """
    labels = _fortran_atom_labels(symbols)
    return [
        XmlBasisFunction(
            number=item.number,
            value=item.value,
            text=_fortran_basis_text(item, labels),
            terms=item.terms,
        )
        for item in basis
    ]


def _fortran_atom_labels(symbols) -> list[str]:
    symbols = [str(symbol) for symbol in symbols]
    counts: dict[str, int] = {}
    for symbol in symbols:
        counts[symbol] = counts.get(symbol, 0) + 1
    seen: dict[str, int] = {}
    labels = []
    for symbol in symbols:
        if counts[symbol] == 1:
            labels.append(symbol)
        else:
            seen[symbol] = seen.get(symbol, 0) + 1
            labels.append(f"{symbol}{seen[symbol]}")
    return labels


def _fortran_basis_text(item: XmlBasisFunction, labels: Sequence[str]) -> str:
    if not item.terms:
        return item.text
    term = item.terms[0]
    parts = []
    directions = {"x": "x", "y": "y", "z": "z", 0: "x", 1: "y", 2: "z"}
    for disp in term.get("displacements", ()):  # type: ignore[union-attr]
        atom_a = int(disp["atom_a"])
        atom_b = int(disp["atom_b"])
        direction = directions.get(disp["direction"], str(disp["direction"]))
        power = int(disp.get("power", 1))
        parts.append(f"({labels[atom_a]}_{direction}-{labels[atom_b]}_{direction})^{power}")
    for strain in term.get("strains", ()):  # type: ignore[union-attr]
        power = int(strain.get("power", 1))
        parts.append(f"eta{int(strain['voigt'])}^{power}")
    return "".join(parts)


def evaluate_basis_features(basis, dataset, ncell, backend: str = "auto", memmap_dir: str | None = None) -> FitFeatureMatrices:
    """Evaluate XML-loaded basis functions into linear E/F/stress features.

    Parameters
    ----------
    basis : list[XmlBasisFunction]
        Basis functions to evaluate.
    dataset : TrainingDataset
        Training data with displacements, strains, etc.
    ncell : tuple[int, int, int]
        Supercell dimensions.
    backend : str
        ``"auto"`` (default) — use the vectorized NumPy evaluator.
        ``"numpy"`` — use the vectorized NumPy evaluator.
        ``"jax"`` — explicitly use the optional JAX evaluator.
        ``"legacy"`` — use the original per-time-frame loop.
    """
    basis_list = list(basis)
    ncell_tuple = _tuple3_int(ncell, "ncell")
    displacement, strain, du_delta, ucvol = _feature_dataset_arrays(dataset)
    ntime, natom_sc, _ = displacement.shape
    ncells = ncell_tuple[0] * ncell_tuple[1] * ncell_tuple[2]
    if natom_sc % ncells != 0:
        raise ValueError(f"natom_sc={natom_sc} is not divisible by ncell product {ncells}")
    natom_uc = natom_sc // ncells
    ncoeff = len(basis_list)

    if backend == "legacy":
        energy = np.zeros((ntime, ncoeff), dtype=float)
        forces = np.zeros((ntime, natom_sc, 3, ncoeff), dtype=float)
        stress = np.zeros((ntime, 6, ncoeff), dtype=float)

        for icoeff, coeff in enumerate(basis_list):
            for term in coeff.terms:
                _accumulate_basis_term(term, displacement, strain, energy[:, icoeff], forces[:, :, :, icoeff], stress[:, :, icoeff], ncell_tuple, natom_uc)

        # Stress correction from strain-dependence of displacements, then final scaling.
        stress -= np.einsum("tvna,tnac->tvc", du_delta, forces)
        axial = np.arange(3)
        shear = np.arange(3, 6)
        stress[:, axial, :] *= (1.0 + strain[:, axial, None]) / ucvol[:, None, None]
        stress[:, shear, :] *= (1.0 - strain[:, shear, None] ** 2) / ucvol[:, None, None]
        return FitFeatureMatrices(energy=energy, forces=forces, stress=stress)

    if backend == "auto":
        backend = "numpy"

    # Use the explicit JAX backend only when requested. Importing JAX can emit
    # CUDA plugin warnings or fail even when NumPy evaluation is available.
    if backend == "jax":
        try:
            from pymultibinit.features import evaluate_basis_features_jax as _eval_jax

            energy, forces, stress = _eval_jax(
                basis_list, displacement, strain, du_delta, ucvol, ncell_tuple, natom_uc,
            )
            return FitFeatureMatrices(
                energy=np.asarray(energy),
                forces=np.asarray(forces),
                stress=np.asarray(stress),
            )
        except ImportError:
            raise

    if backend == "memmap":
        from pymultibinit.features import evaluate_basis_features_memmap as _eval_memmap

        energy, forces, stress = _eval_memmap(
            basis_list, displacement, strain, du_delta, ucvol, ncell_tuple, natom_uc, output_dir=memmap_dir,
        )
        return FitFeatureMatrices(energy=energy, forces=forces, stress=stress)

    if backend != "numpy":
        raise ValueError("backend must be one of: auto, numpy, jax, legacy, memmap")

    from pymultibinit.features import evaluate_basis_features_vectorized as _eval_vec

    energy, forces, stress = _eval_vec(
        basis_list, displacement, strain, du_delta, ucvol, ncell_tuple, natom_uc,
    )
    return FitFeatureMatrices(energy=energy, forces=forces, stress=stress)


def fit_multibinit_model_python(
    ddb,
    hist,
    basis_xml,
    output_xml=None,
    config: Optional[PythonFitConfig] = None,
    fixed_model=None,
    weights=None,
    validation_hist=None,
) -> PythonFitResult:
    """Fit XML coefficient values from DDB, HIST, and XML basis without Fortran."""
    cfg = config or PythonFitConfig(ncell=(1, 1, 1))
    if not isinstance(cfg, PythonFitConfig):
        raise TypeError("config must be a PythonFitConfig instance")
    ddb_path = _existing_path(ddb, "DDB file")
    hist_path = _existing_path(hist, "HIST file")
    basis_path = _existing_path(basis_xml, "basis file")

    reference = _reference_frame_from_ddb(ddb_path, cfg.ncell)
    frames = read_hist_frames(hist_path)
    dataset = build_training_dataset(reference, frames, fixed_model=fixed_model)
    if str(basis_path).endswith(".nc"):
        basis = read_basis_netcdf(basis_path)
    else:
        basis = load_xml_basis(basis_path)
    if cfg.selection == "screened_greedy":
        solve = _fit_screened_greedy(basis, dataset, cfg, weights=weights)
        coefficients = solve.coefficients
        diagnostics = solve.diagnostics
        selection_steps = solve.steps
        output_path = None
        if output_xml is not None:
            output_path = write_fitted(Path(output_xml).resolve(), basis, coefficients)
        return PythonFitResult(
            coefficients=coefficients,
            diagnostics=diagnostics,
            output_xml=str(output_path) if output_path is not None else None,
            ncoeff=len(basis),
            nframes=len(frames),
            ddb=str(ddb_path),
            hist=str(hist_path),
            basis_xml=str(basis_path),
            selection_steps=selection_steps,
        )

    features = evaluate_basis_features(
        basis, dataset, cfg.ncell, backend=cfg.feature_backend, memmap_dir=cfg.feature_memmap_dir
    )
    validation_dataset = None
    validation_features = None
    if validation_hist is not None:
        validation_path = _existing_path(validation_hist, "validation HIST file")
        validation_frames = read_hist_frames(validation_path)
        validation_dataset = build_training_dataset(reference, validation_frames, fixed_model=fixed_model)
        validation_features = evaluate_basis_features(
            basis, validation_dataset, cfg.ncell, backend=cfg.feature_backend, memmap_dir=cfg.feature_memmap_dir
        )
    if cfg.selection == "greedy":
        solve = select_greedy_coefficients(
            features,
            dataset,
            cfg,
            weights=weights,
            validation_features=validation_features,
            validation_dataset=validation_dataset,
        )
        coefficients = solve.coefficients
        diagnostics = solve.diagnostics
        selection_steps = solve.steps
    elif cfg.selection == "lasso":
        solve = _fit_lasso(features, dataset, cfg, weights=weights)
        coefficients = solve.coefficients
        diagnostics = solve.diagnostics
        selection_steps = solve.steps
    else:
        solve = solve_weighted_least_squares(features, dataset, cfg, weights=weights)
        coefficients = solve.coefficients
        diagnostics = solve.diagnostics
        selection_steps = ()

    output_path = None
    if output_xml is not None:
        output_path = write_fitted_xml(Path(output_xml).resolve(), basis, coefficients)

    return PythonFitResult(
        coefficients=coefficients,
        diagnostics=diagnostics,
        output_xml=str(output_path) if output_path is not None else None,
        ncoeff=len(basis),
        nframes=len(frames),
        ddb=str(ddb_path),
        hist=str(hist_path),
        basis_xml=str(basis_path),
        selection_steps=selection_steps,
    )


def select_greedy_coefficients(
    features: FitFeatureMatrices,
    dataset,
    config: PythonFitConfig,
    banned=None,
    preselected=None,
    weights=None,
    validation_features=None,
    validation_dataset=None,
) -> GreedySelectionResult:
    """Select coefficients greedily by minimizing the configured residual norm."""
    if config.selection != "greedy":
        raise ValueError("config.selection must be 'greedy' for greedy selection")
    ncoeff_total = features.energy.shape[1]
    target_count = config.ncoeff or ncoeff_total
    if target_count > ncoeff_total:
        raise ValueError(f"ncoeff={target_count} exceeds available coefficients {ncoeff_total}")
    banned_set = set(banned or ())
    selected = tuple(sorted(set(preselected or ())))
    for index in banned_set | set(selected):
        if index < 0 or index >= ncoeff_total:
            raise ValueError(f"coefficient index out of range: {index}")
    if banned_set & set(selected):
        raise ValueError("banned and preselected coefficient sets overlap")
    if len(selected) > target_count:
        raise ValueError("preselected coefficient count exceeds requested ncoeff")
    if ncoeff_total - len(banned_set) < target_count:
        raise ValueError("Not enough selectable coefficients for requested ncoeff after applying bans")
    if (validation_features is None) != (validation_dataset is None):
        raise ValueError("validation_features and validation_dataset must be provided together")
    if ncoeff_total > 1000:
        return _select_greedy_coefficients_large(
            features,
            dataset,
            config,
            banned_set,
            selected,
            weights,
            validation_features=validation_features,
            validation_dataset=validation_dataset,
        )
    steps = []
    final_result = None

    if selected:
        final_result = _solve_selected_features(features, dataset, config, selected, weights)
        if final_result.diagnostics.info != 0:
            raise ValueError("Preselected coefficients produce a singular or invalid solve")

    while len(selected) < target_count:
        best = None
        skipped_singular = 0
        for candidate in range(ncoeff_total):
            if candidate in banned_set or candidate in selected:
                continue
            trial = selected + (candidate,)
            result = _solve_selected_features(features, dataset, config, trial, weights)
            if result.diagnostics.info != 0:
                skipped_singular += 1
                continue
            score = result.diagnostics.residual_norm
            if best is None or score < best[0] - 1e-15 or (abs(score - best[0]) <= 1e-15 and candidate < best[1]):
                best = (score, candidate, result)
        if best is None:
            raise ValueError("Unable to select requested ncoeff; remaining candidates are singular or unavailable")
        _, candidate, final_result = best
        selected = selected + (candidate,)
        step = {
            "step": len(selected),
            "selected": candidate,
            "score": float(final_result.diagnostics.residual_norm),
            "skipped_singular": skipped_singular,
            "train_rmse": _fit_rmse_components(final_result.coefficients, features, dataset, selected),
        }
        if validation_features is not None and validation_dataset is not None:
            step["validation_rmse"] = _fit_rmse_components(
                final_result.coefficients, validation_features, validation_dataset, selected
            )
        steps.append(step)

    if final_result is None:
        final_result = _solve_selected_features(features, dataset, config, (), weights)
    coefficients = np.zeros(ncoeff_total, dtype=float)
    if selected:
        coefficients[list(selected)] = final_result.coefficients
    return GreedySelectionResult(selected=selected, coefficients=coefficients, diagnostics=final_result.diagnostics, steps=tuple(steps))


def _fit_screened_greedy(basis, dataset, config: PythonFitConfig, weights=None) -> GreedySelectionResult:
    basis_list = list(basis)
    ncoeff_total = len(basis_list)
    target_count = config.ncoeff or ncoeff_total
    pool_size = min(config.candidate_pool_size or max(10 * target_count, 1000), ncoeff_total)
    scores = np.full(ncoeff_total, np.inf, dtype=float)
    chunk = config.feature_chunk_size
    screening_dataset = _screening_dataset(dataset, config.screening_frame_count)
    screening_weights = _screening_weights(weights, dataset.displacement.shape[0], screening_dataset.displacement.shape[0])
    eval_cfg = replace(config, selection="greedy", feature_backend="numpy")
    progress = os.environ.get("PYMULTIBINIT_SCREENED_PROGRESS", "").lower() in {"1", "true", "yes"}
    started = time.monotonic()

    for start in range(0, ncoeff_total, chunk):
        stop = min(start + chunk, ncoeff_total)
        chunk_started = time.monotonic()
        features = evaluate_basis_features(basis_list[start:stop], screening_dataset, config.ncell, backend="numpy")
        rhs, diagonal, chunk_target_norm = _greedy_rhs_diagonal_target(features, screening_dataset, eval_cfg, screening_weights)
        denom = diagonal + config.regularization
        valid = denom > 0.0
        local_scores = np.full(stop - start, np.inf, dtype=float)
        local_scores[valid] = chunk_target_norm - (rhs[valid] ** 2 / denom[valid])
        scores[start:stop] = local_scores
        if progress:
            elapsed = time.monotonic() - started
            chunk_elapsed = time.monotonic() - chunk_started
            print(
                f"screened_greedy screening {stop}/{ncoeff_total} "
                f"chunk_seconds={chunk_elapsed:.1f} elapsed_seconds={elapsed:.1f}",
                flush=True,
            )

    pool = tuple(int(index) for index in np.argsort(scores)[:pool_size])
    pool_basis = [basis_list[index] for index in pool]
    if progress:
        print(f"screened_greedy fitting pool_size={pool_size}", flush=True)
    pool_features = evaluate_basis_features(pool_basis, dataset, config.ncell, backend=config.feature_backend, memmap_dir=config.feature_memmap_dir)
    pool_cfg = replace(config, selection="greedy", candidate_pool_size=None)
    pool_result = select_greedy_coefficients(pool_features, dataset, pool_cfg, weights=weights)
    coefficients = np.zeros(ncoeff_total, dtype=float)
    selected_global = tuple(pool[index] for index in pool_result.selected)
    coefficients[list(selected_global)] = pool_result.coefficients[list(pool_result.selected)]
    steps = []
    for step in pool_result.steps:
        mapped = dict(step)
        mapped["selected"] = pool[int(step["selected"])]
        mapped["screened_pool_size"] = pool_size
        steps.append(mapped)
    return GreedySelectionResult(
        selected=selected_global,
        coefficients=coefficients,
        diagnostics=pool_result.diagnostics,
        steps=tuple(steps),
    )


def _screening_weights(weights, total_frames: int, screening_frames: int):
    if weights is None or screening_frames >= total_frames:
        return weights
    weights_array = _weights(total_frames, weights)
    indices = np.linspace(0, total_frames - 1, int(screening_frames), dtype=int)
    return weights_array[indices]


def _screening_dataset(dataset: TrainingDataset, nframes: Optional[int]) -> TrainingDataset:
    total = dataset.displacement.shape[0]
    if nframes is None or nframes >= total:
        return dataset
    indices = np.linspace(0, total - 1, int(nframes), dtype=int)
    return TrainingDataset(
        displacement=dataset.displacement[indices],
        du_delta=dataset.du_delta[indices],
        strain=dataset.strain[indices],
        ucvol=dataset.ucvol[indices],
        sqomega=dataset.sqomega[indices],
        energy_diff=dataset.energy_diff[indices],
        force_diff=dataset.force_diff[indices],
        stress_diff=dataset.stress_diff[indices],
    )


def _fit_lasso(features: FitFeatureMatrices, dataset, config: PythonFitConfig, weights=None):
    ntime, ncoeff = features.energy.shape
    natom = features.forces.shape[1]
    weights_array = _weights(ntime, weights)
    energy_diff, force_diff, stress_diff, sqomega = _dataset_arrays(dataset, ntime, natom)
    force_on, stress_on, energy_on = config.fit_on
    ff, sf, ef = config.fit_factors
    ridge = config.regularization

    f_flat = features.forces.reshape(ntime * natom * 3, ncoeff)
    s_flat = features.stress.reshape(ntime * 6, ncoeff)
    e_flat = features.energy

    fw2 = np.repeat(weights_array, natom * 3) ** 2 * ff / (3 * natom * ntime)
    sw2 = np.repeat(weights_array * sqomega, 6) ** 2 * sf / (6 * ntime)
    ew2 = (weights_array / np.sqrt(sqomega)) ** 2 * ef / ntime

    mats = []
    ys = []
    w2s = []
    if force_on:
        mats.append(f_flat); ys.append(force_diff.ravel()); w2s.append(fw2)
    if stress_on:
        mats.append(s_flat); ys.append(stress_diff.ravel()); w2s.append(sw2)
    if energy_on:
        mats.append(e_flat); ys.append(energy_diff); w2s.append(ew2)

    rhs = np.zeros(ncoeff, dtype=float)
    col_sq = np.full(ncoeff, ridge, dtype=float)
    _chunk = 256
    for mat, y, w2 in zip(mats, ys, w2s):
        rhs += mat.T @ (w2 * y)
        for s in range(0, ncoeff, _chunk):
            e = min(s + _chunk, ncoeff)
            cols = mat[:, s:e]
            col_sq[s:e] += np.sum(w2[:, None] * cols * cols, axis=0)

    def _ista(lam, max_iter=300):
        beta = np.zeros(ncoeff, dtype=float)
        z = beta.copy()
        step = 1.0 / (np.max(col_sq) * 2)
        thr = lam * step / 2
        tk = 1.0
        for _ in range(max_iter):
            grad = -rhs.copy()
            for mat, w2 in zip(mats, w2s):
                grad += mat.T @ (w2 * (mat @ z))
            nb = z - step * grad
            np.sign(nb, out=nb, where=(np.abs(nb) > thr))
            nb *= np.maximum(np.abs(nb) - thr, 0.0) / np.where(nb != 0, nb, 1.0)
            tk_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * tk * tk))
            z = nb + ((tk - 1.0) / tk_new) * (nb - beta)
            diff = np.max(np.abs(nb - beta))
            beta = nb
            tk = tk_new
            if diff < 1e-10:
                break
        return beta

    target_count = config.ncoeff or 40
    lam_max = np.max(np.abs(rhs))
    lam_lo, lam_hi = 0.0, lam_max * 1.05
    best_beta = np.zeros(ncoeff)
    best_diff = ncoeff + 1

    for _ in range(15):
        lam = 0.5 * (lam_lo + lam_hi)
        beta = _ista(lam)
        nnz = int(np.count_nonzero(beta))
        d = abs(nnz - target_count)
        if d < best_diff:
            best_diff = d
            best_beta = beta
        if nnz == target_count:
            break
        if nnz > target_count:
            lam_lo = lam
        else:
            lam_hi = lam

    selected = tuple(int(j) for j in np.argsort(np.abs(best_beta))[::-1][:target_count])
    if selected:
        sliced = FitFeatureMatrices(
            energy=features.energy[:, selected],
            forces=features.forces[:, :, :, selected],
            stress=features.stress[:, :, selected],
        )
        result = solve_weighted_least_squares(sliced, dataset, config, weights=weights)
        coefficients = np.zeros(ncoeff, dtype=float)
        coefficients[list(selected)] = result.coefficients
        diagnostics = result.diagnostics
    else:
        coefficients = np.zeros(ncoeff, dtype=float)
        diagnostics = FitDiagnostics(
            goal={"force": 0, "stress": 0, "energy": 0, "force_stress": 0},
            residual_norm=0.0, matrix_rank=0, condition_number=0.0,
            regularization=ridge, info=0,
        )

    return GreedySelectionResult(
        selected=selected,
        coefficients=coefficients,
        diagnostics=diagnostics,
        steps=tuple({"step": i + 1, "selected": j, "lasso": True}
                     for i, j in enumerate(selected[:20])),
    )


def _select_greedy_coefficients_large(
    features: FitFeatureMatrices,
    dataset,
    config: PythonFitConfig,
    banned_set: set[int],
    selected: tuple[int, ...],
    weights,
    validation_features=None,
    validation_dataset=None,
) -> GreedySelectionResult:
    """Greedy selector optimized for large candidate bases.

    The small-basis implementation solves every trial by rebuilding normal
    equations from sliced feature arrays. For tens of thousands of candidates,
    most work is repeated. This path precomputes candidate RHS/diagonal terms
    once and adds selected-candidate cross terms only as needed.
    """
    ncoeff_total = features.energy.shape[1]
    target_count = config.ncoeff or ncoeff_total
    rhs, diagonal, target_norm = _greedy_rhs_diagonal_target(features, dataset, config, weights)
    cross_cache: dict[int, np.ndarray] = {}
    steps = []
    final_result = None

    if selected:
        final_result = _solve_selected_features(features, dataset, config, selected, weights)
        if final_result.diagnostics.info != 0:
            raise ValueError("Preselected coefficients produce a singular or invalid solve")

    while len(selected) < target_count:
        selected_list = list(selected)
        selected_set = set(selected)
        selected_normal = _greedy_selected_normal(
            features, dataset, config, weights, selected_list, cross_cache, diagonal
        )
        selected_rhs = rhs[selected_list] if selected_list else np.zeros(0, dtype=float)
        selected_cross = np.column_stack(
            [_greedy_normal_column(features, dataset, config, weights, index, cross_cache) for index in selected_list]
        ) if selected_list else np.zeros((ncoeff_total, 0), dtype=float)

        best = None
        skipped_singular = 0
        for candidate in range(ncoeff_total):
            if candidate in banned_set or candidate in selected_set:
                continue
            score = _greedy_trial_score(
                candidate,
                selected_normal,
                selected_rhs,
                selected_cross,
                rhs,
                diagonal,
                target_norm,
                config.regularization,
            )
            if score is None:
                skipped_singular += 1
                continue
            if best is None or score < best[0] - 1e-15 or (abs(score - best[0]) <= 1e-15 and candidate < best[1]):
                best = (score, candidate)
        if best is None:
            raise ValueError("Unable to select requested ncoeff; remaining candidates are singular or unavailable")
        _, candidate = best
        selected = selected + (candidate,)
        final_result = _solve_selected_features(features, dataset, config, selected, weights)
        step = {
            "step": len(selected),
            "selected": candidate,
            "score": float(final_result.diagnostics.residual_norm),
            "skipped_singular": skipped_singular,
            "train_rmse": _fit_rmse_components(final_result.coefficients, features, dataset, selected),
        }
        if validation_features is not None and validation_dataset is not None:
            step["validation_rmse"] = _fit_rmse_components(
                final_result.coefficients, validation_features, validation_dataset, selected
            )
        steps.append(step)

    if final_result is None:
        final_result = _solve_selected_features(features, dataset, config, (), weights)
    coefficients = np.zeros(ncoeff_total, dtype=float)
    if selected:
        coefficients[list(selected)] = final_result.coefficients
    return GreedySelectionResult(selected=selected, coefficients=coefficients, diagnostics=final_result.diagnostics, steps=tuple(steps))


def _greedy_rhs_diagonal_target(features: FitFeatureMatrices, dataset, config: PythonFitConfig, weights):
    ntime, ncoeff = features.energy.shape
    natom = features.forces.shape[1]
    rhs = np.zeros(ncoeff, dtype=float)
    diagonal = np.zeros(ncoeff, dtype=float)
    target_norm = 0.0
    weights_array = _weights(ntime, weights)
    force_on, stress_on, energy_on = config.fit_on
    force_factor, stress_factor, energy_factor = config.fit_factors
    energy_diff, force_diff, stress_diff, sqomega = _dataset_arrays(dataset, ntime, natom)

    if force_on:
        design = features.forces.reshape(ntime * natom * 3, ncoeff)
        target = force_diff.reshape(ntime * natom * 3)
        row_weights = np.repeat(weights_array, natom * 3)
        factor = force_factor / (3 * natom * ntime)
        weighted_target = target * row_weights
        rhs += factor * (design.T @ weighted_target)
        diagonal += factor * np.einsum("ij,i,ij->j", design, row_weights, design, optimize=True)
        target_norm += factor * float(np.sum(target * weighted_target))

    if stress_on:
        design = features.stress.reshape(ntime * 6, ncoeff)
        target = stress_diff.reshape(ntime * 6)
        row_weights = np.repeat(weights_array * sqomega, 6)
        factor = stress_factor / (6 * ntime)
        weighted_target = target * row_weights
        rhs += factor * (design.T @ weighted_target)
        diagonal += factor * np.einsum("ij,i,ij->j", design, row_weights, design, optimize=True)
        target_norm += factor * float(np.sum(target * weighted_target))

    if energy_on:
        design = features.energy
        target = energy_diff
        row_weights = weights_array / np.sqrt(sqomega)
        factor = energy_factor / ntime
        weighted_target = target * row_weights
        rhs += factor * (design.T @ weighted_target)
        diagonal += factor * np.einsum("ij,i,ij->j", design, row_weights, design, optimize=True)
        target_norm += factor * float(np.sum(target * weighted_target))

    return rhs, diagonal, target_norm


def _greedy_normal_column(features, dataset, config, weights, index: int, cache: dict[int, np.ndarray]) -> np.ndarray:
    if index in cache:
        return cache[index]
    ntime, ncoeff = features.energy.shape
    natom = features.forces.shape[1]
    column = np.zeros(ncoeff, dtype=float)
    weights_array = _weights(ntime, weights)
    force_on, stress_on, energy_on = config.fit_on
    force_factor, stress_factor, energy_factor = config.fit_factors
    _, _, _, sqomega = _dataset_arrays(dataset, ntime, natom)

    if force_on:
        design = features.forces.reshape(ntime * natom * 3, ncoeff)
        row_weights = np.repeat(weights_array, natom * 3)
        factor = force_factor / (3 * natom * ntime)
        column += factor * (design.T @ (design[:, index] * row_weights))

    if stress_on:
        design = features.stress.reshape(ntime * 6, ncoeff)
        row_weights = np.repeat(weights_array * sqomega, 6)
        factor = stress_factor / (6 * ntime)
        column += factor * (design.T @ (design[:, index] * row_weights))

    if energy_on:
        design = features.energy
        row_weights = weights_array / np.sqrt(sqomega)
        factor = energy_factor / ntime
        column += factor * (design.T @ (design[:, index] * row_weights))

    cache[index] = column
    return column


def _greedy_selected_normal(features, dataset, config, weights, selected: list[int], cache, diagonal) -> np.ndarray:
    nselected = len(selected)
    normal = np.zeros((nselected, nselected), dtype=float)
    for j, index in enumerate(selected):
        column = _greedy_normal_column(features, dataset, config, weights, index, cache)
        normal[:, j] = column[selected]
    if nselected:
        normal[np.diag_indices(nselected)] = diagonal[selected]
    return normal


def _greedy_trial_score(
    candidate: int,
    selected_normal: np.ndarray,
    selected_rhs: np.ndarray,
    selected_cross: np.ndarray,
    rhs: np.ndarray,
    diagonal: np.ndarray,
    target_norm: float,
    regularization: float,
) -> Optional[float]:
    nselected = selected_normal.shape[0]
    if nselected == 0:
        normal_value = diagonal[candidate]
        solve_value = normal_value + regularization
        if solve_value <= 0.0 or not np.isfinite(solve_value):
            return None
        coeff = rhs[candidate] / solve_value
        score_sq = target_norm - 2.0 * rhs[candidate] * coeff + normal_value * coeff * coeff
        return float(np.sqrt(max(score_sq, 0.0))) if np.isfinite(score_sq) else None

    cross = selected_cross[candidate]
    normal = np.empty((nselected + 1, nselected + 1), dtype=float)
    normal[:nselected, :nselected] = selected_normal
    normal[:nselected, nselected] = cross
    normal[nselected, :nselected] = cross
    normal[nselected, nselected] = diagonal[candidate]
    solve_normal = normal.copy()
    if regularization:
        solve_normal[np.diag_indices(nselected + 1)] += regularization
    trial_rhs = np.empty(nselected + 1, dtype=float)
    trial_rhs[:nselected] = selected_rhs
    trial_rhs[nselected] = rhs[candidate]
    try:
        coeffs = np.linalg.solve(solve_normal, trial_rhs)
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite(coeffs).all():
        return None
    score_sq = target_norm - 2.0 * float(trial_rhs @ coeffs) + float(coeffs @ normal @ coeffs)
    if not np.isfinite(score_sq):
        return None
    return float(np.sqrt(max(score_sq, 0.0)))


def _fit_rmse_components(coefficients, features: FitFeatureMatrices, dataset, selected) -> dict[str, float]:
    selected = tuple(selected)
    coeffs = np.asarray(coefficients, dtype=float)
    if len(selected) != coeffs.shape[0]:
        raise ValueError("selected coefficient count must match coefficient vector length")
    if selected:
        energy_fit = features.energy[:, selected] @ coeffs
        force_fit = np.einsum("tnac,c->tna", features.forces[:, :, :, selected], coeffs)
        stress_fit = np.einsum("tvc,c->tv", features.stress[:, :, selected], coeffs)
    else:
        energy_fit = np.zeros_like(dataset.energy_diff)
        force_fit = np.zeros_like(dataset.force_diff)
        stress_fit = np.zeros_like(dataset.stress_diff)
    energy_residual = dataset.energy_diff - energy_fit
    relative_energy_residual = energy_residual - energy_residual[0]
    force_residual = dataset.force_diff - force_fit
    stress_residual = dataset.stress_diff - stress_fit
    return {
        "relative_energy_ha": float(np.sqrt(np.mean(relative_energy_residual**2))),
        "forces_ha_bohr": float(np.sqrt(np.mean(force_residual**2))),
        "stress_ha_bohr3": float(np.sqrt(np.mean(stress_residual**2))),
    }


def normalize_pair_key(key: PairKey) -> tuple[PairKey, int]:
    """Normalize inverse pair orientation and return the sign change."""
    direct = PairKey(int(key.direction), int(key.atom_a), int(key.atom_b), tuple(int(v) for v in key.cell_b))
    inverse = PairKey(direct.direction, direct.atom_b, direct.atom_a, tuple(-v for v in direct.cell_b))
    if inverse < direct:
        return inverse, -1
    return direct, 1


def normalize_monomial_key(monomial: MonomialKey) -> tuple[MonomialKey, int]:
    powers: dict[PairKey, int] = {}
    sign = 1
    for factor, power in monomial.factors:
        normalized, factor_sign = normalize_pair_key(factor)
        power = int(power)
        powers[normalized] = powers.get(normalized, 0) + power
        if factor_sign == -1 and power % 2:
            sign *= -1
    return MonomialKey(tuple(sorted((factor, power) for factor, power in powers.items() if power))), sign


def build_factor_action_map(factors, symrel, atom_mappings=None, rprimd=None) -> list[dict[PairKey, tuple[PairKey, int]]]:
    """Build deterministic symmetry actions for displacement factor keys."""
    factor_list = list(factors)
    mappings = _normalize_atom_mappings(atom_mappings, symrel)
    if len(mappings) != len(symrel):
        raise ValueError("atom_mappings must have one entry per symmetry operation")
    actions = []
    for sym, mapping in zip(symrel, mappings):
        rotation = np.asarray(sym, dtype=int)
        if rotation.shape != (3, 3):
            raise ValueError(f"symrel operations must have shape (3, 3), got {rotation.shape}")
        direction_rotation = _cartesian_direction_rotation(rotation, rprimd)
        action = {}
        for factor in factor_list:
            direction, direction_sign = _transform_direction(factor.direction, direction_rotation)
            atom_a, cell_a_shift = _map_atom(factor.atom_a, mapping)
            atom_b, cell_b_shift = _map_atom(factor.atom_b, mapping)
            cell = rotation @ np.array(factor.cell_b, dtype=int) + np.array(cell_b_shift, dtype=int) - np.array(cell_a_shift, dtype=int)
            normalized, orientation_sign = normalize_pair_key(PairKey(direction, atom_a, atom_b, tuple(int(v) for v in cell)))
            action[factor] = (normalized, direction_sign * orientation_sign)
        actions.append(action)
    return actions


def canonicalize_monomial_orbit(monomial: MonomialKey, actions) -> tuple[MonomialKey, int]:
    """Return the lexicographically smallest representative across actions."""
    best = normalize_monomial_key(monomial)
    for action in actions:
        transformed = []
        sign = 1
        for factor, power in monomial.factors:
            mapped, factor_sign = action[factor]
            transformed.append((mapped, power))
            if factor_sign == -1 and int(power) % 2:
                sign *= -1
        normalized, normalized_sign = normalize_monomial_key(MonomialKey(tuple(transformed)))
        candidate = (normalized, sign * normalized_sign)
        if candidate[0] < best[0] or (candidate[0] == best[0] and candidate[1] < best[1]):
            best = candidate
    return best


def generate_displacement_basis(
    xcart,
    cutoff: float,
    power_range=(3, 4),
    symrel=None,
    ncell=(1, 1, 1),
    rprimd=None,
    atom_mappings=None,
    include_strain_coupling: bool = False,
    strain_voigts: Sequence[int] = (1, 2, 3, 4, 5, 6),
) -> list[XmlBasisFunction]:
    """Generate deterministic displacement and optional strain-coupling XML basis functions."""
    positions = np.asarray(xcart, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("xcart must have shape (natom, 3)")
    if cutoff <= 0.0 or not np.isfinite(cutoff):
        raise ValueError("cutoff must be finite and positive")
    ncell_tuple = _tuple3_int(ncell, "ncell")
    if any(value <= 0 for value in ncell_tuple):
        raise ValueError("ncell values must be positive")
    min_power, max_power = _tuple2_int(power_range, "power_range")
    if min_power <= 0 or min_power > max_power:
        raise ValueError("power_range must be positive and ordered")
    strain_indices = tuple(int(value) for value in strain_voigts)
    if any(value < 1 or value > 6 for value in strain_indices):
        raise ValueError("strain_voigts values must be in 1..6")

    factors = _generate_pair_factors(positions, cutoff, ncell_tuple, rprimd)
    symrel_ops = [np.eye(3, dtype=int)] if symrel is None else symrel
    actions = build_factor_action_map(factors, symrel_ops, atom_mappings=atom_mappings, rprimd=rprimd)
    accepted_by_power: dict[int, dict[MonomialKey, dict[MonomialKey, int]]] = {}
    for total_power in range(min_power, max_power + 1):
        accepted_by_power[total_power] = _accepted_displacement_orbits(factors, actions, total_power)

    basis = []
    for total_power in range(min_power, max_power + 1):
        for representative, orbit in sorted(accepted_by_power[total_power].items(), key=lambda item: _monomial_sort_key(item[0])):
            basis.append(_basis_from_orbit(len(basis) + 1, representative, orbit))

    if include_strain_coupling:
        for total_power in range(min_power, max_power + 1):
            displacement_power = total_power - 1
            if displacement_power < 1:
                continue
            accepted = accepted_by_power.get(displacement_power)
            if accepted is None:
                accepted = _accepted_displacement_orbits(factors, actions, displacement_power)
                accepted_by_power[displacement_power] = accepted
            for representative, orbit in sorted(accepted.items(), key=lambda item: _monomial_sort_key(item[0])):
                for voigt in strain_indices:
                    basis.append(_basis_from_orbit(len(basis) + 1, representative, orbit, strains=({"power": 1, "voigt": voigt},)))

    return basis


def displacement_pair_diagnostics(
    xcart,
    cutoff: float,
    ncell=(1, 1, 1),
    symrel=None,
    rprimd=None,
    atom_mappings=None,
) -> dict[str, object]:
    """Summarize pair-factor coverage and symmetry closure for basis generation."""
    positions = np.asarray(xcart, dtype=float)
    ncell_tuple = _tuple3_int(ncell, "ncell")
    factors = _generate_pair_factors(positions, float(cutoff), ncell_tuple, rprimd)
    symrel_ops = [np.eye(3, dtype=int)] if symrel is None else symrel
    actions = build_factor_action_map(factors, symrel_ops, atom_mappings=atom_mappings, rprimd=rprimd)
    factor_set = set(factors)
    missing = []
    for isym, action in enumerate(actions):
        for factor, (mapped, sign) in action.items():
            if mapped not in factor_set:
                missing.append(
                    {
                        "symmetry_index": isym,
                        "factor": _pair_key_dict(factor),
                        "mapped": _pair_key_dict(mapped),
                        "sign": int(sign),
                    }
                )
    distances = [_pair_distance(factor, positions, rprimd) for factor in factors]
    return {
        "ncell": ncell_tuple,
        "cutoff": float(cutoff),
        "n_factors": len(factors),
        "n_symmetry_operations": len(symrel_ops),
        "symmetry_closed": not missing,
        "missing_mapped_factors_count": len(missing),
        "missing_mapped_factors": missing[:50],
        "max_pair_distance": float(max(distances)) if distances else 0.0,
        "min_pair_distance": float(min(distances)) if distances else 0.0,
        "cell_offsets": sorted({factor.cell_b for factor in factors}),
    }


def _pair_key_dict(factor: PairKey) -> dict[str, object]:
    return {
        "direction": int(factor.direction),
        "atom_a": int(factor.atom_a),
        "atom_b": int(factor.atom_b),
        "cell_b": tuple(int(item) for item in factor.cell_b),
    }


def _pair_distance(factor: PairKey, positions: np.ndarray, rprimd) -> float:
    lattice = np.eye(3) if rprimd is None else np.asarray(rprimd, dtype=float)
    vector = positions[factor.atom_b] + np.array(factor.cell_b, dtype=float) @ lattice - positions[factor.atom_a]
    return float(np.linalg.norm(vector))


def _accepted_displacement_orbits(factors, actions, total_power: int) -> dict[MonomialKey, dict[MonomialKey, int]]:
    accepted: dict[MonomialKey, dict[MonomialKey, int]] = {}
    if total_power < 1:
        return accepted
    for combo in combinations_with_replacement(factors, total_power):
        monomial = _monomial_from_factors(combo)
        if not _compatible_monomial(monomial):
            continue
        if not _symmetry_allowed_monomial(monomial, actions):
            continue
        representative, sign = canonicalize_monomial_orbit(monomial, actions)
        accepted.setdefault(representative, {})
        for image, image_sign in _monomial_orbit(monomial, actions):
            accepted[representative].setdefault(image, image_sign)
    return accepted


@dataclass
class MultibinitTrainingResult:
    """Summary of a MULTIBINIT binary training run."""

    model_config: Optional[str]
    output_dir: str
    log_file: str
    stderr_file: str
    metadata_file: str
    returncode: int
    command: list[str]
    ddb: str
    hist: str
    config: Optional[str]
    artifacts: dict[str, str]

    def to_dict(self) -> dict:
        return asdict(self)


def train_multibinit_model(
    ddb,
    hist,
    config=None,
    output_dir="multibinit_training",
    executable=None,
    extra_args: Optional[Sequence[str]] = None,
    timeout: Optional[float] = None,
    env: Optional[Mapping[str, str]] = None,
) -> MultibinitTrainingResult:
    """Build a MULTIBINIT model by invoking the ``multibinit`` executable.

    The Python layer stages paths, calls the binary without shell mode, captures
    logs, and records deterministic metadata. The actual model-building
    semantics remain owned by the MULTIBINIT input file and executable.
    """
    ddb_path = _existing_path(ddb, "DDB file")
    hist_path = _existing_path(hist, "HIST file")
    config_path = _existing_path(config, "configuration file") if config is not None else None
    exe = _resolve_executable(executable)
    outdir = Path(output_dir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    command = [str(exe)]
    if config_path is not None:
        command.append(str(config_path))
    command.extend(str(arg) for arg in (extra_args or ()))

    run_env = os.environ.copy()
    run_env.update(
        {
            "PYMULTIBINIT_DDB": str(ddb_path),
            "PYMULTIBINIT_HIST": str(hist_path),
            "PYMULTIBINIT_OUTPUT_DIR": str(outdir),
        }
    )
    if config_path is not None:
        run_env["PYMULTIBINIT_CONFIG"] = str(config_path)
    if env:
        run_env.update({str(key): str(value) for key, value in env.items()})

    stdout_file = outdir / "multibinit.stdout.log"
    stderr_file = outdir / "multibinit.stderr.log"
    metadata_file = outdir / "pymultibinit_training_result.json"

    completed = subprocess.run(
        command,
        cwd=outdir,
        env=run_env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    stdout_file.write_text(completed.stdout, encoding="utf-8")
    stderr_file.write_text(completed.stderr, encoding="utf-8")

    artifacts = _discover_artifacts(outdir)
    result = MultibinitTrainingResult(
        model_config=_choose_model_config(artifacts),
        output_dir=str(outdir),
        log_file=str(stdout_file),
        stderr_file=str(stderr_file),
        metadata_file=str(metadata_file),
        returncode=completed.returncode,
        command=command,
        ddb=str(ddb_path),
        hist=str(hist_path),
        config=str(config_path) if config_path is not None else None,
        artifacts=artifacts,
    )
    metadata_file.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    if completed.returncode != 0:
        raise RuntimeError(
            f"MULTIBINIT training failed with exit code {completed.returncode}. "
            f"See {stdout_file} and {stderr_file}."
        )

    return result


def _existing_path(path, label: str) -> Path:
    candidate = Path(path).resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return candidate


def _tuple3_int(value, name: str) -> tuple[int, int, int]:
    values = tuple(value)
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly 3 values")
    return tuple(int(item) for item in values)  # type: ignore[return-value]


def _tuple2_int(value, name: str) -> tuple[int, int]:
    values = tuple(value)
    if len(values) != 2:
        raise ValueError(f"{name} must contain exactly 2 values")
    return tuple(int(item) for item in values)  # type: ignore[return-value]


def _tuple3_bool(value, name: str) -> tuple[bool, bool, bool]:
    values = tuple(value)
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly 3 values")
    return tuple(bool(item) for item in values)  # type: ignore[return-value]


def _tuple3_float(value, name: str) -> tuple[float, float, float]:
    values = tuple(value)
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly 3 values")
    return tuple(float(item) for item in values)  # type: ignore[return-value]


def _array_shape(value, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    return array


def _required_var(nc, name: str, ndim: int) -> np.ndarray:
    if name not in nc.variables:
        raise ValueError(f"HIST file is missing required variable '{name}'")
    data = np.array(nc.variables[name].data, dtype=float)
    if data.ndim != ndim:
        raise ValueError(f"HIST variable '{name}' must have {ndim} dimensions, got {data.ndim}")
    return data


def _reference_lattice_positions(reference) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rprimd = np.asarray(getattr(reference, "rprimd"), dtype=float)
    xcart = np.asarray(getattr(reference, "xcart"), dtype=float)
    if hasattr(reference, "xred"):
        xred = np.asarray(getattr(reference, "xred"), dtype=float)
    else:
        xred = xcart @ np.linalg.inv(rprimd.T)
    if rprimd.shape != (3, 3):
        raise ValueError(f"Reference rprimd must have shape (3, 3), got {rprimd.shape}")
    if xcart.ndim != 2 or xcart.shape[1] != 3:
        raise ValueError(f"Reference xcart must have shape (natom, 3), got {xcart.shape}")
    if xred.shape != xcart.shape:
        raise ValueError(f"Reference xred must have shape {xcart.shape}, got {xred.shape}")
    return rprimd, xcart, xred


def _engineering_strain(rprimd: np.ndarray, inv_ref_rprimd: np.ndarray) -> np.ndarray:
    """Compute engineering (Biot-like) strain matching MULTIBINIT convention.

    Fortran reference: m_fit_data.F90 strain_get call + Voigt extraction.
    Formula: eta = h_def @ h_ref^{-T} - I  (same as strain_get in m_strain.F90).
    """
    deformation = rprimd @ inv_ref_rprimd
    return deformation - np.eye(3)


def _strain_tensor_to_voigt(strain: np.ndarray) -> np.ndarray:
    return np.array(
        [
            strain[0, 0],
            strain[1, 1],
            strain[2, 2],
            strain[1, 2] + strain[2, 1],
            strain[2, 0] + strain[0, 2],
            strain[0, 1] + strain[1, 0],
        ],
        dtype=float,
    )


def _compute_du_delta(displacement: np.ndarray, strain_tensor: np.ndarray) -> np.ndarray:
    strain_inv = np.linalg.inv(np.eye(3) + strain_tensor)
    pairs = ((0, 0), (1, 1), (2, 2), (2, 1), (2, 0), (1, 0))
    du_delta = np.zeros((6, displacement.shape[0], 3), dtype=float)
    for iatom, disp in enumerate(displacement):
        strain_inv_u = strain_inv @ disp
        for ivoigt, (alpha, beta) in enumerate(pairs):
            for mu in range(3):
                if alpha == mu:
                    du_delta[ivoigt, iatom, mu] += 0.5 * strain_inv_u[beta]
                if beta == mu:
                    du_delta[ivoigt, iatom, mu] += 0.5 * strain_inv_u[alpha]
    return du_delta


def _stress_to_voigt(stress) -> np.ndarray:
    array = np.asarray(stress, dtype=float)
    if array.shape == (6,):
        return array
    if array.shape == (3, 3):
        return np.array([array[0, 0], array[1, 1], array[2, 2], array[1, 2], array[0, 2], array[0, 1]], dtype=float)
    raise ValueError(f"stress must have shape (6,) or (3, 3), got {array.shape}")


def _evaluate_fixed_model(fixed_model, frame: TrainingFrame) -> tuple[float, np.ndarray, np.ndarray]:
    if fixed_model is None:
        return 0.0, np.zeros_like(frame.forces), np.zeros(6, dtype=float)
    if not hasattr(fixed_model, "evaluate"):
        raise ValueError("fixed_model must provide evaluate(xcart, rprimd)")
    energy, forces, stress = fixed_model.evaluate(frame.xcart, frame.rprimd)
    return float(energy), np.asarray(forces, dtype=float), _stress_to_voigt(stress)


def _weights(ntime: int, weights) -> np.ndarray:
    if weights is None:
        return np.ones(ntime, dtype=float)
    array = np.asarray(weights, dtype=float)
    if array.shape != (ntime,):
        raise ValueError(f"weights must have shape ({ntime},), got {array.shape}")
    if not np.isfinite(array).all() or np.any(array < 0.0):
        raise ValueError("weights must be finite and non-negative")
    return array


def _dataset_arrays(dataset, ntime: int, natom: int):
    energy_diff = np.asarray(dataset.energy_diff, dtype=float)
    force_diff = np.asarray(dataset.force_diff, dtype=float)
    stress_diff = np.asarray(dataset.stress_diff, dtype=float)
    sqomega = np.asarray(dataset.sqomega, dtype=float)
    if energy_diff.shape != (ntime,):
        raise ValueError(f"energy_diff must have shape ({ntime},), got {energy_diff.shape}")
    if force_diff.shape != (ntime, natom, 3):
        raise ValueError(f"force_diff must have shape ({ntime}, {natom}, 3), got {force_diff.shape}")
    if stress_diff.shape != (ntime, 6):
        raise ValueError(f"stress_diff must have shape ({ntime}, 6), got {stress_diff.shape}")
    if sqomega.shape != (ntime,):
        raise ValueError(f"sqomega must have shape ({ntime},), got {sqomega.shape}")
    if not (np.isfinite(energy_diff).all() and np.isfinite(force_diff).all() and np.isfinite(stress_diff).all()):
        raise ValueError("dataset target arrays must be finite")
    if not np.isfinite(sqomega).all() or np.any(sqomega <= 0.0):
        raise ValueError("sqomega values must be finite and positive")
    return energy_diff, force_diff, stress_diff, sqomega


def _normal_equations(features: FitFeatureMatrices, dataset, config: PythonFitConfig, weights: np.ndarray):
    ntime, ncoeff = features.energy.shape
    natom = features.forces.shape[1]
    normal = np.zeros((ncoeff, ncoeff), dtype=float)
    rhs = np.zeros(ncoeff, dtype=float)
    force_on, stress_on, energy_on = config.fit_on
    force_factor, stress_factor, energy_factor = config.fit_factors
    energy_diff, force_diff, stress_diff, sqomega = _dataset_arrays(dataset, ntime, natom)

    if force_on:
        design = features.forces.reshape(ntime * natom * 3, ncoeff)
        target = force_diff.reshape(ntime * natom * 3)
        row_weights = np.repeat(weights, natom * 3)
        factor = force_factor / (3 * natom * ntime)
        normal += factor * (design.T @ (design * row_weights[:, None]))
        rhs += factor * (design.T @ (target * row_weights))

    if stress_on:
        design = features.stress.reshape(ntime * 6, ncoeff)
        target = stress_diff.reshape(ntime * 6)
        row_weights = np.repeat(weights * sqomega, 6)
        factor = stress_factor / (6 * ntime)
        normal += factor * (design.T @ (design * row_weights[:, None]))
        rhs += factor * (design.T @ (target * row_weights))

    if energy_on:
        design = features.energy
        target = energy_diff
        row_weights = weights / np.sqrt(sqomega)
        factor = energy_factor / ntime
        normal += factor * (design.T @ (design * row_weights[:, None]))
        rhs += factor * (design.T @ (target * row_weights))

    return normal, rhs


def _weighted_residual_norm(coefficients, features: FitFeatureMatrices, dataset, weights, config: PythonFitConfig) -> float:
    goal = compute_goal_function(coefficients, features, dataset, weights)
    force_on, stress_on, energy_on = config.fit_on
    total = 0.0
    if force_on:
        total += goal.force * config.fit_factors[0]
    if stress_on:
        total += goal.stress * config.fit_factors[1]
    if energy_on:
        total += goal.energy * config.fit_factors[2]
    return float(np.sqrt(total))


def _basis_term(term) -> Mapping[str, object]:
    return {
        "weight": float(term.weight),
        "displacements": tuple(_basis_displacement(disp) for disp in term.displacements),
        "strains": tuple(_basis_strain(strain) for strain in term.strains),
    }


def _basis_displacement(disp: Mapping[str, object]) -> Mapping[str, object]:
    return {
        "atom_a": int(disp["atom_a"]),
        "atom_b": int(disp["atom_b"]),
        "direction": str(disp["direction"]),
        "power": int(disp["power"]),
        "cell_a": tuple(int(value) for value in disp["cell_a"]),
        "cell_b": tuple(int(value) for value in disp["cell_b"]),
    }


def _basis_strain(strain: Mapping[str, object]) -> Mapping[str, int]:
    return {"power": int(strain["power"]), "voigt": int(strain["voigt"])}


def _basis_values(basis: Sequence[XmlBasisFunction], fitted_values) -> np.ndarray:
    if fitted_values is None:
        return np.array([item.value for item in basis], dtype=float)
    values = np.asarray(fitted_values, dtype=float)
    if values.shape != (len(basis),):
        raise ValueError(f"fitted_values must have shape ({len(basis)},), got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("fitted_values must be finite")
    return values


def _dict_list(items) -> list[dict]:
    converted = []
    for item in items:
        value = dict(item)
        if "cell_a" in value:
            value["cell_a"] = list(value["cell_a"])
        if "cell_b" in value:
            value["cell_b"] = list(value["cell_b"])
        converted.append(value)
    return converted


def _feature_dataset_arrays(dataset):
    displacement = np.asarray(dataset.displacement, dtype=float)
    strain = np.asarray(dataset.strain, dtype=float)
    du_delta = np.asarray(dataset.du_delta, dtype=float)
    ucvol = np.asarray(dataset.ucvol, dtype=float)
    if displacement.ndim != 3 or displacement.shape[2] != 3:
        raise ValueError("displacement must have shape (time, natom, 3)")
    ntime, natom, _ = displacement.shape
    if strain.shape != (ntime, 6):
        raise ValueError(f"strain must have shape ({ntime}, 6), got {strain.shape}")
    if du_delta.shape != (ntime, 6, natom, 3):
        raise ValueError(f"du_delta must have shape ({ntime}, 6, {natom}, 3), got {du_delta.shape}")
    if ucvol.shape != (ntime,):
        raise ValueError(f"ucvol must have shape ({ntime},), got {ucvol.shape}")
    if not (np.isfinite(displacement).all() and np.isfinite(strain).all() and np.isfinite(du_delta).all()):
        raise ValueError("feature input arrays must be finite")
    if not np.isfinite(ucvol).all() or np.any(ucvol <= 0.0):
        raise ValueError("ucvol values must be finite and positive")
    return displacement, strain, du_delta, ucvol


def _accumulate_basis_term(term, displacement, strain, energy_out, forces_out, stress_out, ncell, natom_uc):
    origins = _origin_cells(ncell)
    compiled_displacements = tuple(_compile_displacement_factor(disp, origins, ncell, natom_uc) for disp in term["displacements"])
    compiled_strains = tuple(term["strains"])
    weight = float(term["weight"])
    ntime = displacement.shape[0]

    for itime in range(ntime):
        strain_values = strain[itime]
        strain_multiplier = _strain_product(compiled_strains, strain_values)
        disp_values = [_disp_diff(displacement[itime], disp) for disp in compiled_displacements]
        product = np.ones(len(origins), dtype=float) * strain_multiplier
        for values, disp in zip(disp_values, compiled_displacements):
            product *= values ** disp["power"]
        term_energy_by_origin = weight * product
        energy_out[itime] += float(term_energy_by_origin.sum())

        for idisp, disp in enumerate(compiled_displacements):
            power = disp["power"]
            if power == 0:
                continue
            deriv = np.ones(len(origins), dtype=float) * weight * strain_multiplier * power
            for jdisp, other in enumerate(compiled_displacements):
                exponent = other["power"] - 1 if jdisp == idisp else other["power"]
                deriv *= disp_values[jdisp] ** exponent
            np.add.at(forces_out[itime, :, disp["direction"]], disp["idx_a"], -deriv)
            np.add.at(forces_out[itime, :, disp["direction"]], disp["idx_b"], deriv)

        for istrain, strain_factor in enumerate(compiled_strains):
            power = int(strain_factor["power"])
            if power == 0:
                continue
            voigt = int(strain_factor["voigt"]) - 1
            deriv = np.ones(len(origins), dtype=float) * weight * power
            for jstrain, other in enumerate(compiled_strains):
                exponent = int(other["power"]) - 1 if jstrain == istrain else int(other["power"])
                deriv *= strain_values[int(other["voigt"]) - 1] ** exponent
            for values, disp in zip(disp_values, compiled_displacements):
                deriv *= values ** disp["power"]
            stress_out[itime, voigt] += float(deriv.sum())


def _origin_cells(ncell: tuple[int, int, int]) -> np.ndarray:
    nx, ny, nz = ncell
    return np.array([(ix, iy, iz) for ix in range(nx) for iy in range(ny) for iz in range(nz)], dtype=int)


def _compile_displacement_factor(disp, origins: np.ndarray, ncell: tuple[int, int, int], natom_uc: int) -> Mapping[str, object]:
    direction_map = {"x": 0, "y": 1, "z": 2, 0: 0, 1: 1, 2: 2}
    direction = disp["direction"]
    if direction not in direction_map:
        raise ValueError(f"Unsupported displacement direction: {direction}")
    atom_a = int(disp["atom_a"])
    atom_b = int(disp["atom_b"])
    if not (0 <= atom_a < natom_uc and 0 <= atom_b < natom_uc):
        raise ValueError(f"Displacement atom index out of unit-cell range 0..{natom_uc - 1}")
    return {
        "idx_a": _supercell_indices(atom_a, tuple(disp["cell_a"]), origins, ncell, natom_uc),
        "idx_b": _supercell_indices(atom_b, tuple(disp["cell_b"]), origins, ncell, natom_uc),
        "direction": direction_map[direction],
        "power": int(disp["power"]),
    }


def _supercell_indices(atom: int, cell_shift, origins: np.ndarray, ncell: tuple[int, int, int], natom_uc: int) -> np.ndarray:
    nx, ny, nz = ncell
    shift = np.array(cell_shift, dtype=int)
    cells = (origins + shift) % np.array([nx, ny, nz], dtype=int)
    return atom + natom_uc * (cells[:, 2] + nz * (cells[:, 1] + ny * cells[:, 0]))


def _strain_product(strains, strain_values: np.ndarray) -> float:
    value = 1.0
    for strain in strains:
        value *= strain_values[int(strain["voigt"]) - 1] ** int(strain["power"])
    return float(value)


def _disp_diff(displacement: np.ndarray, disp: Mapping[str, object]) -> np.ndarray:
    return displacement[disp["idx_a"], disp["direction"]] - displacement[disp["idx_b"], disp["direction"]]


def _solve_selected_features(features: FitFeatureMatrices, dataset, config: PythonFitConfig, selected, weights):
    selected = tuple(selected)
    if not selected:
        empty = FitFeatureMatrices(
            energy=np.zeros((features.energy.shape[0], 0), dtype=float),
            forces=np.zeros((features.forces.shape[0], features.forces.shape[1], 3, 0), dtype=float),
            stress=np.zeros((features.stress.shape[0], 6, 0), dtype=float),
        )
        goal = compute_goal_function(np.zeros(0), empty, dataset, weights)
        return LinearFitResult(
            coefficients=np.zeros(0, dtype=float),
            diagnostics=FitDiagnostics(goal=goal, residual_norm=_weighted_residual_norm(np.zeros(0), empty, dataset, _weights(features.energy.shape[0], weights), config), matrix_rank=0, condition_number=0.0, regularization=config.regularization, info=0),
        )
    sliced = FitFeatureMatrices(
        energy=features.energy[:, selected],
        forces=features.forces[:, :, :, selected],
        stress=features.stress[:, :, selected],
    )
    return solve_weighted_least_squares(sliced, dataset, config, weights=weights)


def _transform_direction(direction: int, rotation: np.ndarray) -> tuple[int, int]:
    column = rotation[:, int(direction)]
    nonzero = np.flatnonzero(column)
    if len(nonzero) != 1 or abs(int(column[nonzero[0]])) != 1:
        raise ValueError("Only axis-permutation symmetry operations are supported for factor actions")
    return int(nonzero[0]), int(column[nonzero[0]])


def _generate_pair_factors(positions: np.ndarray, cutoff: float, ncell: tuple[int, int, int], rprimd) -> list[PairKey]:
    factors = []
    natom = positions.shape[0]
    lattice = np.eye(3) if rprimd is None else np.asarray(rprimd, dtype=float)
    if lattice.shape != (3, 3):
        raise ValueError(f"rprimd must have shape (3, 3), got {lattice.shape}")
    ranges = [range(-(value - 1), value) for value in ncell]
    for atom_a in range(natom):
        for atom_b in range(natom):
            for ix in ranges[0]:
                for iy in ranges[1]:
                    for iz in ranges[2]:
                        cell = (ix, iy, iz)
                        if atom_a == atom_b and cell == (0, 0, 0):
                            continue
                        vector = positions[atom_b] + np.array(cell, dtype=float) @ lattice - positions[atom_a]
                        if np.linalg.norm(vector) <= cutoff:
                            for direction in range(3):
                                key, _ = normalize_pair_key(PairKey(direction=direction, atom_a=atom_a, atom_b=atom_b, cell_b=cell))
                                factors.append(key)
    return sorted(set(factors))


def generate_fortran_pair_list(
    xcart,
    xred,
    cutoff: float,
    fit_iatom: int,
    symrel,
    ncell=(1, 1, 1),
    rprimd=None,
    tnons=None,
) -> FortranPairList:
    """Generate the Fortran-style anchored pair list for one 0-based central atom.

    This mirrors the pair-list stage of ``polynomial_coeff_getList`` closely
    enough to compare hard diagnostics against MULTIBINIT logs. It intentionally
    preserves the anchored first atom instead of applying the global pair
    inversion used by the older Python generator.
    """
    positions = np.asarray(xcart, dtype=float)
    reduced = np.asarray(xred, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("xcart must have shape (natom, 3)")
    if reduced.shape != positions.shape:
        raise ValueError("xred must have the same shape as xcart")
    if cutoff <= 0.0 or not np.isfinite(cutoff):
        raise ValueError("cutoff must be finite and positive")
    natom = positions.shape[0]
    if fit_iatom < 0 or fit_iatom >= natom:
        raise ValueError("fit_iatom must be a 0-based atom index")
    ncell_tuple = _tuple3_int(ncell, "ncell")
    if any(value <= 0 for value in ncell_tuple):
        raise ValueError("ncell values must be positive")
    lattice = np.eye(3) if rprimd is None else np.asarray(rprimd, dtype=float)
    if lattice.shape != (3, 3):
        raise ValueError(f"rprimd must have shape (3, 3), got {lattice.shape}")
    symrel_ops = np.asarray([np.eye(3, dtype=int)] if symrel is None else symrel, dtype=int)
    if symrel_ops.ndim != 3 or symrel_ops.shape[1:] != (3, 3):
        raise ValueError("symrel must have shape (nsym, 3, 3)")
    translations = np.zeros((len(symrel_ops), 3), dtype=float) if tnons is None else np.asarray(tnons, dtype=float)
    if translations.shape != (len(symrel_ops), 3):
        raise ValueError("tnons must have shape (nsym, 3)")

    from pymultibinit.pyeffpot.symmetry import build_atom_mapping, find_equivalent_atom, get_reciprocal_symmetry

    atom_mapping = build_atom_mapping(reduced, symrel_ops, translations)
    symrec = get_reciprocal_symmetry(symrel_ops)
    cells = _fortran_cells(ncell_tuple)
    cell_to_index = {cell: index for index, cell in enumerate(cells)}
    dist = _fortran_pair_distances(positions, lattice, cells)
    distances = np.linalg.norm(dist, axis=0)
    range_ifc = np.array([np.linalg.norm(lattice[ii]) * ncell_tuple[ii] / 2.0 for ii in range(3)])

    blocks = np.ones((3, natom, 3, natom, len(cells)), dtype=bool)
    raw_entries: list[tuple[tuple[PairKey, int] | None, ...]] = []
    for atom_a in range(natom):
        for irpt, atom_b in _fortran_sorted_neighbors(distances, atom_a):
            possible = bool(distances[atom_a, atom_b, irpt] <= cutoff)
            if possible and np.any(np.abs(dist[:, atom_a, atom_b, irpt]) - range_ifc > 1e-10):
                possible = False
            if not possible:
                blocks[:, atom_a, :, atom_b, irpt] = False
                if irpt == 0:
                    blocks[:, atom_b, :, atom_a, irpt] = False
                continue
            if not blocks[:, atom_a, :, atom_b, irpt].any():
                continue
            dist_orig = distances[atom_a, atom_b, irpt]
            for mu in range(3):
                for nu in range(3):
                    if mu != nu:
                        blocks[mu, atom_a, nu, atom_b, irpt] = False
                        blocks[nu, atom_a, mu, atom_b, irpt] = False
                        continue
                    if irpt == 0 and atom_a == atom_b:
                        blocks[mu, atom_a, nu, atom_b, irpt] = False
                        blocks[nu, atom_b, mu, atom_a, irpt] = False
                        continue
                    if blocks[mu, atom_a, nu, atom_b, irpt]:
                        images = []
                        for isym, sym in enumerate(symrel_ops):
                            image = _fortran_transform_pair(
                                PairKey(mu, atom_a, atom_b, cells[irpt]),
                                isym,
                                sym,
                                symrec[isym],
                                atom_mapping,
                                reduced,
                                translations[isym],
                                lattice,
                                cells,
                                cell_to_index,
                                dist,
                                dist_orig,
                                find_equivalent_atom,
                            )
                            images.append(image)
                        raw_entries.append(tuple(images))
                    blocks[mu, atom_a, nu, atom_b, irpt] = False

    valid = [all(image is not None for image in entry) for entry in raw_entries]
    raw_sym_indices = [[-1] * len(symrel_ops) for _ in raw_entries]
    for index, entry in enumerate(raw_entries):
        if not valid[index]:
            continue
        for isym, image in enumerate(entry):
            factor, _ = image  # type: ignore[misc]
            mapped = _fortran_find_entry(raw_entries, factor, 0)
            if mapped < 0:
                valid[index] = False
            raw_sym_indices[index][isym] = mapped

    raw_to_compact = {index: sum(valid[:index]) for index, keep in enumerate(valid) if keep}
    entries = [entry for index, entry in enumerate(raw_entries) if valid[index]]
    sym_indices = [tuple(raw_to_compact[raw_sym_indices[index][isym]] for isym in range(len(symrel_ops))) for index, keep in enumerate(valid) if keep]
    sym_signs = [tuple(int(entry[isym][1]) for isym in range(len(symrel_ops))) for entry in entries]  # type: ignore[index]
    factors = [entry[0][0] for entry in entries]  # type: ignore[index]

    keep = [factor.atom_a == fit_iatom for factor in factors]
    for index, factor in enumerate(factors):
        if not keep[index] or factor.atom_a == factor.atom_b:
            continue
        for isym, image in enumerate(entries[index]):
            image_factor = image[0]  # type: ignore[index]
            opposite_direction = _fortran_opposite_direction_index(cells, image_factor.direction)
            for jsym in range(len(symrel_ops)):
                mapped = _fortran_find_entry(entries, PairKey(opposite_direction, image_factor.atom_b, image_factor.atom_a, image_factor.cell_b), jsym)
                if mapped > index:
                    keep[mapped] = False

    irreducible_keep = keep.copy()
    for index, image_indices in enumerate(sym_indices):
        if not keep[index]:
            continue
        for mapped in image_indices:
            if mapped > index:
                irreducible_keep[mapped] = False

    final_source_indices = [index for index, selected in enumerate(irreducible_keep) if selected]
    final_source_indices.extend(index for index, selected in enumerate(keep) if selected and not irreducible_keep[index])
    compact_to_final = {source_index: final_index for final_index, source_index in enumerate(final_source_indices)}
    final_factors = tuple(factors[index] for index in final_source_indices)
    final_sym_factors = tuple(tuple(None if image is None else image[0] for image in entries[index]) for index in final_source_indices)
    final_sym_indices = tuple(
        tuple(compact_to_final.get(mapped, -1) for mapped in sym_indices[index])
        for index in final_source_indices
    )
    final_sym_signs = tuple(tuple(sym_signs[index]) for index in final_source_indices)
    irreducible = tuple(factors[index] for index, selected in enumerate(irreducible_keep) if selected)
    return FortranPairList(
        ncoeff_sym=len(irreducible),
        ncoeff_total=len(final_factors),
        irreducible=irreducible,
        factors=final_factors,
        sym_factors=final_sym_factors,
        sym_indices=final_sym_indices,
        sym_signs=final_sym_signs,
        cells=tuple(cells),
        dist=dist,
    )


def count_fortran_irreducible_pair_combinations(
    pair_list: FortranPairList,
    power_range=(3, 4),
    ncell=(1, 1, 1),
    rprimd=None,
) -> int:
    """Count Fortran-compatible combinations of irreducible pair factors."""
    min_power, max_power = _tuple2_int(power_range, "power_range")
    if min_power <= 0 or min_power > max_power:
        raise ValueError("power_range must be positive and ordered")
    ncell_tuple = _tuple3_int(ncell, "ncell")
    lattice = np.eye(3) if rprimd is None else np.asarray(rprimd, dtype=float)
    if lattice.shape != (3, 3):
        raise ValueError(f"rprimd must have shape (3, 3), got {lattice.shape}")
    cell_to_index = {cell: index for index, cell in enumerate(pair_list.cells)}
    compatible = _fortran_pair_compatibility(pair_list.irreducible, pair_list.dist, cell_to_index, lattice, ncell_tuple)
    count = 0
    for total_power in range(min_power, max_power + 1):
        for combo in combinations_with_replacement(range(pair_list.ncoeff_sym), total_power):
            if all(compatible[left, right] for left in combo for right in combo):
                count += 1
    return count


def count_fortran_displacement_coefficients(
    pair_list: FortranPairList,
    power_range=(3, 4),
    ncell=(1, 1, 1),
    rprimd=None,
) -> int:
    """Count Fortran-style displacement coefficients after symmetry reduction."""
    return sum(
        1
        for combination in generate_fortran_displacement_combination_keys(pair_list, power_range=power_range, ncell=ncell, rprimd=rprimd)
        if _fortran_combination_has_nonzero_terms(combination, pair_list)
    )


def generate_fortran_displacement_combination_keys(
    pair_list: FortranPairList,
    power_range=(3, 4),
    ncell=(1, 1, 1),
    rprimd=None,
) -> tuple[tuple[int, ...], ...]:
    """Return reduced Fortran-style displacement combination keys.

    Keys are tuples of indices into ``pair_list.factors``. The first
    ``pair_list.ncoeff_sym`` factors are the irreducible pair representatives,
    matching MULTIBINIT's ``list_symcoeff`` ordering.
    """
    min_power, max_power = _tuple2_int(power_range, "power_range")
    if min_power <= 0 or min_power > max_power:
        raise ValueError("power_range must be positive and ordered")
    ncell_tuple = _tuple3_int(ncell, "ncell")
    lattice = np.eye(3) if rprimd is None else np.asarray(rprimd, dtype=float)
    if lattice.shape != (3, 3):
        raise ValueError(f"rprimd must have shape (3, 3), got {lattice.shape}")
    cell_to_index = {cell: index for index, cell in enumerate(pair_list.cells)}
    compatible_irreducible = _fortran_pair_compatibility(pair_list.irreducible, pair_list.dist, cell_to_index, lattice, ncell_tuple)
    compatible_all = _fortran_pair_compatibility(pair_list.factors, pair_list.dist, cell_to_index, lattice, ncell_tuple)
    accepted: dict[tuple[int, ...], tuple[int, ...]] = {}
    for total_power in range(min_power, max_power + 1):
        for combo in combinations_with_replacement(range(pair_list.ncoeff_sym), total_power):
            if not all(compatible_irreducible[left, right] for left in combo for right in combo):
                continue
            if len(set(combo)) == 1 and not _fortran_onebody_right_order(pair_list.irreducible[combo[0]]):
                continue
            for expanded in _fortran_expand_combination(combo, pair_list, compatible_all):
                candidate = _fortran_reduction_key(expanded, max_power)
                if not _fortran_combination_seen(candidate, pair_list, accepted):
                    accepted[candidate] = tuple(sorted(expanded))
    return tuple(sorted(accepted.values()))


def _monomial_from_factors(factors) -> MonomialKey:
    powers: dict[PairKey, int] = {}
    sign = 1
    for factor in factors:
        normalized, factor_sign = normalize_pair_key(factor)
        powers[normalized] = powers.get(normalized, 0) + 1
        if factor_sign < 0:
            sign *= -1
    monomial = MonomialKey(tuple(sorted(powers.items())))
    if sign < 0:
        # A negative representative is equivalent to a term weight sign, not key identity.
        return monomial
    return monomial


def _fortran_cells(ncell: tuple[int, int, int]) -> list[tuple[int, int, int]]:
    limits = [value // 2 + 1 + (1 if value % 2 else 0) for value in ncell]
    cells = [(0, 0, 0)]
    for r1 in range(limits[0], -limits[0] - 1, -1):
        for r2 in range(limits[1], -limits[1] - 1, -1):
            for r3 in range(limits[2], -limits[2] - 1, -1):
                cell = (r1, r2, r3)
                if cell != (0, 0, 0):
                    cells.append(cell)
    return cells


def _fortran_pair_distances(positions: np.ndarray, lattice: np.ndarray, cells: Sequence[tuple[int, int, int]]) -> np.ndarray:
    natom = positions.shape[0]
    dist = np.zeros((3, natom, natom, len(cells)), dtype=float)
    for atom_a in range(natom):
        for atom_b in range(natom):
            for irpt, cell in enumerate(cells):
                dist[:, atom_a, atom_b, irpt] = positions[atom_b] + np.array(cell, dtype=float) @ lattice - positions[atom_a]
    return dist


def _fortran_sorted_neighbors(distances: np.ndarray, atom_a: int) -> list[tuple[int, int]]:
    natom = distances.shape[1]
    nrpt = distances.shape[2]
    entries = []
    for irpt in range(nrpt):
        for atom_b in range(natom):
            entries.append((float(distances[atom_a, atom_b, irpt]), irpt * natom + atom_b, irpt, atom_b))
    entries.sort(key=lambda item: (item[0], item[1]))
    return [(irpt, atom_b) for _, __, irpt, atom_b in entries]


def _fortran_transform_pair(
    factor: PairKey,
    isym: int,
    symrel: np.ndarray,
    symrec: np.ndarray,
    atom_mapping: np.ndarray,
    xred: np.ndarray,
    tnons: np.ndarray,
    rprimd: np.ndarray,
    cells: Sequence[tuple[int, int, int]],
    cell_to_index: Mapping[tuple[int, int, int], int],
    dist: np.ndarray,
    dist_orig: float,
    find_equivalent_atom,
) -> tuple[PairKey, int] | None:
    direction_rotation = _cartesian_direction_rotation(symrel, rprimd)
    direction, direction_sign = _transform_direction(factor.direction, direction_rotation)
    atom_a = int(atom_mapping[3, isym, factor.atom_a])
    shift_atom_a = atom_mapping[:3, isym, factor.atom_a]
    transformed = symrec.T @ (xred[factor.atom_b] + np.array(factor.cell_b, dtype=float) - tnons)
    try:
        atom_b, translation = find_equivalent_atom(transformed, xred)
    except ValueError:
        return None
    cell = tuple(int(value) for value in (translation - shift_atom_a))
    irpt = cell_to_index.get(cell)
    if irpt is None or (irpt == 0 and atom_a == atom_b):
        return None
    dist_sym = float(np.linalg.norm(dist[:, atom_a, atom_b, irpt]))
    if abs(dist_orig - dist_sym) > 1e-8:
        return None
    return PairKey(direction, atom_a, int(atom_b), cell), direction_sign


def _fortran_find_entry(entries, factor: PairKey, isym: int) -> int:
    for index, entry in enumerate(entries):
        if entry is None or entry[isym] is None:
            continue
        candidate = entry[isym][0]
        if candidate == factor:
            return index
    return -1


def _fortran_opposite_direction_index(cells: Sequence[tuple[int, int, int]], direction: int) -> int:
    target = tuple(-value for value in cells[int(direction)])
    for index, cell in enumerate(cells):
        if cell == target:
            return index
    return int(direction)


def _fortran_pair_compatibility(
    factors: Sequence[PairKey],
    dist: np.ndarray,
    cell_to_index: Mapping[tuple[int, int, int], int],
    rprimd: np.ndarray,
    ncell: tuple[int, int, int],
) -> np.ndarray:
    nfactor = len(factors)
    compatible = np.ones((nfactor, nfactor), dtype=bool)
    bounds = np.array([np.sum(rprimd[ii]) * ncell[ii] for ii in range(3)], dtype=float)
    for left, factor_left in enumerate(factors):
        left_irpt = cell_to_index[factor_left.cell_b]
        left_dist = dist[:, factor_left.atom_a, factor_left.atom_b, left_irpt]
        for right, factor_right in enumerate(factors):
            if factor_left.atom_a != factor_right.atom_a:
                compatible[left, right] = False
                continue
            right_irpt = cell_to_index[factor_right.cell_b]
            right_dist = dist[:, factor_left.atom_a, factor_right.atom_b, right_irpt]
            if np.any(np.abs(left_dist - right_dist) >= bounds):
                compatible[left, right] = False
    return compatible


def _fortran_onebody_right_order(factor: PairKey) -> bool:
    # MULTIBINIT's local `is_right_order` reads list_symcoeff(1) and (2),
    # i.e. direction and first atom, despite naming them ia/ib.
    return factor.direction >= factor.atom_a


def _fortran_expand_combination(combo: tuple[int, ...], pair_list: FortranPairList, compatible: np.ndarray):
    """Expand irreducible combination over symmetry images."""
    image_choices = [_unique_in_order(image for image in pair_list.sym_indices[factor_index] if image >= 0) for factor_index in combo]
    expanded: list[int] = []
    seen: set[tuple[int, ...]] = set()

    def visit(position: int):
        if position == len(combo):
            candidate = tuple(sorted(expanded))
            if candidate not in seen:
                seen.add(candidate)
                yield candidate
            return
        for mapped in image_choices[position]:
            if all(compatible[previous, mapped] for previous in expanded):
                expanded.append(mapped)
                yield from visit(position + 1)
                expanded.pop()

    yield from visit(0)


def _unique_in_order(values) -> tuple[int, ...]:
    seen = set()
    ordered = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return tuple(ordered)


def _fortran_reduction_key(combo: tuple[int, ...], power: int) -> tuple[int, ...]:
    """Fortran-style fixed-length key padded with zeros to ``power``."""
    values = [index + 1 for index in combo]
    values.extend([0] * (power - len(values)))
    return _fortran_sort_combination(values)


def _fortran_sort_combination(values: Sequence[int]) -> tuple[int, ...]:
    """Mirror Fortran ``sort_combination``: sort positive values in-place without crossing zeros."""
    result = list(values)
    n = len(result)
    for j in range(2, n + 1):
        k = j
        swapped = True
        while k >= 2 and swapped:
            swapped = False
            if result[k - 2] > result[k - 1] and result[k - 1] > 0:
                result[k - 2], result[k - 1] = result[k - 1], result[k - 2]
                k -= 1
                swapped = True
    return tuple(result)


def _fortran_combination_seen(combo: tuple[int, ...], pair_list: FortranPairList, accepted: Mapping[tuple[int, ...], tuple[int, ...]]) -> bool:
    if combo in accepted:
        return True
    nsym = len(pair_list.sym_indices[0]) if pair_list.sym_indices else 0
    for isym in range(1, nsym):
        mapped = []
        for factor_index in combo:
            if factor_index == 0:
                mapped.append(0)
            else:
                image = pair_list.sym_indices[factor_index - 1][isym]
                mapped.append(image + 1 if image >= 0 else 0)
        if _fortran_sort_combination(mapped) in accepted:
            return True
    return False


def _fortran_combination_has_nonzero_terms(combo: tuple[int, ...], pair_list: FortranPairList) -> bool:
    nsym = len(pair_list.sym_indices[0]) if pair_list.sym_indices else 0
    terms: dict[tuple[tuple[PairKey, int], ...], int] = {}
    for isym in range(nsym):
        powers: dict[PairKey, int] = {}
        weight = 1
        valid = True
        for factor_index in combo:
            factor = pair_list.sym_factors[factor_index][isym]
            if factor is None:
                valid = False
                break
            powers[factor] = powers.get(factor, 0) + 1
            weight *= pair_list.sym_signs[factor_index][isym]
        if valid:
            key = tuple(sorted((factor, power) for factor, power in powers.items() if power))
            terms[key] = terms.get(key, 0) + weight
    return any(weight != 0 for weight in terms.values())


def generate_fortran_anchored_basis(
    xcart,
    xred,
    cutoff: float,
    symrel,
    ncell=(1, 1, 1),
    rprimd=None,
    tnons=None,
    power_range=(3, 4),
    atom_types: Sequence[int] | None = None,
    include_strain_coupling: bool = False,
    strain_voigts: Sequence[int] = (1, 2, 3, 4, 5, 6),
    max_nbody: int | None = None,
) -> list[XmlBasisFunction]:
    """Generate an anchored displacement basis matching Fortran ``fit_iatom=-2``.

    For each irreducible atom type (or all atoms if ``atom_types`` is ``None``),
    generates a per-central-atom pair list, enumerates compatible symmetry-unique
    combinations, materializes symmetry orbits as XML terms, and collects all
    resulting basis functions into a single list.
    """
    positions = np.asarray(xcart, dtype=float)
    reduced = np.asarray(xred, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("xcart must have shape (natom, 3)")
    natom = positions.shape[0]
    ncell_tuple = _tuple3_int(ncell, "ncell")
    min_power, max_power = _tuple2_int(power_range, "power_range")
    lattice = np.eye(3) if rprimd is None else np.asarray(rprimd, dtype=float)
    symrel_ops = np.asarray([np.eye(3, dtype=int)] if symrel is None else symrel, dtype=int)

    from pymultibinit.pyeffpot.symmetry import build_atom_mapping
    translations = np.zeros((len(symrel_ops), 3), dtype=float) if tnons is None else np.asarray(tnons, dtype=float)

    if atom_types is None:
        atom_types = list(range(natom))

    basis: list[XmlBasisFunction] = []
    for iatom in atom_types:
        pair_list = generate_fortran_pair_list(
            positions, reduced, cutoff, iatom, symrel_ops,
            ncell=ncell_tuple, rprimd=lattice, tnons=translations,
        )

        from itertools import combinations_with_replacement
        cell_to_index = {cell: idx for idx, cell in enumerate(pair_list.cells)}
        compatible_all = _fortran_pair_compatibility(
            pair_list.factors, pair_list.dist, cell_to_index, lattice, ncell_tuple,
        )
        compatible_irreducible = _fortran_pair_compatibility(
            pair_list.irreducible, pair_list.dist, cell_to_index, lattice, ncell_tuple,
        )

        accepted_combos: dict[tuple[int, ...], tuple[int, ...]] = {}
        for total_power in range(min_power, max_power + 1):
            for combo in combinations_with_replacement(range(pair_list.ncoeff_sym), total_power):
                if max_nbody is not None and len(set(combo)) > max_nbody:
                    continue
                if not all(compatible_irreducible[left, right] for left in combo for right in combo):
                    continue
                if len(set(combo)) == 1 and not _fortran_onebody_right_order(pair_list.irreducible[combo[0]]):
                    continue
                for expanded in _fortran_expand_combination(combo, pair_list, compatible_all):
                    key = _fortran_reduction_key(expanded, max_power)
                    if not _fortran_combination_seen(key, pair_list, accepted_combos):
                        accepted_combos[key] = tuple(sorted(expanded))

        for expanded in accepted_combos.values():
            if not _fortran_combination_has_nonzero_terms(expanded, pair_list):
                continue
            # Build the orbit of symmetry images for XML materialization
            nsym = len(pair_list.sym_indices[0]) if pair_list.sym_indices else 0
            orbit: dict[MonomialKey, int] = {}
            for isym in range(nsym):
                powers: dict[PairKey, int] = {}
                weight = 1
                valid = True
                for factor_index in expanded:
                    factor = pair_list.sym_factors[factor_index][isym]
                    if factor is None:
                        valid = False
                        break
                    powers[factor] = powers.get(factor, 0) + 1
                    weight *= pair_list.sym_signs[factor_index][isym]
                if valid:
                    monomial = MonomialKey(tuple(
                        sorted((k, v) for k, v in powers.items() if v)
                    ))
                    # Fortran uses weight=1 per unique monomial (no multiplicity accumulation)
                    if monomial not in orbit:
                        orbit[monomial] = 1

            # Find representative (lexicographically smallest MonomialKey)
            representative = min(orbit.keys())
            basis.append(_basis_from_orbit(
                len(basis) + 1, representative, orbit,
            ))

            if include_strain_coupling:
                nbody_disp = len(representative.factors)
                if max_nbody is None or nbody_disp + 1 <= max_nbody:
                    for voigt in strain_voigts:
                        basis.append(_basis_from_orbit(
                            len(basis) + 1, representative, orbit,
                            strains=({"power": 1, "voigt": voigt},),
                        ))

    return basis


def _basis_from_monomial(number: int, monomial: MonomialKey, sign: int) -> XmlBasisFunction:
    return XmlBasisFunction(number=number, value=0.0, text=_monomial_text(monomial), terms=(_term_from_monomial(monomial, sign),))


def _basis_from_orbit(number: int, representative: MonomialKey, orbit: Mapping[MonomialKey, int], strains=()) -> XmlBasisFunction:
    terms = tuple(_term_from_monomial(monomial, sign, strains=strains) for monomial, sign in sorted(orbit.items(), key=lambda item: _monomial_sort_key(item[0])))
    text = _monomial_text(representative)
    if strains:
        text = "*".join((text, *(_strain_text(strain) for strain in strains)))
    return XmlBasisFunction(number=number, value=0.0, text=text, terms=terms)


def _term_from_monomial(monomial: MonomialKey, sign: int, strains=()):
    displacements = []
    for factor, power in monomial.factors:
        displacements.append(
            {
                "atom_a": factor.atom_a,
                "atom_b": factor.atom_b,
                "direction": "xyz"[factor.direction],
                "power": power,
                "cell_a": (0, 0, 0),
                "cell_b": factor.cell_b,
            }
        )
    return {"weight": float(sign), "displacements": tuple(displacements), "strains": tuple(dict(strain) for strain in strains)}


def _compatible_monomial(monomial: MonomialKey) -> bool:
    if len(monomial.factors) <= 1:
        return True
    node_sets = [{(factor.atom_a, (0, 0, 0)), (factor.atom_b, factor.cell_b)} for factor, _ in monomial.factors]
    connected = set(node_sets[0])
    changed = True
    remaining = node_sets[1:]
    while changed and remaining:
        changed = False
        keep = []
        for atoms in remaining:
            if connected & atoms:
                connected |= atoms
                changed = True
            else:
                keep.append(atoms)
        remaining = keep
    return not remaining


def _monomial_orbit(monomial: MonomialKey, actions) -> list[tuple[MonomialKey, int]]:
    orbit = [normalize_monomial_key(monomial)]
    for action in actions:
        transformed = []
        sign = 1
        for factor, power in monomial.factors:
            mapped, factor_sign = action[factor]
            transformed.append((mapped, power))
            if factor_sign == -1 and int(power) % 2:
                sign *= -1
        normalized, normalized_sign = normalize_monomial_key(MonomialKey(tuple(transformed)))
        orbit.append((normalized, sign * normalized_sign))
    unique = {}
    for key, sign in orbit:
        unique.setdefault(key, sign)
    return list(unique.items())


def _symmetry_allowed_monomial(monomial: MonomialKey, actions) -> bool:
    normalized, _ = normalize_monomial_key(monomial)
    for action in actions:
        transformed = []
        sign = 1
        for factor, power in monomial.factors:
            mapped, factor_sign = action[factor]
            transformed.append((mapped, power))
            if factor_sign == -1 and int(power) % 2:
                sign *= -1
        transformed_key, transformed_sign = normalize_monomial_key(MonomialKey(tuple(transformed)))
        if transformed_key == normalized and sign * transformed_sign == -1:
            return False
    return True


def _monomial_text(monomial: MonomialKey) -> str:
    parts = []
    for factor, power in monomial.factors:
        part = f"u{factor.direction}_{factor.atom_a}_{factor.atom_b}"
        if power != 1:
            part += f"^{power}"
        parts.append(part)
    return "*".join(parts)


def _strain_text(strain) -> str:
    power = int(strain["power"])
    text = f"eta{int(strain['voigt'])}"
    if power != 1:
        text += f"^{power}"
    return text


def _monomial_sort_key(monomial: MonomialKey):
    return tuple((factor.direction, factor.atom_a, factor.atom_b, factor.cell_b, -power) for factor, power in monomial.factors)


def _cartesian_direction_rotation(symrel: np.ndarray, rprimd) -> np.ndarray:
    if rprimd is None:
        rotation = symrel
    else:
        lattice = np.asarray(rprimd, dtype=float)
        if lattice.shape != (3, 3):
            raise ValueError(f"rprimd must have shape (3, 3), got {lattice.shape}")
        rotation = lattice @ symrel.astype(float) @ np.linalg.inv(lattice)
    rounded = np.rint(rotation).astype(int)
    if not np.allclose(rotation, rounded, atol=1e-8):
        raise ValueError("Symmetry operation is not a Cartesian signed-axis permutation for factor actions")
    return rounded


def _map_atom(atom: int, mapping) -> tuple[int, tuple[int, int, int]]:
    if mapping is None:
        return int(atom), (0, 0, 0)
    mapped = mapping[int(atom)]
    if isinstance(mapped, tuple) and len(mapped) == 2:
        mapped_atom, cell_shift = mapped
        return int(mapped_atom), tuple(int(v) for v in cell_shift)
    return int(mapped), (0, 0, 0)


def _normalize_atom_mappings(atom_mappings, symrel):
    nsym = len(symrel)
    if atom_mappings is None:
        return [None] * nsym
    if isinstance(atom_mappings, np.ndarray):
        if atom_mappings.ndim != 3 or atom_mappings.shape[0] != 4 or atom_mappings.shape[1] != nsym:
            raise ValueError("atom_mappings array must have shape (4, nsym, natom)")
        mappings = []
        for isym in range(nsym):
            mapping = {}
            for iatom in range(atom_mappings.shape[2]):
                mapped_atom = int(atom_mappings[3, isym, iatom])
                translation = atom_mappings[:3, isym, iatom].astype(int)
                mapping[iatom] = (mapped_atom, tuple(int(v) for v in translation))
            mappings.append(mapping)
        return mappings
    mappings = list(atom_mappings)
    if len(mappings) != nsym:
        raise ValueError("atom_mappings must have one entry per symmetry operation")
    return mappings


def _reference_frame_from_ddb(ddb_path: Path, ncell: tuple[int, int, int]) -> TrainingFrame:
    from pymultibinit.pyeffpot.ddb_parser_complete import read_ddb
    from pymultibinit.pyeffpot.supercell_builder import build_supercell

    supercell = build_supercell(read_ddb(str(ddb_path)), ncell)
    crystal = supercell.crystal_sc
    return TrainingFrame(
        rprimd=crystal.rprimd,
        xred=crystal.xred,
        xcart=crystal.xcart,
        energy=supercell.unitcell.energy * supercell.ncells,
        forces=np.zeros((crystal.natom, 3), dtype=float),
        stress=np.zeros(6, dtype=float),
    )


def _resolve_executable(executable=None) -> Path:
    value = executable or os.environ.get("MULTIBINIT_BINARY") or os.environ.get("PYMULTIBINIT_MULTIBINIT_BINARY")
    if value is None:
        value = shutil.which("multibinit")
    if value is None:
        raise FileNotFoundError(
            "MULTIBINIT executable not found. Provide executable=..., set MULTIBINIT_BINARY, "
            "or put 'multibinit' on PATH."
        )
    candidate = Path(value)
    if candidate.parent != Path(".") or os.sep in str(value):
        candidate = candidate.resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"MULTIBINIT executable not found: {value}")
        return candidate
    found = shutil.which(str(value))
    if found is None:
        raise FileNotFoundError(f"MULTIBINIT executable not found: {value}")
    return Path(found).resolve()


def _discover_artifacts(outdir: Path) -> dict[str, str]:
    skipped = {"multibinit.stdout.log", "multibinit.stderr.log", "pymultibinit_training_result.json"}
    artifacts: dict[str, str] = {}
    for path in sorted(outdir.iterdir()):
        if path.is_file() and path.name not in skipped:
            artifacts[path.name] = str(path)
    return artifacts


def _choose_model_config(artifacts: Mapping[str, str]) -> Optional[str]:
    preferred = ("model.conf", "multibinit.conf", "trained_model.conf")
    for name in preferred:
        if name in artifacts:
            return artifacts[name]
    for name, path in artifacts.items():
        if name.endswith((".conf", ".ini")):
            return path
    return None
