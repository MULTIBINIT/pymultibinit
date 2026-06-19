"""Vectorized basis feature evaluators."""

from __future__ import annotations
from typing import Sequence, Mapping
from pathlib import Path
import tempfile
import numpy as np


# ---------------------------------------------------------------------------
# Helper: build a flat "compiled term" representation for vectorized eval
# ---------------------------------------------------------------------------

_ORIGIN_CACHE: dict[tuple[int, int, int], np.ndarray] = {}


def _origin_cells(ncell: tuple[int, int, int]) -> np.ndarray:
    key = ncell
    if key not in _ORIGIN_CACHE:
        nx, ny, nz = ncell
        _ORIGIN_CACHE[key] = np.array(
            [(ix, iy, iz) for ix in range(nx) for iy in range(ny) for iz in range(nz)],
            dtype=np.int32,
        )
    return _ORIGIN_CACHE[key]


def _supercell_indices_vec(atom, cell_shifts, origins, ncell, natom_uc):
    nx, ny, nz = ncell
    cells = (origins + cell_shifts) % np.array([nx, ny, nz], dtype=np.int32)
    return atom + natom_uc * (cells[:, 2] + nz * (cells[:, 1] + ny * cells[:, 0]))


def compile_term(term, ncell, natom_uc):
    """Pre-compile a term dict into flat arrays for vectorized evaluation."""
    origins = _origin_cells(ncell)
    weight = float(term["weight"])
    disps = list(term["displacements"])
    strains = list(term["strains"])

    ndisp = len(disps)
    nstrain = len(strains)

    disp_direction = np.zeros(ndisp, dtype=np.int32)
    disp_power = np.zeros(ndisp, dtype=np.int32)
    disp_idx_a = np.zeros((ndisp, len(origins)), dtype=np.int32)
    disp_idx_b = np.zeros((ndisp, len(origins)), dtype=np.int32)

    for i, d in enumerate(disps):
        direction_map = {"x": 0, "y": 1, "z": 2}
        disp_direction[i] = direction_map[d["direction"]]
        disp_power[i] = int(d["power"])
        cell_a = np.array(d.get("cell_a", [0, 0, 0]), dtype=np.int32)
        cell_b = np.array(d["cell_b"], dtype=np.int32)
        disp_idx_a[i] = _supercell_indices_vec(
            int(d["atom_a"]), cell_a, origins, ncell, natom_uc
        )
        disp_idx_b[i] = _supercell_indices_vec(
            int(d["atom_b"]), cell_b, origins, ncell, natom_uc
        )

    strain_voigt = np.zeros(nstrain, dtype=np.int32)
    strain_power = np.zeros(nstrain, dtype=np.int32)
    for i, s in enumerate(strains):
        strain_voigt[i] = int(s["voigt"]) - 1
        strain_power[i] = int(s["power"])

    return CompiledTerm(
        weight=weight,
        ncell=ncell,
        natom_uc=natom_uc,
        n_origins=len(origins),
        ndisp=ndisp,
        nstrain=nstrain,
        disp_direction=disp_direction,
        disp_power=disp_power,
        disp_idx_a=disp_idx_a,
        disp_idx_b=disp_idx_b,
        strain_voigt=strain_voigt,
        strain_power=strain_power,
    )


class CompiledTerm:
    """Pre-compiled term data for fast evaluation."""

    __slots__ = (
        "weight",
        "ncell",
        "natom_uc",
        "n_origins",
        "ndisp",
        "nstrain",
        "disp_direction",
        "disp_power",
        "disp_idx_a",
        "disp_idx_b",
        "strain_voigt",
        "strain_power",
    )

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def compile_basis(basis, ncell, natom_uc):
    """Pre-compile all terms of all basis functions."""
    compiled = []
    ncoeff = len(basis)
    for coeff in basis:
        coeff_terms = [compile_term(term, ncell, natom_uc) for term in coeff.terms]
        compiled.append(coeff_terms)
    return compiled


# ---------------------------------------------------------------------------
# Vectorized NumPy evaluator
# ---------------------------------------------------------------------------


def evaluate_term_vec(ct: CompiledTerm, displacement, strain, energy_out, forces_out, stress_out):
    """Vectorized term evaluation — all time frames at once."""
    ntime = displacement.shape[0]
    weight = ct.weight

    if ct.ndisp == 0 and ct.nstrain > 0:
        # Pure strain term
        strain_mult = np.ones(ntime, dtype=np.float64)
        for iv in range(ct.nstrain):
            strain_mult *= strain[:, ct.strain_voigt[iv]] ** ct.strain_power[iv]
        energy_out[:] += weight * strain_mult
        for iv in range(ct.nstrain):
            voigt = ct.strain_voigt[iv]
            pw = ct.strain_power[iv]
            if pw == 0:
                continue
            deriv = np.ones(ntime, dtype=np.float64) * weight * pw
            for jv in range(ct.nstrain):
                exp = ct.strain_power[jv] - 1 if jv == iv else ct.strain_power[jv]
                deriv *= strain[:, ct.strain_voigt[jv]] ** exp
            stress_out[:, voigt] += deriv
        return

    # Displacement differences for each factor: (ndisp, n_origins, ntime)
    disp_vals = np.zeros((ct.ndisp, ct.n_origins, ntime), dtype=np.float64)
    for i in range(ct.ndisp):
        idx_a = ct.disp_idx_a[i]  # (n_origins,)
        idx_b = ct.disp_idx_b[i]
        dir_ = ct.disp_direction[i]
        disp_vals[i] = displacement[:, idx_a, dir_].T - displacement[:, idx_b, dir_].T

    # Strain multiplier per time frame
    strain_mult = np.ones(ntime, dtype=np.float64)
    for iv in range(ct.nstrain):
        strain_mult *= strain[:, ct.strain_voigt[iv]] ** ct.strain_power[iv]

    # Product of disp_vals ** power: (n_origins, ntime)
    product = np.ones((ct.n_origins, ntime), dtype=np.float64) * strain_mult[None, :]
    for i in range(ct.ndisp):
        product *= disp_vals[i] ** ct.disp_power[i]
    product *= weight

    # Energy
    energy_out[:] += product.sum(axis=0)

    # Forces
    for i in range(ct.ndisp):
        pw = ct.disp_power[i]
        if pw == 0:
            continue
        deriv = np.ones((ct.n_origins, ntime), dtype=np.float64) * weight * pw
        for j in range(ct.ndisp):
            exp = ct.disp_power[j] - 1 if j == i else ct.disp_power[j]
            deriv *= disp_vals[j] ** exp
        deriv *= strain_mult[None, :]
        # np.add.at handles scattered accumulation
        dir_ = ct.disp_direction[i]
        np.add.at(forces_out[:, :, dir_], (slice(None), ct.disp_idx_a[i]), -deriv.T)
        np.add.at(forces_out[:, :, dir_], (slice(None), ct.disp_idx_b[i]), deriv.T)

    # Stress
    for iv in range(ct.nstrain):
        pw = ct.strain_power[iv]
        if pw == 0:
            continue
        voigt = ct.strain_voigt[iv]
        deriv = np.ones((ct.n_origins, ntime), dtype=np.float64) * weight * pw
        for jv in range(ct.nstrain):
            exp = ct.strain_power[jv] - 1 if jv == iv else ct.strain_power[jv]
            deriv *= strain[:, ct.strain_voigt[jv]] ** exp
        for i in range(ct.ndisp):
            deriv *= disp_vals[i] ** ct.disp_power[i]
        stress_out[:, voigt] += deriv.sum(axis=0)


def evaluate_basis_features_vectorized(
    basis,
    displacement,
    strain,
    du_delta,
    ucvol,
    ncell,
    natom_uc,
):
    """Evaluate basis functions into features using fully vectorized NumPy."""
    basis_list = list(basis)
    ntime = displacement.shape[0]
    natom_sc = displacement.shape[1]
    ncoeff = len(basis_list)

    energy = np.zeros((ntime, ncoeff), dtype=np.float64)
    forces = np.zeros((ntime, natom_sc, 3, ncoeff), dtype=np.float64)
    stress = np.zeros((ntime, 6, ncoeff), dtype=np.float64)

    compiled = compile_basis(basis_list, ncell, natom_uc)

    for icoeff, terms in enumerate(compiled):
        for ct in terms:
            evaluate_term_vec(
                ct, displacement, strain,
                energy[:, icoeff], forces[:, :, :, icoeff], stress[:, :, icoeff],
            )

    # Post-processing: du_delta correction, strain scaling
    stress -= np.einsum("tvna,tnac->tvc", du_delta, forces)
    axial = np.arange(3)
    shear = np.arange(3, 6)
    stress[:, axial, :] *= (1.0 + strain[:, axial, None]) / ucvol[:, None, None]
    stress[:, shear, :] *= (1.0 - strain[:, shear, None] ** 2) / ucvol[:, None, None]

    return energy, forces, stress


def evaluate_basis_features_memmap(
    basis,
    displacement,
    strain,
    du_delta,
    ucvol,
    ncell,
    natom_uc,
    output_dir=None,
):
    """Evaluate basis features into disk-backed arrays to reduce RAM use."""
    basis_list = list(basis)
    ntime = displacement.shape[0]
    natom_sc = displacement.shape[1]
    ncoeff = len(basis_list)
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="pymultibinit_features_")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    energy = np.memmap(output_path / "energy.dat", dtype=np.float64, mode="w+", shape=(ntime, ncoeff))
    forces = np.memmap(output_path / "forces.dat", dtype=np.float64, mode="w+", shape=(ntime, natom_sc, 3, ncoeff))
    stress = np.memmap(output_path / "stress.dat", dtype=np.float64, mode="w+", shape=(ntime, 6, ncoeff))
    energy[:] = 0.0
    forces[:] = 0.0
    stress[:] = 0.0

    compiled = compile_basis(basis_list, ncell, natom_uc)
    for icoeff, terms in enumerate(compiled):
        for ct in terms:
            evaluate_term_vec(
                ct,
                displacement,
                strain,
                energy[:, icoeff],
                forces[:, :, :, icoeff],
                stress[:, :, icoeff],
            )
        if icoeff % 256 == 0:
            energy.flush()
            forces.flush()
            stress.flush()

    stress -= np.einsum("tvna,tnac->tvc", du_delta, forces, optimize=True)
    axial = np.arange(3)
    shear = np.arange(3, 6)
    stress[:, axial, :] *= (1.0 + strain[:, axial, None]) / ucvol[:, None, None]
    stress[:, shear, :] *= (1.0 - strain[:, shear, None] ** 2) / ucvol[:, None, None]
    energy.flush()
    forces.flush()
    stress.flush()
    return energy, forces, stress


# ---------------------------------------------------------------------------
# Optional JAX evaluator
# ---------------------------------------------------------------------------


def _evaluate_terms_for_jax(compiled_data, displacement, strain):
    """Evaluate terms using NumPy before optional JAX post-processing."""
    ntime = displacement.shape[0]
    ncoeff = len(compiled_data)
    natom = displacement.shape[1]

    energy = np.zeros((ntime, ncoeff), dtype=np.float64)
    forces = np.zeros((ntime, natom, 3, ncoeff), dtype=np.float64)
    stress = np.zeros((ntime, 6, ncoeff), dtype=np.float64)

    for icoeff, terms in enumerate(compiled_data):
        for ct in terms:
            evaluate_term_vec(
                ct, displacement, strain,
                energy[:, icoeff], forces[:, :, :, icoeff], stress[:, :, icoeff],
            )

    return energy, forces, stress


def evaluate_basis_features_jax(basis, displacement, strain, du_delta, ucvol, ncell, natom_uc):
    """Evaluate basis features using JAX for stress post-processing.

    JAX is imported lazily because CUDA plugin discovery can emit warnings or
    fail on systems where the vectorized NumPy backend works correctly.
    """
    try:
        import jax
        import jax.numpy as jnp
    except Exception as exc:
        raise ImportError("JAX not available.") from exc

    compiled = compile_basis(basis, ncell, natom_uc)
    energy, forces, stress = _evaluate_terms_for_jax(compiled, displacement, strain)

    @jax.jit
    def _finalize_stress(energy_jax, forces_jax, stress_jax, strain_jax, du_delta_jax, ucvol_jax):
        stress_jax -= jnp.einsum("tvna,tnac->tvc", du_delta_jax, forces_jax)
        axial = jnp.arange(3)
        shear = jnp.arange(3, 6)
        stress_jax = stress_jax.at[:, axial, :].multiply((1.0 + strain_jax[:, axial, None]) / ucvol_jax[:, None, None])
        stress_jax = stress_jax.at[:, shear, :].multiply((1.0 - strain_jax[:, shear, None] ** 2) / ucvol_jax[:, None, None])
        return energy_jax, forces_jax, stress_jax

    try:
        energy_jax, forces_jax, stress_jax = _finalize_stress(
            jnp.asarray(energy),
            jnp.asarray(forces),
            jnp.asarray(stress),
            jnp.asarray(strain),
            jnp.asarray(du_delta),
            jnp.asarray(ucvol),
        )
        return np.asarray(energy_jax), np.asarray(forces_jax), np.asarray(stress_jax)
    except Exception as exc:
        raise ImportError("JAX feature backend failed to initialize.") from exc


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------


def evaluate_basis_features_auto(
    basis, displacement, strain, du_delta, ucvol, ncell, natom_uc,
    backend: str = "auto",
):
    """Evaluate features with the requested backend.

    ``auto`` detects JAX GPU if available, otherwise falls back to NumPy.
    Use ``jax`` explicitly to require JAX, or ``numpy`` to force NumPy.
    """
    if backend == "auto":
        try:
            from .pyeffpot.jax_eval import _detect_backend
            if _detect_backend() == "gpu":
                backend = "jax"
            else:
                backend = "numpy"
        except Exception:
            backend = "numpy"
    if backend == "jax":
        return evaluate_basis_features_jax(
            basis, displacement, strain, du_delta, ucvol, ncell, natom_uc,
        )
    if backend != "numpy":
        raise ValueError("backend must be one of: auto, numpy, jax")
    return evaluate_basis_features_vectorized(
        basis, displacement, strain, du_delta, ucvol, ncell, natom_uc,
    )
