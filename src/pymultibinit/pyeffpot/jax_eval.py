"""JAX GPU-accelerated polynomial term evaluation for pymultibinit.

Shared kernel serving both training (feature matrix construction) and
model prediction (EffectivePotential.evaluate). Converts Python dict
basis terms to flat static-shape arrays, then evaluates energy/forces/stress
via vectorized jax.numpy operations.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Sequence, Optional


def _detect_backend() -> str:
    try:
        import jax
        devs = jax.devices()
        if any("gpu" in str(d).lower() or "cuda" in str(d).lower() for d in devs):
            return "gpu"
        return "cpu"
    except Exception:
        return "none"


@dataclass
class CompiledTerms:
    coeff: np.ndarray
    atom_a: np.ndarray
    atom_b: np.ndarray
    direction: np.ndarray
    power: np.ndarray
    factor_mask: np.ndarray
    strain_voigt_idx: np.ndarray  # (n, max_strains)
    strain_power: np.ndarray      # (n, max_strains)
    strain_mask: np.ndarray       # (n, max_strains)
    natom_sc: int
    ncells: int
    max_factors: int
    max_strains: int


def _supercell_atom_index(atom_uc: int, ix: int, iy: int, iz: int,
                          ncell: tuple[int, int, int], natom_uc: int) -> int:
    return ((ix % ncell[0]) * ncell[1] * ncell[2] +
            (iy % ncell[1]) * ncell[2] +
            (iz % ncell[2])) * natom_uc + atom_uc


def compile_terms(basis_coeffs: Sequence, ncell: tuple[int, int, int],
                  natom_uc: int) -> CompiledTerms:
    nx, ny, nz = int(ncell[0]), int(ncell[1]), int(ncell[2])
    ncells = nx * ny * nz

    def get_sc_indices(atom_uc: int, cell: tuple) -> np.ndarray:
        idx = np.zeros(ncells, dtype=np.int64)
        count = 0
        for ix0 in range(nx):
            for iy0 in range(ny):
                for iz0 in range(nz):
                    ix = (ix0 + int(cell[0])) % nx
                    iy = (iy0 + int(cell[1])) % ny
                    iz = (iz0 + int(cell[2])) % nz
                    idx[count] = _supercell_atom_index(atom_uc, ix, iy, iz, ncell, natom_uc)
                    count += 1
        return idx

    dir_map = {"x": 0, "y": 1, "z": 2}

    all_terms = []
    for coeff in basis_coeffs:
        for term in coeff.terms:
            disps = []
            for d in term.displacements:
                if d.get("power", 0) == 0:
                    continue
                idx_a = get_sc_indices(d["atom_a"], d.get("cell_a", (0, 0, 0)))
                idx_b = get_sc_indices(d["atom_b"], d.get("cell_b", (0, 0, 0)))
                disps.append((idx_a, idx_b, dir_map[d["direction"]], int(d["power"])))

            strains = term.strains if hasattr(term, "strains") else term.get("strains", [])
            if isinstance(strains, dict):
                strains = [strains]

            strain_list = []
            for s in strains:
                sv = int(s.get("voigt", s.get("voigt_index", 0)))
                sp = int(s.get("power", 1))
                if sv > 0 and sp > 0:
                    strain_list.append((sv, sp))

            all_terms.append({
                "coeff": float(coeff.value) * float(term.weight if hasattr(term, "weight") else term.get("weight", 1.0)),
                "disps": disps,
                "strains": strain_list,
            })

    if not all_terms:
        return CompiledTerms(
            coeff=np.zeros(0), atom_a=np.zeros((0,1,1),dtype=np.int64),
            atom_b=np.zeros((0,1,1),dtype=np.int64), direction=np.zeros((0,1),dtype=np.int64),
            power=np.zeros((0,1)), factor_mask=np.zeros((0,1)),
            strain_voigt_idx=np.zeros((0,1),dtype=np.int64), strain_power=np.zeros((0,1)),
            strain_mask=np.zeros((0,1)),
            natom_sc=natom_uc*ncells, ncells=ncells, max_factors=1, max_strains=1,
        )

    max_f = max(len(t["disps"]) for t in all_terms)
    max_f = max(max_f, 1)
    max_s = max(len(t["strains"]) for t in all_terms)
    max_s = max(max_s, 1)
    n = len(all_terms)
    natom_sc = natom_uc * ncells

    coeff = np.zeros(n)
    atom_a = np.zeros((n, max_f, ncells), dtype=np.int64)
    atom_b = np.zeros((n, max_f, ncells), dtype=np.int64)
    direction = np.zeros((n, max_f), dtype=np.int64)
    power = np.zeros((n, max_f))
    factor_mask = np.zeros((n, max_f))
    strain_voigt_idx = np.zeros((n, max_s), dtype=np.int64)
    strain_power = np.zeros((n, max_s))
    strain_mask = np.zeros((n, max_s))

    for i, t in enumerate(all_terms):
        coeff[i] = t["coeff"]
        for j, (ia, ib, d, p) in enumerate(t["disps"]):
            atom_a[i, j] = ia
            atom_b[i, j] = ib
            direction[i, j] = d
            power[i, j] = p
            factor_mask[i, j] = 1.0
        for k, (sv, sp) in enumerate(t["strains"]):
            strain_voigt_idx[i, k] = sv
            strain_power[i, k] = sp
            strain_mask[i, k] = 1.0

    return CompiledTerms(
        coeff=coeff, atom_a=atom_a, atom_b=atom_b,
        direction=direction, power=power, factor_mask=factor_mask,
        strain_voigt_idx=strain_voigt_idx, strain_power=strain_power,
        strain_mask=strain_mask,
        natom_sc=natom_sc, ncells=ncells, max_factors=max_f, max_strains=max_s,
    )


def evaluate_numpy(compiled: CompiledTerms,
                   displacements: np.ndarray,
                   strain_voigt: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """NumPy reference implementation for parity testing."""
    n, max_f, c = compiled.atom_a.shape
    natom = compiled.natom_sc
    forces = np.zeros((natom, 3))
    stress_voigt = np.zeros(6)
    energy = 0.0

    for i in range(n):
        base = compiled.coeff[i]

        strain_val = 1.0
        for k in range(compiled.max_strains):
            if compiled.strain_mask[i, k] == 0:
                continue
            sv = int(compiled.strain_voigt_idx[i, k])
            sp = int(compiled.strain_power[i, k])
            if sv > 0 and sp > 0:
                strain_val *= strain_voigt[sv - 1] ** sp

        diffs = np.ones((max_f, c))
        for f in range(max_f):
            if compiled.factor_mask[i, f] == 0:
                continue
            ia = compiled.atom_a[i, f]
            ib = compiled.atom_b[i, f]
            d = compiled.direction[i, f]
            diffs[f] = displacements[ia, d] - displacements[ib, d]

        prod = np.ones(c)
        for f in range(max_f):
            if compiled.factor_mask[i, f] == 0:
                continue
            prod *= diffs[f] ** compiled.power[i, f]

        term_energy = base * strain_val * prod
        energy += term_energy.sum()

        for f in range(max_f):
            if compiled.factor_mask[i, f] == 0 or compiled.power[i, f] == 0:
                continue
            deriv = base * strain_val * compiled.power[i, f] * prod / (diffs[f] + 1e-10)
            np.add.at(forces[:, compiled.direction[i, f]], compiled.atom_a[i, f], -deriv)
            np.add.at(forces[:, compiled.direction[i, f]], compiled.atom_b[i, f], deriv)

        for k in range(compiled.max_strains):
            if compiled.strain_mask[i, k] == 0:
                continue
            sv = int(compiled.strain_voigt_idx[i, k])
            sp = int(compiled.strain_power[i, k])
            if sv <= 0 or sp <= 0:
                continue
            # Compute partial derivative of term_energy w.r.t. strain[sv-1]
            # d/d(ε) of (coeff * Π(ε_j^p_j) * prod_disp) = term_energy * p_k / ε_k
            s_val = strain_voigt[sv - 1]
            if abs(s_val) > 1e-12:
                stress_voigt[sv - 1] += (term_energy * sp / s_val).sum()
            elif sp == 1:
                # At ε=0, d(ε)/dε = 1, so stress = base * prod_disp
                stress_voigt[sv - 1] += (base * prod).sum()

    return energy, forces, stress_voigt


def evaluate_jax(compiled: CompiledTerms,
                 displacements: np.ndarray,
                 strain_voigt: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """JAX-accelerated evaluation for a single configuration."""
    backend = _detect_backend()
    if backend == "none":
        return evaluate_numpy(compiled, displacements, strain_voigt)

    import jax
    import jax.numpy as jnp

    n, max_f, c = compiled.atom_a.shape
    natom = compiled.natom_sc

    coeff = jnp.array(compiled.coeff)
    atom_a = jnp.array(compiled.atom_a)
    atom_b = jnp.array(compiled.atom_b)
    direction = jnp.array(compiled.direction)
    power = jnp.array(compiled.power)
    factor_mask = jnp.array(compiled.factor_mask)
    sv_idx = jnp.array(compiled.strain_voigt_idx)  # (N, max_s)
    sp = jnp.array(compiled.strain_power)           # (N, max_s)
    sm = jnp.array(compiled.strain_mask)            # (N, max_s)

    disp = jnp.array(displacements)
    sv = jnp.array(strain_voigt)

    disp_a = disp[atom_a, direction[:, :, None]]
    disp_b = disp[atom_b, direction[:, :, None]]
    diffs = disp_a - disp_b

    safe_power = jnp.where(factor_mask > 0, power, 0.0)
    powered = jnp.where(factor_mask[:, :, None] > 0, diffs ** safe_power[:, :, None], 1.0)

    total_prod = powered.prod(axis=1)

    safe_sv = sv_idx - 1
    safe_sv = jnp.where(sm > 0, safe_sv, 0)
    strain_vals = jnp.where(sm > 0, sv[safe_sv] ** jnp.where(sm > 0, sp, 0), 1.0)
    strain_mult = strain_vals.prod(axis=1)

    term_energy = coeff[:, None] * strain_mult[:, None] * total_prod
    energy = float(term_energy.sum())

    eye_f = jnp.eye(max_f)
    masked_powered = powered[:, None, :, :] * (1 - eye_f[None, :, :, None]) + eye_f[None, :, :, None]
    prod_without_f = masked_powered.prod(axis=2)

    safe_pm1 = jnp.where(factor_mask > 0, jnp.maximum(power - 1, 0), 0.0)
    diff_reduced = jnp.where(factor_mask[:, :, None] > 0,
                             diffs ** safe_pm1[:, :, None], 0.0)

    contrib = coeff[:, None, None] * strain_mult[:, None, None] * power[:, :, None] * \
              diff_reduced * prod_without_f
    contrib = jnp.where(factor_mask[:, :, None] > 0, contrib, 0.0)

    flat_idx_a = (atom_a * 3 + direction[:, :, None]).reshape(-1)
    flat_idx_b = (atom_b * 3 + direction[:, :, None]).reshape(-1)
    flat_contrib = contrib.reshape(-1)
    forces_flat = jnp.zeros(natom * 3)
    forces_flat = forces_flat.at[flat_idx_a].add(-flat_contrib)
    forces_flat = forces_flat.at[flat_idx_b].add(flat_contrib)
    forces = np.array(forces_flat.reshape(natom, 3))

    stress_voigt = np.zeros(6)
    for v in range(1, 7):
        for k in range(compiled.max_strains):
            kmask = jnp.where((sv_idx[:, k] == v) & (sm[:, k] > 0), 1.0, 0.0)
            if not bool(kmask.any()):
                continue
            pk = int(compiled.strain_power[:, k].max()) if bool(kmask.any()) else 0
            s_val = float(sv[v - 1])
            if abs(s_val) > 1e-12:
                contrib = term_energy * float(pk) * kmask[:, None] / s_val
                stress_voigt[v - 1] += float(contrib.sum())
            elif pk == 1:
                contrib = coeff[:, None] * total_prod * kmask[:, None]
                stress_voigt[v - 1] += float(contrib.sum())

    return energy, forces, stress_voigt


def evaluate_batch_jax(compiled: CompiledTerms,
                       displacements_batch: np.ndarray,
                       strain_batch: np.ndarray,
                       chunk_size: int = 512) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Batched evaluation for training feature matrices.

    Returns energy (nframes,), forces (nframes, natom, 3), stress_voigt (nframes, 6).
    """
    nframes = displacements_batch.shape[0]
    natom = compiled.natom_sc
    energies = np.zeros(nframes)
    forces = np.zeros((nframes, natom, 3))
    stresses = np.zeros((nframes, 6))

    for t in range(nframes):
        e, f, s = evaluate_jax(compiled, displacements_batch[t], strain_batch[t])
        energies[t] = e
        forces[t] = f
        stresses[t] = s

    return energies, forces, stresses
