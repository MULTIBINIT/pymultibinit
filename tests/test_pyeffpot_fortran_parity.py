"""
Pyeffpot (Python) vs libabinit (Fortran) effective-potential parity tests.

Ground truth is libabinit reached through mb_init_potential (the from_abi
path). mb_init_potential_simple / MultibinitPotential.from_params crashes on
an uninitialized efield_gmean in the compiled libabinit, so we use the
invars10 reader path with the DDB passed as the sys argument.

The suite isolates each physical component by feeding both backends identical
configurations and comparing energy, forces, and stress:

  reference      u=0, eps=0            -> E0, zero forces
  rattle         atoms displaced, eps=0-> harmonic IFCs (forces)
  scale          isotropic strain, u=0 -> elastic diagonal
  shear          off-diagonal strain, u=0 -> elastic shear
  rattle_strain  displacement + strain -> displacement-strain coupling

Two known bugs are pinned as xfail regression gates:
  * shear          -- displacement reference under strain (see shear xfail)
  * rattle_strain  -- the +/-0.66 eV coupling gap (see rattle_strain xfail)
When each is fixed, flip the xfail to a hard assertion.

Skip if libabinit is not loadable or the DDB fixture is absent.
"""
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pymultibinit.wrapper_cffi import MultibinitWrapperCFFI
from pymultibinit.potential import MultibinitPotential
from pymultibinit.pyeffpot.ddb_parser_complete import read_ddb
from pymultibinit.atom_matching import (
    find_atom_mapping_pbc,
    apply_mapping_to_positions,
    apply_inverse_mapping_to_forces,
)

HA = 27.211386245988
BOHR = 0.529177210903

_WORKSPACE = Path(__file__).resolve().parents[2]
_DDB_PATHS = {
    "BFO": _WORKSPACE / "debugs/BFO_arijit_harmonic/BFO-ref-arijit.ddb.out",
    "BaHfO3": Path(__file__).resolve().parents[1] / "examples/BaHfO3_example/BaHfO3_DDB",
}
NCELL = (2, 2, 2)


def _resolve_libabinit():
    """Return a loadable libabinit path, or None.

    Tries LIBABINIT_PATH, the committed workspace .libs build, then the
    generic find_library() -- returns the first that actually dlopens.
    """
    from cffi import FFI

    candidates = []
    env = os.environ.get("LIBABINIT_PATH")
    if env and os.path.exists(env):
        candidates.append(env)
    workspace_lib = _WORKSPACE / ".libs" / "libabinit.so"
    if workspace_lib.exists():
        candidates.append(str(workspace_lib))
    try:
        from pymultibinit.utils import find_library as _fl
        found = _fl()
        if found:
            candidates.append(found)
    except Exception:
        pass
    for cand in candidates:
        try:
            FFI().dlopen(cand)
            return cand
        except Exception:
            continue
    return None


RESOLVED_LIB = _resolve_libabinit()
skip_no_lib = pytest.mark.skipif(RESOLVED_LIB is None, reason="no loadable libabinit")


# --------------------------------------------------------------------------- #
# Fortran reference harness (ground-truth oracle)
# --------------------------------------------------------------------------- #
class FortranRef:
    """libabinit backend via mb_init_potential, returning ASE-unit quantities."""

    _cache = {}

    @classmethod
    def get(cls, ddb_name, dipdip):
        key = (ddb_name, dipdip)
        if key not in cls._cache:
            cls._cache[key] = cls(_DDB_PATHS[ddb_name], NCELL, dipdip)
        return cls._cache[key]

    def __init__(self, ddb_path, ncell, dipdip):
        self.w = MultibinitWrapperCFFI(lib_path=RESOLVED_LIB)
        ngqpt = tuple(int(i) for i in read_ddb(str(ddb_path)).ngqpt)
        abi_path = self._write_abi(ncell, ngqpt, dipdip)
        ddb_b = str(ddb_path).encode()
        abi_b = abi_path.encode()
        st = self.w.ffi.new("int*")
        nc = self.w.ffi.new("int[3]", list(ncell))
        self.handle = self.w.ffi.new("void**")
        self.w.lib.mb_init_potential(
            abi_b, len(abi_b), ddb_b, len(ddb_b), b"", 0, nc, self.handle, st
        )
        if st[0] != 0:
            raise RuntimeError(f"mb_init_potential failed with status {st[0]}")
        self.w.handle[0] = self.handle[0]
        self.n, self._spec, ref_b, lat_b = self.w.get_supercell_structure()
        self.ref = ref_b * BOHR
        self.flat = lat_b * BOHR
        self._map = None
        self._abi_path = abi_path

    @staticmethod
    def _write_abi(ncell, ngqpt, dipdip):
        fd, path = tempfile.mkstemp(suffix=".abi")
        with os.fdopen(fd, "w") as f:
            f.write(f" ncell {' '.join(map(str, ncell))}\n")
            f.write(f" ngqpt {' '.join(map(str, ngqpt))}\n")
            f.write(f" dipdip {int(bool(dipdip))}\n")
            f.write(" brav 1\n")
            f.write(" efield_gmean 0.0 0.0 0.0\n")
            f.write(" efield_gvel 0.0 0.0 0.0\n")
        return path

    def evaluate(self, pos_ang, lat_ang):
        if self._map is None:
            self._map, self._imap = find_atom_mapping_pbc(
                pos_ang, self.ref, lat_ang, tolerance=0.6
            )
        pos_f = apply_mapping_to_positions(pos_ang, self._map)
        pos = np.ascontiguousarray((pos_f / BOHR).flatten())
        # mb_evaluate reads lattice Fortran column-major (rprimd(:,j)=vector j);
        # ASE/pymultibinit use row-major (lat[i,:]=vector i). Transpose so the
        # Fortran sees the same lattice. Invisible for symmetric cells, required
        # for shear. Positions are Cartesian (no transpose).
        lat = np.ascontiguousarray((lat_ang.T / BOHR).flatten())
        e = self.w.ffi.new("double*")
        f = np.zeros(self.n * 3)
        s = np.zeros(6)
        st = self.w.ffi.new("int*")
        self.w.lib.mb_evaluate(
            self.handle[0],
            self.w.ffi.cast("double*", pos.ctypes.data),
            self.w.ffi.cast("double*", lat.ctypes.data),
            self.n, e,
            self.w.ffi.cast("double*", f.ctypes.data),
            self.w.ffi.cast("double*", s.ctypes.data),
            st,
        )
        energy = e[0] * HA
        forces = f.reshape(self.n, 3) * (HA / BOHR)
        stress = s * (HA / BOHR ** 3)
        return energy, apply_inverse_mapping_to_forces(forces, self._imap), stress

    def free(self):
        try:
            if self._abi_path and os.path.exists(self._abi_path):
                os.remove(self._abi_path)
        except OSError:
            pass


_PYMB = {}


def _get_pymb(ddb_name, dipdip):
    key = (ddb_name, dipdip)
    if key not in _PYMB:
        _PYMB[key] = MultibinitPotential.from_pyeffpot(
            ddb_file=str(_DDB_PATHS[ddb_name]), ncell=NCELL, dipdip=dipdip, asr=True,
        )
    return _PYMB[key]


# --------------------------------------------------------------------------- #
# Configuration generators (identical input to both backends)
# --------------------------------------------------------------------------- #
def _configs(ref_pos, ref_lat):
    """Return {name: (positions, lattice)} built from the pymultibinit reference."""
    rng = np.random.default_rng(42)
    xred = ref_pos @ np.linalg.inv(ref_lat).T
    natom = len(ref_pos)
    delta = rng.normal(0.0, 0.05, size=(natom, 3))  # ~0.05 Angstrom rattle
    f = 1.01
    lat_scale = f * ref_lat
    lat_shear = ref_lat.copy()
    lat_shear[2, 1] += 0.02 * ref_lat[1, 1]

    def affine(lat):
        return xred @ lat.T

    return {
        "reference": (ref_pos.copy(), ref_lat.copy()),
        "rattle": (ref_pos + delta, ref_lat.copy()),
        "scale": (affine(lat_scale), lat_scale),
        "shear": (affine(lat_shear), lat_shear),
        "rattle_strain": (ref_pos + delta, lat_shear),
    }


CONFIGS = [
    pytest.param("reference", id="reference"),
    pytest.param("rattle", id="rattle"),
    pytest.param("scale", id="scale"),
    pytest.param("shear", id="shear"),
    pytest.param("rattle_strain", id="rattle_strain"),
]


def _ddb_fixture_value(ddb_name):
    if not _DDB_PATHS[ddb_name].exists():
        pytest.skip(f"DDB fixture not available: {_DDB_PATHS[ddb_name]}")
    return ddb_name


_eref_py = {}
_eref_f = {}


def _eref(ddb_name, dipdip):
    key = (ddb_name, dipdip)
    if key not in _eref_py:
        pot = _get_pymb(ddb_name, dipdip)
        fref = FortranRef.get(ddb_name, dipdip)
        rp, rl, _ = pot.get_supercell_structure()
        e_py, _, _ = pot.evaluate(rp, rl, skip_atom_matching=True)
        e_f, _, _ = fref.evaluate(rp, rl)
        _eref_py[key] = e_py
        _eref_f[key] = e_f
    return _eref_py[key], _eref_f[key]


# Per-config energy check: rtol on the reference-subtracted signal + an atol floor.
# All configs pass against the libabinit/HIST ground truth once the mb_evaluate
# lattice (column-major) transpose is applied in FortranRef.evaluate.
_E_TOL = {
    "reference": dict(rtol=1e-8, atol=1e-10),
    "rattle": dict(rtol=1e-6, atol=1e-8),
    "scale": dict(rtol=1e-8, atol=1e-10),
    "shear": dict(rtol=1e-6, atol=1e-8),
    "rattle_strain": dict(rtol=1e-6, atol=1e-8),
}


@pytest.mark.parametrize("ddb_name", ["BFO", "BaHfO3"])
@pytest.mark.parametrize("dipdip", [True, False])
@pytest.mark.parametrize("config", CONFIGS)
@skip_no_lib
def test_energy_forces_stress_parity(ddb_name, dipdip, config):
    _ddb_fixture_value(ddb_name)
    pot = _get_pymb(ddb_name, dipdip)
    fref = FortranRef.get(ddb_name, dipdip)

    ref_pos, ref_lat, _ = pot.get_supercell_structure()
    pos, lat = _configs(ref_pos, ref_lat)[config]

    e_py, f_py, s_py = pot.evaluate(pos, lat, skip_atom_matching=True)
    e_f, f_f, s_f = fref.evaluate(pos, lat)

    eref_py, eref_f = _eref(ddb_name, dipdip)
    sig_py = e_py - eref_py
    sig_f = e_f - eref_f
    tol = _E_TOL[config]
    de_sig = abs(sig_py - sig_f)
    de_lim = tol["rtol"] * max(abs(sig_f), tol["atol"]) + tol["atol"]
    assert de_sig < de_lim, (
        f"{config}/{ddb_name}: |dE_sig|={de_sig:.4e} > {de_lim:.4e} "
        f"(sig_py={sig_py:.6f} sig_f={sig_f:.6f})")

    f_scale = max(np.max(np.abs(f_f)), 1e-3)
    df = np.max(np.abs(f_py - f_f)) / f_scale
    f_rtol = {"reference": 1e-2, "rattle": 8e-2, "scale": 1e-2,
              "shear": 1e-2, "rattle_strain": 8e-2}[config]
    assert df < f_rtol, f"{config}/{ddb_name}: force rel-err={df:.4e}"

    assert np.allclose(s_py[:3], s_f[:3], atol=1e-3, rtol=3e-2), (
        f"{config}/{ddb_name}: stress diag py={s_py[:3]} f={s_f[:3]}")


@pytest.mark.parametrize("ddb_name", ["BFO", "BaHfO3"])
def test_dipdip_is_energy_neutral(ddb_name):
    """dipdip on/off must not change the harmonic energy (sum_R Phi = sum_q D)."""
    _ddb_fixture_value(ddb_name)
    pot_on = _get_pymb(ddb_name, True)
    pot_off = _get_pymb(ddb_name, False)
    ref_pos, ref_lat, _ = pot_on.get_supercell_structure()
    for name, (pos, lat) in _configs(ref_pos, ref_lat).items():
        if name in ("shear", "rattle_strain"):
            continue
        e_on, _, _ = pot_on.evaluate(pos, lat, skip_atom_matching=True)
        e_off, _, _ = pot_off.evaluate(pos, lat, skip_atom_matching=True)
        assert abs(e_on - e_off) < 1e-6, f"{name}: dipdip changed energy by {e_on-e_off:.2e}"
