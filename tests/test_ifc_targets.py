"""Tests for pymultibinit.pyeffpot.ifc_targets (Story 2, AC-1..AC-7).

Run:  PYTHONPATH=src pytest tests/test_ifc_targets.py -v
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from phonopy import file_IO

from pymultibinit.phonon import build_phonopy, calculate_analytic_phonon
from pymultibinit.pyeffpot.datastructures import CrystalInfo, IFCData, UnitcellData
from pymultibinit.pyeffpot.ifc_targets import (
    ASR_TOL,
    HA_BOHR2_TO_EV_ANGSTROM2,
    IfcTargetError,
    IfcTargetSpec,
    IfcUnitCell,
    fixed_ifc,
    generate_ifc_target,
    load_ifc_target,
    with_fitted_values,
)
from pymultibinit.pyeffpot.potential import EffectivePotential
from pymultibinit.pyeffpot.second_derivatives import analytic_blocks
from pymultibinit.pyeffpot.supercell_builder import build_supercell, set_anharmonic_coeffs
from pymultibinit.pyeffpot.xml_parser import PolynomialCoefficient, PolynomialTerm

ASE = pytest.importorskip("ase")
phonopy = pytest.importorskip("phonopy")


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def unitcell() -> IfcUnitCell:
    return IfcUnitCell(
        cell=np.eye(3) * 4.0,
        symbols=("Ba", "Ti", "O", "O", "O"),
        scaled_positions=np.array(
            [[0, 0, 0], [.5, .5, .5], [.5, 0, 0], [0, .5, 0], [0, 0, .5]]),
    )


@pytest.fixture(scope="module")
def spring_ifc() -> np.ndarray:
    """Symmetric ASR-clean spring matrix for a 40-atom supercell."""
    n3 = 3 * 40
    rng = np.random.default_rng(0)
    b = rng.normal(size=(n3, n3)) * 0.05
    k = b + b.T + np.eye(n3) * 0.3
    k -= np.diag(k.sum(axis=1))
    assert np.abs(k.sum(axis=1)).max() < 1e-12
    return k


@pytest.fixture(scope="module")
def spring_forces(spring_ifc):
    """get_forces callable for the phonopy supercell of `unitcell`."""
    def get_forces(atoms):
        u = (atoms.get_positions() - get_forces.ref).reshape(-1)
        return -(spring_ifc @ u).reshape(-1, 3)
    return get_forces


def _sidecar(unitcell, smat=None, **overrides):
    sidecar = {
        "supercell_matrix": smat or [[2, 0, 0], [0, 2, 0], [0, 0, 2]],
        "primitive_matrix": np.eye(3).tolist(),
        "unitcell": unitcell.to_dict(),
        "atom_order": "phonopy",
        "units": "eV/angstrom^2",
        "semantics": "total",
        "asr_applied": False,
        "dipdip_removed": False,
    }
    sidecar.update(overrides)
    return sidecar


def _write_target(tmp_path, ifc_flat, unitcell, name="FORCE_CONSTANTS",
                  smat=None, **overrides):
    n3 = ifc_flat.shape[0]
    n = n3 // 3
    layout = ifc_flat.reshape(n, 3, n, 3).transpose(0, 2, 1, 3)
    fc_path = tmp_path / name
    if fc_path.suffix == ".hdf5":
        file_IO.write_force_constants_to_hdf5(layout, filename=str(fc_path))
    else:
        file_IO.write_FORCE_CONSTANTS(layout, filename=str(fc_path))
    sidecar_path = tmp_path / (name + ".sidecar.json")
    sidecar_path.write_text(json.dumps(_sidecar(unitcell, smat, **overrides)))
    return fc_path, sidecar_path


# ----------------------------------------------------------------------
# Import (AC-1, AC-2, AC-3)
# ----------------------------------------------------------------------

class TestImport:
    def test_text_roundtrip(self, tmp_path, unitcell, spring_ifc):
        fc_path, _ = _write_target(tmp_path, spring_ifc, unitcell)
        target = load_ifc_target(
            IfcTargetSpec(id="t", mode="import", fc_file=str(fc_path)))
        assert target.natsuper == 40
        assert target.unitcell.to_dict() == unitcell.to_dict()
        assert np.array_equal(target.supercell_matrix, np.diag([2, 2, 2]))
        np.testing.assert_allclose(target.ifc, spring_ifc, atol=1e-12)
        assert target.metadata["source_mode"] == "import"
        assert target.metadata["atom_order"] == "phonopy"

    def test_hdf5_roundtrip(self, tmp_path, unitcell, spring_ifc):
        fc_path, _ = _write_target(tmp_path, spring_ifc, unitcell,
                                   name="force_constants.hdf5")
        target = load_ifc_target(
            IfcTargetSpec(id="t", mode="import", fc_file=str(fc_path)))
        np.testing.assert_allclose(target.ifc, spring_ifc, atol=1e-12)

    def test_explicit_sidecar_path(self, tmp_path, unitcell, spring_ifc):
        fc_path, sidecar_path = _write_target(tmp_path, spring_ifc, unitcell)
        sidecar_path.rename(tmp_path / "elsewhere.json")
        target = load_ifc_target(IfcTargetSpec(
            id="t", mode="import", fc_file=str(fc_path),
            sidecar_file=str(tmp_path / "elsewhere.json")))
        np.testing.assert_allclose(target.ifc, spring_ifc, atol=1e-12)

    def test_ha_bohr_units_are_converted_once(self, tmp_path, unitcell, spring_ifc):
        fc_path, _ = _write_target(
            tmp_path, spring_ifc / HA_BOHR2_TO_EV_ANGSTROM2, unitcell,
            name="fc_ha.hdf5", units="Ha/Bohr^2")
        target = load_ifc_target(
            IfcTargetSpec(id="t", mode="import", fc_file=str(fc_path)))
        assert target.metadata["units_converted_from_ha_bohr"] is True
        np.testing.assert_allclose(target.ifc, spring_ifc, atol=1e-10)

    def test_missing_fc_file(self, unitcell):
        with pytest.raises(IfcTargetError, match="t_x"):
            load_ifc_target(IfcTargetSpec(
                id="t_x", mode="import", fc_file="/nonexistent/FORCE_CONSTANTS"))

    def test_missing_sidecar(self, tmp_path, unitcell, spring_ifc):
        fc_path, _ = _write_target(tmp_path, spring_ifc, unitcell)
        (tmp_path / "FORCE_CONSTANTS.sidecar.json").unlink()
        with pytest.raises(IfcTargetError, match="mandatory sidecar not found"):
            load_ifc_target(
                IfcTargetSpec(id="t_x", mode="import", fc_file=str(fc_path)))

    @pytest.mark.parametrize("missing", [
        "supercell_matrix", "primitive_matrix", "unitcell", "atom_order",
        "units", "semantics", "asr_applied", "dipdip_removed",
    ])
    def test_sidecar_missing_field(self, tmp_path, unitcell, spring_ifc, missing):
        fc_path, sidecar_path = _write_target(tmp_path, spring_ifc, unitcell)
        sidecar = json.loads(sidecar_path.read_text())
        del sidecar[missing]
        sidecar_path.write_text(json.dumps(sidecar))
        with pytest.raises(IfcTargetError, match=f"missing mandatory field.*{missing}|{missing}"):
            load_ifc_target(
                IfcTargetSpec(id="t_x", mode="import", fc_file=str(fc_path)))

    def test_bad_atom_order(self, tmp_path, unitcell, spring_ifc):
        fc_path, _ = _write_target(tmp_path, spring_ifc, unitcell,
                                   atom_order="internal")
        with pytest.raises(IfcTargetError, match="unsupported atom_order"):
            load_ifc_target(
                IfcTargetSpec(id="t_x", mode="import", fc_file=str(fc_path)))

    def test_bad_units(self, tmp_path, unitcell, spring_ifc):
        fc_path, _ = _write_target(tmp_path, spring_ifc, unitcell,
                                   units="Ry/bohr^2")
        with pytest.raises(IfcTargetError, match="units must be"):
            load_ifc_target(
                IfcTargetSpec(id="t_x", mode="import", fc_file=str(fc_path)))

    def test_shape_mismatch_vs_sidecar(self, tmp_path, unitcell, spring_ifc):
        # sidecar says 2x2x2 of a 5-atom cell (40 atoms); file has 20
        small = spring_ifc[:60, :60].copy()
        small -= np.diag(small.sum(axis=1))
        fc_path, _ = _write_target(tmp_path, small, unitcell)
        with pytest.raises(IfcTargetError, match="sidecar defines 40"):
            load_ifc_target(
                IfcTargetSpec(id="t_x", mode="import", fc_file=str(fc_path)))

    def test_non_finite(self, tmp_path, unitcell, spring_ifc):
        bad = spring_ifc.copy()
        bad[3, 4] = np.nan
        fc_path, _ = _write_target(tmp_path, bad, unitcell)
        with pytest.raises(IfcTargetError, match="non-finite"):
            load_ifc_target(
                IfcTargetSpec(id="t_x", mode="import", fc_file=str(fc_path)))

    def test_reciprocity_violation(self, tmp_path, unitcell, spring_ifc):
        bad = spring_ifc.copy()
        bad[3, 4] += 0.5  # breaks symmetry well above roundoff
        fc_path, _ = _write_target(tmp_path, bad, unitcell)
        with pytest.raises(IfcTargetError, match="reciprocity"):
            load_ifc_target(
                IfcTargetSpec(id="t_x", mode="import", fc_file=str(fc_path)))

    def test_asr_violation_total(self, tmp_path, unitcell, spring_ifc):
        bad = spring_ifc.copy()
        bad[0, 1] += 10.0 * ASR_TOL
        bad[1, 0] += 10.0 * ASR_TOL
        fc_path, _ = _write_target(tmp_path, bad, unitcell)
        with pytest.raises(IfcTargetError, match="acoustic"):
            load_ifc_target(
                IfcTargetSpec(id="t_x", mode="import", fc_file=str(fc_path)))

    def test_asr_skipped_for_short_range(self, tmp_path, unitcell, spring_ifc):
        bad = spring_ifc.copy()
        bad[0, 1] += 10.0 * ASR_TOL
        bad[1, 0] += 10.0 * ASR_TOL
        fc_path, _ = _write_target(tmp_path, bad, unitcell,
                                   semantics="short_range",
                                   dipdip_removed=True)
        target = load_ifc_target(
            IfcTargetSpec(id="t", mode="import", fc_file=str(fc_path)))
        assert target.metadata["semantics"] == "short_range"
        assert target.ifc[0, 1] == pytest.approx(spring_ifc[0, 1] + 10.0 * ASR_TOL)

    def test_structure_ref_with_hash(self, tmp_path, unitcell, spring_ifc):
        fc_path, _ = _write_target(tmp_path, spring_ifc, unitcell)
        struct_path = tmp_path / "unitcell.json"
        struct_path.write_text(json.dumps(unitcell.to_dict()))
        sidecar_path = tmp_path / "FORCE_CONSTANTS.sidecar.json"
        sidecar = _sidecar(unitcell)
        del sidecar["unitcell"]
        import hashlib
        sidecar["structure_ref"] = {
            "path": str(struct_path),
            "sha256": hashlib.sha256(struct_path.read_bytes()).hexdigest(),
        }
        sidecar_path.write_text(json.dumps(sidecar))
        target = load_ifc_target(
            IfcTargetSpec(id="t", mode="import", fc_file=str(fc_path)))
        assert target.unitcell.to_dict() == unitcell.to_dict()

    def test_structure_ref_hash_mismatch(self, tmp_path, unitcell, spring_ifc):
        fc_path, _ = _write_target(tmp_path, spring_ifc, unitcell)
        struct_path = tmp_path / "unitcell.json"
        struct_path.write_text(json.dumps(unitcell.to_dict()))
        sidecar_path = tmp_path / "FORCE_CONSTANTS.sidecar.json"
        sidecar = _sidecar(unitcell)
        del sidecar["unitcell"]
        sidecar["structure_ref"] = {"path": str(struct_path), "sha256": "0" * 64}
        sidecar_path.write_text(json.dumps(sidecar))
        with pytest.raises(IfcTargetError, match="content hash mismatch"):
            load_ifc_target(
                IfcTargetSpec(id="t_x", mode="import", fc_file=str(fc_path)))

    def test_spec_validation(self, unitcell):
        with pytest.raises(IfcTargetError, match="mode"):
            IfcTargetSpec(id="x", mode="magic")
        with pytest.raises(IfcTargetError, match="fc_file"):
            IfcTargetSpec(id="x", mode="import")
        with pytest.raises(IfcTargetError, match="structure is required"):
            IfcTargetSpec(id="x", mode="generate")
        with pytest.raises(IfcTargetError, match="weight"):
            IfcTargetSpec(id="x", mode="import", fc_file="f", weight=-1.0)
        with pytest.raises(IfcTargetError, match="stencil"):
            IfcTargetSpec(id="x", mode="generate", structure=unitcell,
                          stencil="forward")


# ----------------------------------------------------------------------
# Generation (AC-4) and caching (AC-5)
# ----------------------------------------------------------------------

class TestGeneration:
    @pytest.fixture
    def atoms(self, unitcell):
        from ase import Atoms
        return Atoms(symbols=list(unitcell.symbols), cell=unitcell.cell,
                     scaled_positions=unitcell.scaled_positions, pbc=True)

    @pytest.fixture(autouse=True)
    def _bind_reference(self, atoms, spring_forces):
        _, supercell = build_phonopy(atoms, supercell_matrix=np.diag([2, 2, 2]))
        spring_forces.ref = supercell.get_positions()
        yield

    def _spec(self, unitcell, **kw):
        params = dict(id="g", mode="generate", structure=unitcell,
                      supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
                      displacement=0.01, stencil="central",
                      calculator_config={"type": "spring"})
        params.update(kw)
        return IfcTargetSpec(**params)

    def test_central_stencil(self, unitcell, spring_ifc, spring_forces, tmp_path):
        target = generate_ifc_target(self._spec(unitcell),
                                     spring_forces, cache_dir=str(tmp_path))
        assert target.metadata["cache"] == "miss"
        np.testing.assert_allclose(target.ifc, spring_ifc, atol=1e-8)

    def test_central5_stencil(self, unitcell, spring_ifc, spring_forces):
        target = generate_ifc_target(self._spec(unitcell, stencil="central5"),
                                     spring_forces)
        np.testing.assert_allclose(target.ifc, spring_ifc, atol=1e-8)

    def test_drift_correction_recorded(self, unitcell, spring_ifc,
                                       spring_forces, tmp_path):
        target = generate_ifc_target(
            self._spec(unitcell, drift_correction=True),
            spring_forces, cache_dir=str(tmp_path))
        gen = target.metadata["generator"]
        assert gen["drift_correction"] is True
        assert target.metadata["asr_applied"] is True
        np.testing.assert_allclose(target.ifc, spring_ifc, atol=1e-8)

    def test_supercell_order_matches_calculate_analytic_phonon(
            self, atoms, unitcell, spring_ifc, spring_forces):
        class StubCalculator:
            def get_analytic_blocks(self, supercell_atoms):
                class Blocks:
                    ifc = spring_ifc
                return Blocks()

        phonon_ref = calculate_analytic_phonon(
            atoms, StubCalculator(), supercell_matrix=np.diag([2, 2, 2]))
        phonon_gen, supercell = build_phonopy(
            atoms, supercell_matrix=np.diag([2, 2, 2]))
        np.testing.assert_array_equal(
            phonon_ref.supercell.scaled_positions,
            phonon_gen.supercell.scaled_positions)
        np.testing.assert_array_equal(
            phonon_ref.supercell.numbers, phonon_gen.supercell.numbers)
        np.testing.assert_allclose(
            phonon_ref.supercell.cell, phonon_gen.supercell.cell)
        # generated artifact round-trips through import (same FD data)
        target = generate_ifc_target(
            self._spec(unitcell), spring_forces)
        np.testing.assert_allclose(target.ifc, spring_ifc, atol=1e-8)

    def test_cache_hit_and_invalidation(self, tmp_path, unitcell, spring_ifc,
                                        spring_forces):
        spec = self._spec(unitcell)
        first = generate_ifc_target(spec, spring_forces, cache_dir=str(tmp_path))
        assert first.metadata["cache"] == "miss"
        second = generate_ifc_target(spec, spring_forces, cache_dir=str(tmp_path))
        assert second.metadata["cache"] == "hit"
        assert second.content_hash == first.content_hash
        np.testing.assert_array_equal(second.ifc, first.ifc)

        changed_disp = self._spec(unitcell, displacement=0.02)
        third = generate_ifc_target(changed_disp, spring_forces,
                                    cache_dir=str(tmp_path))
        assert third.metadata["cache"] == "miss"
        assert third.content_hash != first.content_hash

        changed_calc = self._spec(
            unitcell, calculator_config={"type": "mace", "model": "x"})
        fourth = generate_ifc_target(changed_calc, spring_forces,
                                     cache_dir=str(tmp_path))
        assert fourth.metadata["cache"] == "miss"

        perturbed = IfcUnitCell(
            cell=unitcell.cell, symbols=unitcell.symbols,
            scaled_positions=unitcell.scaled_positions + 1e-4)
        fifth = generate_ifc_target(self._spec(perturbed), spring_forces,
                                    cache_dir=str(tmp_path))
        assert fifth.metadata["cache"] == "miss"

    def test_no_cache_dir_writes_artifact(self, unitcell, spring_ifc,
                                          spring_forces):
        target = generate_ifc_target(self._spec(unitcell), spring_forces)
        assert target.metadata["cache"] == "write-only"
        assert Path(target.metadata["fc_file"]).exists()


# ----------------------------------------------------------------------
# K_fixed (AC-6)
# ----------------------------------------------------------------------

def _simple_supercell():
    crystal = CrystalInfo(
        natom=2, ntypat=2, rprimd=np.eye(3) * 7.0,
        xred=np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
        xcart=np.array([[0.0, 0.0, 0.0], [3.5, 3.5, 3.5]]),
        typat=np.array([1, 2]), amu=np.array([100.0, 50.0]),
        znucl=np.array([50, 25]))
    ifcs = IFCData(nrpt=1, cell=np.zeros((3, 1), dtype=int),
                   atmfrc=np.zeros((3, 2, 3, 2, 1)),
                   short_atmfrc=np.zeros((3, 2, 3, 2, 1)))
    ifcs.atmfrc[:, 0, :, 0, 0] = np.eye(3) * 0.1
    ifcs.atmfrc[:, 1, :, 1, 0] = np.eye(3) * 0.1
    ifcs.short_atmfrc = ifcs.atmfrc.copy()
    unitcell = UnitcellData(crystal=crystal, energy=-100.0, ifcs=ifcs,
                            epsilon_inf=np.eye(3),
                            zeff=np.zeros((3, 3, 2)))
    return build_supercell(unitcell, (2, 2, 2))


def _anharmonic_coeff(value=1.7):
    term = PolynomialTerm(
        weight=1.0,
        displacements=[{
            "atom_a": 0, "atom_b": 1, "direction": "x", "power": 2,
            "cell_a": [0, 0, 0], "cell_b": [0, 0, 0],
        }],
        strains=[{"power": 1, "voigt": 1}],
    )
    return PolynomialCoefficient(number=1, value=value, text="u2_eta",
                                 terms=[term])


class TestFixedIfc:
    def test_fixed_ifc_equals_coefficient_free_potential(self):
        sc_full = _simple_supercell()
        set_anharmonic_coeffs(sc_full, [_anharmonic_coeff(value=1.7)])
        potential = EffectivePotential(sc_full)

        sc_free = _simple_supercell()  # no anharmonic coeffs at all
        free = EffectivePotential(sc_free)

        rng = np.random.default_rng(3)
        xcart = potential._reference_positions + rng.normal(scale=0.05,
                                                            size=(16, 3))
        rprimd = potential._reference_lattice.copy()

        k_fixed = fixed_ifc(potential, xcart, rprimd)

        u = free._compute_displacements(xcart, rprimd)
        eta = free._compute_strain(rprimd)
        k_free = analytic_blocks(free, u, eta).ifc * HA_BOHR2_TO_EV_ANGSTROM2
        # exact by coefficient linearity (derivation D3): zero-valued terms
        # scatter exact 0.0
        np.testing.assert_array_equal(k_fixed, k_free)

    def test_fixed_ifc_excludes_fitted_contribution(self):
        sc = _simple_supercell()
        set_anharmonic_coeffs(sc, [_anharmonic_coeff(value=1.7)])
        potential = EffectivePotential(sc)

        xcart = potential._reference_positions.copy()
        xcart[0, 0] += 0.05
        rprimd = potential._reference_lattice.copy()

        xcart = potential._reference_positions.copy()
        xcart[0, 0] += 0.05
        # strain the cell too: the fitted term carries eta_xx^1, which
        # vanishes (and hides the term) at the unstrained reference
        rprimd = potential._reference_lattice * 1.01

        u = potential._compute_displacements(xcart, rprimd)
        eta = potential._compute_strain(rprimd)
        k_full = analytic_blocks(potential, u, eta).ifc * HA_BOHR2_TO_EV_ANGSTROM2

        k_fixed = fixed_ifc(potential, xcart, rprimd)
        assert not np.allclose(k_fixed, k_full)
        assert np.abs(k_full - k_fixed).max() > 0.0

    def test_with_fitted_values_recompiles(self):
        sc = _simple_supercell()
        set_anharmonic_coeffs(sc, [_anharmonic_coeff(value=1.7)])
        potential = EffectivePotential(sc)
        assert potential._anharmonic_compiled[0]["value"] == 1.7

        zeroed = with_fitted_values(potential, [0.0])
        assert zeroed._anharmonic_compiled[0]["value"] == 0.0
        # original untouched
        assert potential._anharmonic_compiled[0]["value"] == 1.7
        with pytest.raises(IfcTargetError, match="values for"):
            with_fitted_values(potential, [0.0, 1.0])

    def test_unit_boundary_is_ev_per_angstrom2(self):
        sc = _simple_supercell()
        set_anharmonic_coeffs(sc, [_anharmonic_coeff(value=0.0)])
        potential = EffectivePotential(sc)
        xcart = potential._reference_positions.copy()
        rprimd = potential._reference_lattice.copy()
        u = potential._compute_displacements(xcart, rprimd)
        eta = potential._compute_strain(rprimd)
        raw = analytic_blocks(potential, u, eta).ifc
        np.testing.assert_allclose(
            fixed_ifc(potential, xcart, rprimd),
            raw * HA_BOHR2_TO_EV_ANGSTROM2, rtol=0, atol=0)
