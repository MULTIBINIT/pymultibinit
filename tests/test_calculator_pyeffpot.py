"""
Tests for the pure-Python (pyeffpot) ASE calculator path.

Covers ``MultibinitCalculator.from_pyeffpot`` which builds an ASE
calculator from a DDB file (+ optional XML) without requiring
libabinit. The tests exercise the ASE ``calculate`` interface
(``get_potential_energy``, ``get_forces``, ``get_stress``) using the
locally bundled BaTiO3 example fixture so they run in any environment.

Run with:
    pytest pymultibinit/tests/test_calculator_pyeffpot.py -v
"""

import pytest
import numpy as np
from pathlib import Path

try:
    from ase import Atoms  # noqa: F401
    _HAS_ASE = True
except ImportError:
    _HAS_ASE = False

from pymultibinit import MultibinitCalculator
from pymultibinit.potential import MultibinitPotential

# Local fixture: always present in the repo, does not depend on external ABINIT test data
BTO_DIR = Path(__file__).resolve().parents[1] / "examples" / "BaTiO3_example"
BTO_DDB = BTO_DIR / "BaTiO3_DDB"
BTO_XML = BTO_DIR / "BaTiO3.xml"

_skip_no_ase = pytest.mark.skipif(not _HAS_ASE, reason="ASE not installed")
_skip_no_fixture = pytest.mark.skipif(
    not BTO_DDB.exists(),
    reason=f"BaTiO3 example DDB fixture missing: {BTO_DDB}",
)


def _bto_atoms_supercell(ncell=(2, 2, 2)):
    """Build an ASE Atoms object matching the MULTIBINIT reference supercell.

    The calculator's atom-matcher will reconcile any ordering/translation
    differences automatically, so we only need a structure with the right
    cell and the right number of atoms at roughly the right sites.
    """
    from pymultibinit import MultibinitPotential

    pot = MultibinitPotential.from_pyeffpot(
        ddb_file=str(BTO_DDB), ncell=ncell, dipdip=True, asr=True
    )
    ref_pos, ref_lat, _ = pot.get_supercell_structure()
    atoms = Atoms(symbols=["X"] * len(ref_pos), positions=ref_pos, cell=ref_lat, pbc=True)
    return atoms


@_skip_no_ase
@_skip_no_fixture
class TestMultibinitCalculatorFromPyeffpot:
    """Verify ``MultibinitCalculator.from_pyeffpot`` end-to-end."""

    def test_constructor_returns_calculator(self):
        calc = MultibinitCalculator.from_pyeffpot(
            ddb_file=str(BTO_DDB), ncell=(2, 2, 2)
        )
        assert isinstance(calc, MultibinitCalculator)
        assert calc.potential is not None
        assert calc.potential.backend == "pyeffpot"
        assert calc.potential.expected_natoms == 40  # 5 atoms/cell * 2^3
        # ASE interface is wired
        assert calc.implemented_properties == ["energy", "forces", "stress"]

    def test_constructor_with_optional_xml(self):
        # Same XML that ships with the example - exercises the optional xml_file path
        calc = MultibinitCalculator.from_pyeffpot(
            ddb_file=str(BTO_DDB),
            xml_file=str(BTO_XML),
            ncell=(2, 2, 2),
        )
        assert calc.potential.backend == "pyeffpot"
        assert calc.potential.expected_natoms == 40

    def test_constructor_xml_none_is_explicit_no_xml(self):
        # Passing xml_file=None explicitly should not raise
        calc = MultibinitCalculator.from_pyeffpot(
            ddb_file=str(BTO_DDB), xml_file=None, ncell=(1, 1, 1)
        )
        assert calc.potential.expected_natoms == 5

    def test_dipdip_flag_does_not_crash(self):
        # Both True and False must be accepted; only checking construction here.
        for dipdip in (True, False):
            calc = MultibinitCalculator.from_pyeffpot(
                ddb_file=str(BTO_DDB), ncell=(2, 2, 2), dipdip=dipdip
            )
            assert calc.potential.backend == "pyeffpot"

    def test_calculate_energy_forces_stress_through_ase(self):
        calc = MultibinitCalculator.from_pyeffpot(
            ddb_file=str(BTO_DDB), ncell=(2, 2, 2), dipdip=True, asr=True
        )
        atoms = _bto_atoms_supercell((2, 2, 2))
        atoms.calc = calc

        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        stress = atoms.get_stress()

        # Finite and correct shapes
        assert np.isfinite(energy)
        assert forces.shape == (40, 3)
        assert stress.shape == (6,)
        assert np.all(np.isfinite(forces))
        assert np.all(np.isfinite(stress))

        # At the equilibrium reference positions, forces must vanish.
        assert np.max(np.abs(forces)) < 1e-6

    def test_energy_responds_to_displacement(self):
        # NOTE: we deliberately do NOT assert the sign of ΔE here. The BaTiO3
        # cubic reference is a saddle point along its ferroelectric soft mode,
        # so the energy can legitimately DECREASE along certain directions.
        calc = MultibinitCalculator.from_pyeffpot(
            ddb_file=str(BTO_DDB), ncell=(2, 2, 2)
        )
        atoms = _bto_atoms_supercell((2, 2, 2))
        atoms.calc = calc
        e0 = atoms.get_potential_energy()
        f0 = atoms.get_forces()

        atoms.positions[0] += np.array([0.05, 0.0, 0.0])
        e1 = atoms.get_potential_energy()
        f1 = atoms.get_forces()

        assert e1 != e0
        assert abs(e1 - e0) > 1e-9
        assert np.max(np.abs(f0)) < 1e-6
        assert np.max(np.abs(f1)) > 1e-3

        # Energy-force consistency via single-point finite difference:
        # -dE/dx ~ F. Sign must match when both terms are above noise floor.
        fd_force_x = -(e1 - e0) / 0.05
        actual_force_x = f1[0, 0]
        if abs(fd_force_x) > 1e-3 and abs(actual_force_x) > 1e-3:
            assert np.sign(fd_force_x) == np.sign(actual_force_x), (
                f"FD force sign {fd_force_x:.3e} vs actual {actual_force_x:.3e}"
            )

    def test_no_fortran_dependency(self):
        """The pyeffpot calculator must NOT touch the CFFI wrapper / libabinit."""
        calc = MultibinitCalculator.from_pyeffpot(
            ddb_file=str(BTO_DDB), ncell=(2, 2, 2)
        )
        assert calc.potential.wrapper is None
        # evaluate must work without ever loading libabinit
        atoms = _bto_atoms_supercell((2, 2, 2))
        atoms.calc = calc
        _ = atoms.get_potential_energy()
        assert calc.potential.wrapper is None  # still no wrapper


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
