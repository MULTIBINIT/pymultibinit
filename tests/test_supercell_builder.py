"""
Unit tests for supercell builder module.

How to run:
    pytest pymultibinit/tests/test_supercell_builder.py -v

What it tests:
- Supercell geometry generation
- IFC replication
- ASR enforcement
- Data structure consistency
"""

import pytest
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from pymultibinit.pyeffpot.datastructures import (
    UnitcellData, CrystalInfo, IFCData
)
from pymultibinit.pyeffpot.supercell_builder import (
    build_supercell,
    _build_supercell_geometry,
    _apply_asr,
    set_anharmonic_coeffs
)
from pymultibinit.pyeffpot import read_ddb


class TestSupercellBuilder:
    """Test supercell builder functionality."""
    
    @pytest.fixture
    def simple_unitcell(self):
        """Create a simple 2-atom unitcell for testing."""
        # Simple cubic lattice
        rprimd = np.eye(3) * 7.0  # 7 Bohr lattice constant
        
        # 2 atoms at (0,0,0) and (0.5, 0.5, 0.5)
        xred = np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5]
        ])
        
        xcart = xred @ rprimd.T
        
        crystal = CrystalInfo(
            natom=2,
            ntypat=2,
            rprimd=rprimd,
            xred=xred,
            xcart=xcart,
            typat=np.array([1, 2]),
            amu=np.array([100.0, 50.0]),
            znucl=np.array([50, 25])
        )
        
        # Simple IFCs (just diagonal for testing)
        nrpt = 1
        cell = np.array([[0, 0, 0]]).T  # (3, 1)
        
        # Initialize IFCs
        atmfrc = np.zeros((3, 2, 3, 2, 1))
        short_atmfrc = np.zeros((3, 2, 3, 2, 1))
        
        # Add some simple force constants
        atmfrc[:, 0, :, 0, 0] = np.eye(3) * 0.1
        atmfrc[:, 1, :, 1, 0] = np.eye(3) * 0.1
        short_atmfrc = atmfrc.copy()
        
        ifcs = IFCData(
            nrpt=nrpt,
            cell=cell,
            atmfrc=atmfrc,
            short_atmfrc=short_atmfrc
        )
        
        unitcell = UnitcellData(
            crystal=crystal,
            energy=-100.0,
            ifcs=ifcs,
            epsilon_inf=np.eye(3) * 6.0,
            zeff=np.zeros((3, 3, 2))
        )
        
        return unitcell
    
    def test_supercell_geometry(self, simple_unitcell):
        """Test supercell geometry generation."""
        ncell = (2, 2, 2)
        crystal_sc = _build_supercell_geometry(simple_unitcell.crystal, ncell)
        
        # Check number of atoms
        assert crystal_sc.natom == simple_unitcell.natom * 8
        
        # Check lattice vectors
        expected_rprimd = np.diag([2, 2, 2]) @ simple_unitcell.crystal.rprimd
        np.testing.assert_array_almost_equal(crystal_sc.rprimd, expected_rprimd)
        
        # Check atom types preserved
        assert len(np.unique(crystal_sc.typat)) == simple_unitcell.ntypat
        
    def test_build_supercell(self, simple_unitcell):
        """Test full supercell building."""
        ncell = (2, 2, 2)
        supercell = build_supercell(simple_unitcell, ncell)
        
        # Check basic properties
        assert supercell.ncell == ncell
        assert supercell.natom_sc == simple_unitcell.natom * 8
        assert supercell.ncells == 8
        
        # Check IFCs exist
        assert supercell.ifcs_sc is not None
        assert supercell.ifcs_sc.atmfrc.shape[1] == supercell.natom_sc
        
    def test_asr_enforcement(self):
        """Test acoustic sum rule enforcement."""
        natom = 4
        nrpt = 1
        
        # Create IFCs that violate ASR
        atmfrc = np.random.rand(3, natom, 3, natom, nrpt)
        short_atmfrc = atmfrc.copy()
        
        ifcs = IFCData(
            nrpt=nrpt,
            cell=np.zeros((3, nrpt)),
            atmfrc=atmfrc,
            short_atmfrc=short_atmfrc
        )
        
        # Apply ASR
        _apply_asr(ifcs)
        
        # Check that sum over j,nu is zero for each i,mu
        for i in range(natom):
            for mu in range(3):
                sum_ifc = np.sum(ifcs.atmfrc[mu, i, :, :, 0])
                assert abs(sum_ifc) < 1e-10, f"ASR violated for atom {i}, direction {mu}"
                
    def test_supercell_volume(self, simple_unitcell):
        """Test that supercell volume is correct."""
        ncell = (2, 2, 2)
        supercell = build_supercell(simple_unitcell, ncell)
        
        # Unitcell volume
        vol_uc = np.abs(np.linalg.det(simple_unitcell.crystal.rprimd))
        
        # Supercell volume
        vol_sc = np.abs(np.linalg.det(supercell.crystal_sc.rprimd))
        
        # Should be 8x larger
        assert abs(vol_sc / vol_uc - 8.0) < 1e-10
        
    def test_anharmonic_coeffs(self, simple_unitcell):
        """Test setting anharmonic coefficients."""
        ncell = (2, 2, 2)
        supercell = build_supercell(simple_unitcell, ncell)
        
        # Initially None
        assert supercell.anharmonic_coeffs is None
        
        # Set coefficients
        coeffs = [1, 2, 3]  # Simplified
        set_anharmonic_coeffs(supercell, coeffs)
        
        # Check set
        assert supercell.anharmonic_coeffs == coeffs
        
    def test_different_supercell_sizes(self, simple_unitcell):
        """Test different supercell dimensions."""
        sizes = [(2, 2, 2), (3, 3, 3), (2, 3, 4)]
        
        for ncell in sizes:
            supercell = build_supercell(simple_unitcell, ncell)
            expected_natom = simple_unitcell.natom * ncell[0] * ncell[1] * ncell[2]
            assert supercell.natom_sc == expected_natom

    def test_read_ddb_integrates_with_supercell_builder(self):
        """Test exported read_ddb output works with build_supercell."""
        ddb_path = Path(__file__).parent.parent.parent / 'abinit/tests/v9/Input/BTO.DDB'
        unitcell = read_ddb(str(ddb_path))

        assert unitcell.crystal.natom == 5
        assert unitcell.ifcs is not None
        assert unitcell.ifcs.nrpt == 1
        assert unitcell.rprimd.shape == (3, 3)

        supercell = build_supercell(unitcell, (2, 2, 2))
        assert supercell.natom_sc == 40
        assert supercell.ifcs_sc.atmfrc.shape == (3, 40, 3, 40, 1)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
