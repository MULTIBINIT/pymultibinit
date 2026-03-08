"""
Unit tests for XML parser module.

How to run:
    pytest pymultibinit/tests/test_xml_parser.py -v

What it tests:
- Reading coefficient XML files
- Writing coefficient XML files
- Round-trip read/write consistency
- Data structure correctness
"""

import pytest
import tempfile
import os
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from pymultibinit.pyeffpot.xml_parser import (
    read_coefficient_xml,
    write_coefficient_xml,
    PolynomialCoefficient,
    PolynomialTerm
)


class TestXMLParser:
    """Test XML coefficient parser."""
    
    @pytest.fixture
    def test_file(self):
        """Path to test coefficient file."""
        return Path(__file__).parent.parent.parent / 'abinit/tests/tutomultibinit/Input/tmulti_l_7_1_coeffs.xml'
    
    @pytest.fixture
    def test_file2(self):
        """Path to another test coefficient file."""
        return Path(__file__).parent.parent.parent / 'abinit/tests/tutomultibinit/Input/tmulti_l_8_1.xml'
    
    def test_read_coefficients(self, test_file):
        """Test reading coefficient file."""
        coeffs = read_coefficient_xml(test_file)
        
        assert len(coeffs) == 12
        assert all(isinstance(c, PolynomialCoefficient) for c in coeffs)
        
        # Check first coefficient
        c0 = coeffs[0]
        assert c0.number == 1
        assert abs(c0.value - (-0.4278388107)) < 1e-6
        assert len(c0.terms) == 6
        
    def test_term_structure(self, test_file):
        """Test structure of polynomial terms."""
        coeffs = read_coefficient_xml(test_file)
        c0 = coeffs[0]
        term0 = c0.terms[0]
        
        assert isinstance(term0, PolynomialTerm)
        assert abs(term0.weight - 1.0) < 1e-6
        assert len(term0.displacements) > 0 or len(term0.strains) > 0
        
        # Check displacement structure
        if term0.displacements:
            disp = term0.displacements[0]
            assert 'atom_a' in disp
            assert 'atom_b' in disp
            assert 'direction' in disp
            assert 'power' in disp
            assert 'cell_a' in disp
            assert 'cell_b' in disp
            assert len(disp['cell_a']) == 3
            assert len(disp['cell_b']) == 3
    
    def test_strain_coupling(self, test_file):
        """Test strain-phonon coupling terms."""
        coeffs = read_coefficient_xml(test_file)
        
        # Find a coefficient with strain coupling
        has_strain = False
        for coeff in coeffs:
            for term in coeff.terms:
                if term.strains:
                    has_strain = True
                    strain = term.strains[0]
                    assert 'power' in strain
                    assert 'voigt' in strain
                    # Voigt indices are 1-6 in XML (1-based, Fortran convention)
                    assert 1 <= strain['voigt'] <= 6
        
        assert has_strain, "Should have at least one strain term"
    
    def test_roundtrip(self, test_file):
        """Test read→write→read consistency."""
        # Read original
        coeffs1 = read_coefficient_xml(test_file)
        
        # Write to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            temp_file = f.name
        
        try:
            write_coefficient_xml(temp_file, coeffs1)
            
            # Read back
            coeffs2 = read_coefficient_xml(temp_file)
            
            # Compare
            assert len(coeffs1) == len(coeffs2)
            
            for c1, c2 in zip(coeffs1, coeffs2):
                assert c1.number == c2.number
                assert abs(c1.value - c2.value) < 1e-10
                assert len(c1.terms) == len(c2.terms)
                
                for t1, t2 in zip(c1.terms, c2.terms):
                    assert abs(t1.weight - t2.weight) < 1e-6
                    assert len(t1.displacements) == len(t2.displacements)
                    assert len(t1.strains) == len(t2.strains)
        
        finally:
            os.unlink(temp_file)
    
    def test_large_file(self, test_file2):
        """Test reading larger coefficient file."""
        coeffs = read_coefficient_xml(test_file2)
        
        assert len(coeffs) == 99
        
        # All coefficients should have valid structure
        for coeff in coeffs:
            assert isinstance(coeff, PolynomialCoefficient)
            assert isinstance(coeff.terms, list)
            assert len(coeff.terms) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
