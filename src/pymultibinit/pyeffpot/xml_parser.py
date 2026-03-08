"""
XML parser for MULTIBINIT coefficient files.

This module provides pure Python parsers for reading and writing
MULTIBINIT XML files containing anharmonic polynomial coefficients.

References:
- abinit/src/78_effpot/effpot_xml.c (C implementation with libxml)
- abinit/src/78_effpot/m_effective_potential_file.F90 (Fortran interface)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any
import xml.etree.ElementTree as ET
import numpy as np


@dataclass
class PolynomialTerm:
    """Single term in a polynomial coefficient."""
    weight: float
    displacements: List[Dict[str, Any]] = field(default_factory=list)
    strains: List[Dict[str, int]] = field(default_factory=list)
    
    def __repr__(self):
        parts = []
        for disp in self.displacements:
            parts.append(f"(atom_{disp['atom_a']}-{disp['atom_b']})")
        if self.strains:
            parts.append(f"(η)")
        return f"Term(weight={self.weight:.6f}, {' '.join(parts)})"


@dataclass
class PolynomialCoefficient:
    """Polynomial coefficient with multiple terms."""
    number: int
    value: float
    text: str
    terms: List[PolynomialTerm] = field(default_factory=list)
    
    def __repr__(self):
        return f"Coeff({self.number}, {self.value:.6e}, {len(self.terms)} terms)"


def parse_text_to_float(text: str) -> float:
    """Parse text content to float, handling whitespace."""
    return float(text.strip())


def parse_text_to_int_array(text: str, n: int) -> List[int]:
    """Parse space-separated integers."""
    values = text.split()
    return [int(v) for v in values[:n]]


def parse_text_to_float_array(text: str, shape: tuple) -> np.ndarray:
    """Parse space-separated floats into array of given shape."""
    values = [float(v) for v in text.split()]
    return np.array(values).reshape(shape)


def read_coefficient_xml(filename: str) -> List[PolynomialCoefficient]:
    """
    Read anharmonic polynomial coefficients from XML file.
    
    Parameters
    ----------
    filename : str
        Path to XML coefficient file.
    
    Returns
    -------
    List[PolynomialCoefficient]
        List of polynomial coefficients.
    
    Examples
    --------
    >>> coeffs = read_coefficient_xml("coeffs.xml")
    >>> print(f"Read {len(coeffs)} coefficients")
    >>> print(f"First coefficient: {coeffs[0].value:.6e}")
    """
    tree = ET.parse(filename)
    root = tree.getroot()
    
    coefficients = []
    
    for coeff_elem in root.findall('coefficient'):
        coeff_number_str = coeff_elem.get('number')
        coeff_value_str = coeff_elem.get('value')
        if coeff_number_str is None or coeff_value_str is None:
            raise ValueError("Coefficient element missing required attributes")
        coeff_number = int(coeff_number_str)
        coeff_value = parse_text_to_float(coeff_value_str)
        coeff_text = coeff_elem.get('text', '')
        
        coeff = PolynomialCoefficient(
            number=coeff_number,
            value=coeff_value,
            text=coeff_text
        )
        
        for term_elem in coeff_elem.findall('term'):
            weight_str = term_elem.get('weight')
            if weight_str is None:
                raise ValueError("Term element missing weight attribute")
            weight = parse_text_to_float(weight_str)
            
            term = PolynomialTerm(weight=weight)
            
            for disp_elem in term_elem.findall('displacement_diff'):
                atom_a_str = disp_elem.get('atom_a')
                atom_b_str = disp_elem.get('atom_b')
                direction = disp_elem.get('direction')
                power_str = disp_elem.get('power')
                
                if atom_a_str is None or atom_b_str is None or direction is None or power_str is None:
                    raise ValueError("Displacement element missing required attributes")
                
                atom_a = int(atom_a_str)
                atom_b = int(atom_b_str)
                power = int(power_str)
                
                cell_a_elem = disp_elem.find('cell_a')
                cell_b_elem = disp_elem.find('cell_b')
                
                cell_a = parse_text_to_int_array(cell_a_elem.text, 3) if cell_a_elem is not None and cell_a_elem.text is not None else [0, 0, 0]
                cell_b = parse_text_to_int_array(cell_b_elem.text, 3) if cell_b_elem is not None and cell_b_elem.text is not None else [0, 0, 0]
                
                term.displacements.append({
                    'atom_a': atom_a,
                    'atom_b': atom_b,
                    'direction': direction,
                    'power': power,
                    'cell_a': cell_a,
                    'cell_b': cell_b
                })
            
            for strain_elem in term_elem.findall('strain'):
                power_str = strain_elem.get('power')
                voigt_str = strain_elem.get('voigt')
                if power_str is None or voigt_str is None:
                    raise ValueError("Strain element missing required attributes")
                power = int(power_str)
                voigt = int(voigt_str)
                term.strains.append({'power': power, 'voigt': voigt})
            
            coeff.terms.append(term)
        
        coefficients.append(coeff)
    
    return coefficients


def format_fortran_float(value: float, width: int = 16) -> str:
    """
    Format float in Fortran scientific notation style.
    
    Matches Fortran format: "  -0.4278388107E+00"
    """
    if value == 0.0:
        return f"{' ' * (width - 14)}0.0000000000E+00"
    
    # Get exponent
    import math
    if value != 0:
        exp = int(math.floor(math.log10(abs(value))))
    else:
        exp = 0
    
    # Normalize mantissa
    mantissa = value / (10.0 ** exp)
    
    # Format: sign + mantissa + E + sign + exp
    sign = '' if value >= 0 else '-'
    mantissa_abs = abs(mantissa)
    
    # Fortran style: 0.XXXXXXXXXE+YY
    formatted = f"{sign}{mantissa_abs:.10f}E{exp:+03d}"
    
    # Pad to width
    padding = ' ' * (width - len(formatted))
    return padding + formatted


def write_coefficient_xml(filename: str, coefficients: List[PolynomialCoefficient]):
    """
    Write polynomial coefficients to XML file.
    
    Parameters
    ----------
    filename : str
        Output XML file path.
    coefficients : List[PolynomialCoefficient]
        List of coefficients to write.
    """
    root = ET.Element('Heff_definition')
    
    for coeff in coefficients:
        coeff_elem = ET.SubElement(root, 'coefficient')
        coeff_elem.set('number', str(coeff.number))
        # Use Fortran-style formatting for value
        coeff_elem.set('value', format_fortran_float(coeff.value))
        coeff_elem.set('text', coeff.text)
        
        for term in coeff.terms:
            term_elem = ET.SubElement(coeff_elem, 'term')
            term_elem.set('weight', f" {term.weight:.6f}")
            
            for disp in term.displacements:
                disp_elem = ET.SubElement(term_elem, 'displacement_diff')
                disp_elem.set('atom_a', str(disp['atom_a']))
                disp_elem.set('atom_b', str(disp['atom_b']))
                disp_elem.set('direction', disp['direction'])
                disp_elem.set('power', str(disp['power']))
                
                cell_a_elem = ET.SubElement(disp_elem, 'cell_a')
                cell_a_elem.text = f"{disp['cell_a'][0]} {disp['cell_a'][1]} {disp['cell_a'][2]}"
                
                cell_b_elem = ET.SubElement(disp_elem, 'cell_b')
                cell_b_elem.text = f"{disp['cell_b'][0]} {disp['cell_b'][1]} {disp['cell_b'][2]}"
            
            for strain in term.strains:
                strain_elem = ET.SubElement(term_elem, 'strain')
                strain_elem.set('power', f" {strain['power']}")
                strain_elem.set('voigt', f" {strain['voigt']}")
    
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    
    # Write with custom XML declaration to match Fortran format
    with open(filename, 'w') as f:
        f.write('<?xml version="1.0" ?>\n')
        tree.write(f, encoding='unicode')


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python xml_parser.py <coeffs.xml>")
        sys.exit(1)
    
    filename = sys.argv[1]
    print(f"Reading {filename}...")
    
    coeffs = read_coefficient_xml(filename)
    
    print(f"\nRead {len(coeffs)} coefficients:")
    for i, coeff in enumerate(coeffs[:5]):
        print(f"  {i+1}. {coeff}")
        if coeff.terms:
            print(f"     First term: {coeff.terms[0]}")
    
    if len(coeffs) > 5:
        print(f"  ... and {len(coeffs) - 5} more")
