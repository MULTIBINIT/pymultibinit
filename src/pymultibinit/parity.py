"""Order-independent structural comparison of MULTIBINIT coefficient bases.

This module is intentionally independent of coefficient values and text labels.  It
compares the polynomial structure emitted by different generators or read from
coefficient XML, including the signed symmetry-orbit terms that define one
coefficient.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import re

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from pymultibinit.pyeffpot.xml_parser import read_coefficient_xml
from pymultibinit.training import XmlBasisFunction


DisplacementFingerprint = tuple[str, int, int, tuple[int, int, int], int]
StrainFingerprint = tuple[int, int]
TermFingerprint = tuple[int, tuple[DisplacementFingerprint, ...], tuple[StrainFingerprint, ...]]
CanonicalIdentity = tuple[TermFingerprint, ...]


@dataclass(frozen=True)
class DuplicateIdentity:
    """One structural identity occurring more than once in either compared set."""

    identity: CanonicalIdentity
    reference_indices: tuple[int, ...]
    candidate_indices: tuple[int, ...]



@dataclass(frozen=True)
class FortranGenerationCount:
    """Candidate-generation stages reported for one Fortran anchor."""

    anchor_number: int
    label: str
    irreducible_pairs: int
    pair_combinations: int
    irreducible_combinations: int
    coefficients: int


@dataclass(frozen=True)
class BasisSetComparison:
    """Order-independent difference between a reference and candidate basis.

    ``missing`` are identities present in ``reference`` but absent from
    ``candidate``; ``extra`` are identities present only in ``candidate``.
    Identities themselves contain the complete normalized per-term detail.
    """

    missing: tuple[CanonicalIdentity, ...]
    extra: tuple[CanonicalIdentity, ...]
    duplicates: tuple[DuplicateIdentity, ...]

    @property
    def matches(self) -> bool:
        """Whether both sets contain precisely one instance of every identity."""
        return not self.missing and not self.extra and not self.duplicates


def _weight_sign(weight: object) -> int:
    value = float(weight)
    if value == 0.0:
        raise ValueError("A zero-weight term has no signed structural identity")
    return 1 if value > 0.0 else -1


def _cell3(value: object, name: str) -> tuple[int, int, int]:
    cell = tuple(int(component) for component in value)  # type: ignore[arg-type]
    if len(cell) != 3:
        raise ValueError(f"{name} must contain exactly three integers")
    return cell


def _term_fingerprint(term: Mapping[str, object]) -> TermFingerprint:
    """Normalize one XML term with its first displacement endpoint at cell zero."""
    displacements: list[DisplacementFingerprint] = []
    for raw_displacement in term.get("displacements", ()):  # type: ignore[union-attr]
        displacement = raw_displacement  # type: ignore[assignment]
        atom_a = int(displacement["atom_a"])
        atom_b = int(displacement["atom_b"])
        cell_a = _cell3(displacement.get("cell_a", (0, 0, 0)), "cell_a")
        cell_b = _cell3(displacement.get("cell_b", (0, 0, 0)), "cell_b")
        relative_cell_b = tuple(cell_b[axis] - cell_a[axis] for axis in range(3))
        direction = displacement["direction"]
        if isinstance(direction, (int, np.integer)):
            if int(direction) not in (0, 1, 2):
                raise ValueError(f"Unsupported displacement direction: {direction}")
            direction = "xyz"[int(direction)]
        direction_text = str(direction).lower()
        if direction_text not in {"x", "y", "z"}:
            raise ValueError(f"Unsupported displacement direction: {direction}")
        displacements.append(
            (direction_text, atom_a, atom_b, relative_cell_b, int(displacement["power"]))
        )

    strains: list[StrainFingerprint] = []
    for raw_strain in term.get("strains", ()):  # type: ignore[union-attr]
        strain = raw_strain  # type: ignore[assignment]
        strains.append((int(strain["voigt"]), int(strain["power"])))

    return (
        _weight_sign(term["weight"]),
        tuple(sorted(displacements)),
        tuple(sorted(strains)),
    )


def canonical_identity(basis_function: XmlBasisFunction | Mapping[str, object]) -> CanonicalIdentity:
    """Return the order-independent structural identity of a basis function.

    A term is represented by its normalized sign, sorted displacement factors,
    and sorted strain factors.  Displacement cells are shifted so every
    ``atom_a`` lies in cell ``(0, 0, 0)``.  Coefficient numbers, values and text
    labels are deliberately excluded.
    """
    terms = basis_function.terms if isinstance(basis_function, XmlBasisFunction) else basis_function["terms"]
    return tuple(sorted(_term_fingerprint(term) for term in terms))


def parse_fortran_coefficient_xml(filename: str | Path, *, populated_only: bool = False) -> tuple[XmlBasisFunction, ...]:
    """Read a MULTIBINIT coefficient XML into ``XmlBasisFunction`` objects.

    Atom indices are retained exactly as encoded in the XML.  The BaTiO3 parity
    fixture is zero-based, which is also the Python generator convention.
    ``populated_only`` filters the zero-term placeholder coefficients produced
    by the singular 919-column fit.
    """
    basis: list[XmlBasisFunction] = []
    for coefficient in read_coefficient_xml(str(filename)):
        terms = tuple(
            {
                "weight": float(term.weight),
                "displacements": tuple(
                    {
                        "atom_a": int(displacement["atom_a"]),
                        "atom_b": int(displacement["atom_b"]),
                        "direction": displacement["direction"],
                        "power": int(displacement["power"]),
                        "cell_a": _cell3(displacement.get("cell_a", (0, 0, 0)), "cell_a"),
                        "cell_b": _cell3(displacement.get("cell_b", (0, 0, 0)), "cell_b"),
                    }
                    for displacement in term.displacements
                ),
                "strains": tuple(
                    {"voigt": int(strain["voigt"]), "power": int(strain["power"])}
                    for strain in term.strains
                ),
            }
            for term in coefficient.terms
        )
        if populated_only and not terms:
            continue
        basis.append(
            XmlBasisFunction(
                number=int(coefficient.number),
                value=float(coefficient.value),
                text=str(coefficient.text),
                terms=terms,
            )
        )
    return tuple(basis)


def basis_anchor(basis_function: XmlBasisFunction | Mapping[str, object]) -> int | None:
    """Return the first normalized displacement's anchor atom, or ``None`` for strain-only terms."""
    identity = canonical_identity(basis_function)
    for _, displacements, _ in identity:
        if displacements:
            return displacements[0][1]
    return None


def irreducible_atom_classes(xred, symrel, tnons=None) -> dict[int, int]:
    """Map each atom to the minimum-index representative of its symmetry orbit."""
    from pymultibinit.pyeffpot.symmetry import build_atom_mapping

    positions = np.asarray(xred, dtype=float)
    operations = np.asarray(symrel, dtype=int)
    translations = np.zeros((len(operations), 3), dtype=float) if tnons is None else np.asarray(tnons, dtype=float)
    mapping = build_atom_mapping(positions, operations, translations)
    return {
        atom: min(int(mapping[3, isym, atom]) for isym in range(len(operations)))
        for atom in range(len(positions))
    }


def group_basis_by_anchor(
    basis: Iterable[XmlBasisFunction | Mapping[str, object]],
    *,
    xred,
    symrel,
    tnons=None,
) -> dict[int, tuple[XmlBasisFunction | Mapping[str, object], ...]]:
    """Group displacement basis functions by irreducible atom anchor.

    The anchor is the first displacement's normalized ``atom_a`` and then the
    symmetry-orbit representative.  Strain-only and zero-term entries have no
    displacement anchor and are omitted.
    """
    classes = irreducible_atom_classes(xred, symrel, tnons)
    grouped: dict[int, list[XmlBasisFunction | Mapping[str, object]]] = defaultdict(list)
    for basis_function in basis:
        anchor = basis_anchor(basis_function)
        if anchor is not None:
            grouped[classes[anchor]].append(basis_function)
    return {anchor: tuple(items) for anchor, items in sorted(grouped.items())}


def _identity_indices(basis: Sequence[XmlBasisFunction | Mapping[str, object]]) -> dict[CanonicalIdentity, tuple[int, ...]]:
    indices: dict[CanonicalIdentity, list[int]] = defaultdict(list)
    for index, basis_function in enumerate(basis):
        indices[canonical_identity(basis_function)].append(index)
    return {identity: tuple(found) for identity, found in indices.items()}


def compare_basis_sets(
    reference: Sequence[XmlBasisFunction | Mapping[str, object]],
    candidate: Sequence[XmlBasisFunction | Mapping[str, object]],
) -> BasisSetComparison:
    """Compare two bases by canonical structural identity.

    The comparison is insensitive to basis-function order, term order, factor
    order, cell_a translations and nonzero weight magnitudes.
    """
    reference_indices = _identity_indices(reference)
    candidate_indices = _identity_indices(candidate)
    reference_ids = set(reference_indices)
    candidate_ids = set(candidate_indices)
    duplicates = []
    for identity in sorted(reference_ids | candidate_ids):
        reference_occurrences = reference_indices.get(identity, ())
        candidate_occurrences = candidate_indices.get(identity, ())
        if len(reference_occurrences) > 1 or len(candidate_occurrences) > 1:
            duplicates.append(DuplicateIdentity(identity, reference_occurrences, candidate_occurrences))
    return BasisSetComparison(
        missing=tuple(sorted(reference_ids - candidate_ids)),
        extra=tuple(sorted(candidate_ids - reference_ids)),
        duplicates=tuple(duplicates),
    )


_FORTRAN_GENERATION_PATTERN = re.compile(
    r"The coefficients for the fit around atom\s+(?P<anchor>\d+):\s*(?P<label>\S+).*?"
    r"Number of irreducible pairs within cutoff:\s*(?P<pairs>\d+).*?"
    r"Number of combinations of irreducible pairs:\s*(?P<combinations>\d+).*?"
    r"(?P<irreducible>\d+)\s+irreducible combinations generated.*?"
    r"(?P<coefficients>\d+)\s+coefficients generated",
    re.DOTALL,
)


def parse_fortran_generation_counts(filename: str | Path) -> tuple[FortranGenerationCount, ...]:
    """Parse the per-anchor candidate stages printed by MULTIBINIT's fit log."""
    text = Path(filename).read_text(encoding="utf-8")
    return tuple(
        FortranGenerationCount(
            anchor_number=int(match["anchor"]),
            label=match["label"],
            irreducible_pairs=int(match["pairs"]),
            pair_combinations=int(match["combinations"]),
            irreducible_combinations=int(match["irreducible"]),
            coefficients=int(match["coefficients"]),
        )
        for match in _FORTRAN_GENERATION_PATTERN.finditer(text)
    )
