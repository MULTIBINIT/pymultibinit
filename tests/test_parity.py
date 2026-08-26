from pathlib import Path

import numpy as np

from pymultibinit.parity import (
    canonical_identity,
    compare_basis_sets,
    group_basis_by_anchor,
    parse_fortran_coefficient_xml,
    parse_fortran_generation_counts,
)
from pymultibinit.training import (
    FORTRAN_ANCHORED_GENERATOR_TAG,
    XmlBasisFunction,
    generate_fortran_anchored_basis,
    _compute_strain_symmetry_map,
    is_pure_strain_basis_function,
)


FIXTURE_DIR = (
    Path(__file__).resolve().parents[2]
    / "atomchain"
    / "examples"
    / "07_multibinit_workflow_batio3"
    / "batio3_multibinit_model"
    / "fortran_parity"
)
FIXTURE_XML = FIXTURE_DIR / "BaTiO3_current_iatom_minus2_ncoeff919_fortran_hist_coeffs.xml"
FIXTURE_LOG = FIXTURE_DIR / "multibinit_iatom_minus2_ncoeff919.stdout.log"


def _basis(number, terms):
    return XmlBasisFunction(number=number, value=0.0, text="ignored", terms=tuple(terms))


def _cubic_symrel():
    operations = []
    for permutation in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
        for signs in ((sx, sy, sz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)):
            matrix = np.zeros((3, 3), dtype=int)
            for row, column in enumerate(permutation):
                matrix[row, column] = signs[row]
            operations.append(matrix)
    return np.array(operations, dtype=int)


def _batio3_xred():
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
        ]
    )


def test_canonical_identity_order_independent():
    first_term = {
        "weight": 3.0,
        "displacements": (
            {"direction": "z", "atom_a": 1, "atom_b": 2, "cell_a": (0, 0, 0), "cell_b": (0, 0, 1), "power": 1},
            {"direction": "x", "atom_a": 1, "atom_b": 3, "cell_a": (0, 0, 0), "cell_b": (1, 0, 0), "power": 2},
        ),
        "strains": ({"voigt": 3, "power": 1}, {"voigt": 1, "power": 2}),
    }
    second_term = {
        "weight": -4.0,
        "displacements": (
            {"direction": "y", "atom_a": 1, "atom_b": 4, "cell_a": (0, 0, 0), "cell_b": (0, 1, 0), "power": 3},
        ),
        "strains": (),
    }
    shuffled_and_translated = {
        "weight": 3.0,
        "displacements": tuple(
            {
                **displacement,
                "cell_a": tuple(value + 2 for value in displacement["cell_a"]),
                "cell_b": tuple(value + 2 for value in displacement["cell_b"]),
            }
            for displacement in reversed(first_term["displacements"])
        ),
        "strains": tuple(reversed(first_term["strains"])),
    }

    original = _basis(1, (first_term, second_term))
    shuffled = _basis(2, (second_term, shuffled_and_translated))

    assert canonical_identity(original) == canonical_identity(shuffled)
    assert compare_basis_sets((original,), (shuffled,)).matches


def test_signed_weights_normalize_to_plus_or_minus_one():
    term = {
        "displacements": (
            {"direction": "x", "atom_a": 0, "atom_b": 1, "cell_a": (0, 0, 0), "cell_b": (0, 0, 0), "power": 2},
        ),
        "strains": (),
    }
    positive_large = _basis(1, ({**term, "weight": 7.5},))
    positive_unit = _basis(2, ({**term, "weight": 1.0},))
    negative_large = _basis(3, ({**term, "weight": -2.0},))

    assert canonical_identity(positive_large) == canonical_identity(positive_unit)
    assert canonical_identity(positive_large) != canonical_identity(negative_large)


def test_fortran_fixture_parses_populated_structural_oracle_and_log_counts():
    fixture_basis = parse_fortran_coefficient_xml(FIXTURE_XML)
    populated = parse_fortran_coefficient_xml(FIXTURE_XML, populated_only=True)
    grouped = group_basis_by_anchor(populated, xred=_batio3_xred(), symrel=_cubic_symrel())
    counts = parse_fortran_generation_counts(FIXTURE_LOG)

    assert len(fixture_basis) == 919
    assert len(populated) == 78
    assert {anchor: len(entries) for anchor, entries in grouped.items()} == {0: 22, 1: 24, 2: 32}
    assert [(count.irreducible_pairs, count.pair_combinations, count.irreducible_combinations, count.coefficients) for count in counts] == [
        (3, 25, 314, 308),
        (3, 25, 273, 246),
        (6, 182, 411, 365),
    ]


def test_v2_pure_strain_orbits_are_invariant_and_even_by_default():
    symrel = _cubic_symrel()
    basis = generate_fortran_anchored_basis(
        np.array([[0.0, 0.0, 0.0]]),
        np.array([[0.0, 0.0, 0.0]]),
        1.0,
        symrel,
        power_range=(2, 2),
        include_pure_strain=True,
        max_strain_power=3,
    )
    pure = [item for item in basis if is_pure_strain_basis_function(item)]
    state = np.array([0.13, -0.21, 0.31, 0.17, -0.19, 0.23])

    def evaluate(item, strain):
        return sum(
            term["weight"] * np.prod(
                [strain[int(factor["voigt"]) - 1] ** int(factor["power"]) for factor in term["strains"]]
            )
            for term in item.terms
        )

    for item in pure:
        assert sum(factor["power"] for factor in item.terms[0]["strains"]) % 2 == 0
        reference = evaluate(item, state)
        for operation in symrel:
            transformed = np.zeros(6)
            for old, mapped in enumerate(_compute_strain_symmetry_map([operation])[0]):
                transformed[abs(mapped) - 1] = np.sign(mapped) * state[old]
            assert np.isclose(evaluate(item, transformed), reference, rtol=1e-12, atol=1e-12)


def test_v2_expanded_nbody_and_fingerprint():
    lattice = np.eye(3) * 7.49649813
    xred = _batio3_xred()
    basis = generate_fortran_anchored_basis(
        xred @ lattice.T,
        xred,
        8.0,
        _cubic_symrel(),
        rprimd=lattice,
        power_range=(3, 3),
        atom_types=(0,),
        max_nbody={3: 1},
    )

    assert FORTRAN_ANCHORED_GENERATOR_TAG == "fortran_anchored_v2"
    assert basis
    assert all(len(term["displacements"]) <= 1 for item in basis for term in item.terms)




def test_v2_combined_displacement_strain_orbits_are_invariant():
    lattice = np.eye(3) * 7.49649813
    xred = _batio3_xred()
    symrel = _cubic_symrel()
    basis = generate_fortran_anchored_basis(
        xred @ lattice.T,
        xred,
        8.0,
        symrel,
        rprimd=lattice,
        power_range=(3, 3),
        atom_types=(0,),
        include_strain_coupling=True,
        max_strain_power=1,
    )
    state_u = np.random.default_rng(2).normal(size=(len(xred), 3))
    state_eta = np.array([0.13, -0.21, 0.31, 0.17, -0.19, 0.23])

    def evaluate(item, displacement, strain):
        total = 0.0
        for term in item.terms:
            value = float(term["weight"])
            for factor in term["displacements"]:
                direction = "xyz".index(factor["direction"])
                value *= (
                    displacement[factor["atom_b"], direction] - displacement[factor["atom_a"], direction]
                ) ** factor["power"]
            for factor in term["strains"]:
                value *= strain[factor["voigt"] - 1] ** factor["power"]
            total += value
        return total

    def transform(operation):
        transformed_u = np.empty_like(state_u)
        for atom, position in enumerate(xred):
            target = operation @ position
            target -= np.floor(target)
            mapped_atom = np.argmin(np.linalg.norm(((xred - target + 0.5) % 1.0) - 0.5, axis=1))
            transformed_u[mapped_atom] = operation @ state_u[atom]
        tensor = np.array(
            [
                [state_eta[0], state_eta[5], state_eta[4]],
                [state_eta[5], state_eta[1], state_eta[3]],
                [state_eta[4], state_eta[3], state_eta[2]],
            ]
        )
        tensor = operation @ tensor @ operation.T
        transformed_eta = np.array(
            [tensor[0, 0], tensor[1, 1], tensor[2, 2], tensor[1, 2], tensor[0, 2], tensor[0, 1]]
        )
        return transformed_u, transformed_eta

    for item in basis:
        reference = evaluate(item, state_u, state_eta)
        for operation in symrel:
            transformed_u, transformed_eta = transform(operation)
            assert np.isclose(
                evaluate(item, transformed_u, transformed_eta), reference, rtol=1e-12, atol=1e-12
            )

def test_fixture_identity_comparison_reports_per_term_duplicates():
    populated = parse_fortran_coefficient_xml(FIXTURE_XML, populated_only=True)
    comparison = compare_basis_sets(populated, tuple(reversed(populated)) + (populated[0],))

    assert not comparison.missing
    assert not comparison.extra
    assert len(comparison.duplicates) == 1
    duplicate = comparison.duplicates[0]
    assert duplicate.reference_indices == (0,)
    assert duplicate.candidate_indices == (77, 78)
    assert duplicate.identity == canonical_identity(populated[0])


def test_python_v2_basis_matches_populated_fortran_fixture_identities():
    lattice = np.eye(3) * 7.49649813
    xred = _batio3_xred()
    xcart = xred @ lattice.T
    symrel = _cubic_symrel()
    fixture_groups = group_basis_by_anchor(
        parse_fortran_coefficient_xml(FIXTURE_XML, populated_only=True),
        xred=xred,
        symrel=symrel,
    )
    python_groups = group_basis_by_anchor(
        generate_fortran_anchored_basis(
            xcart,
            xred,
            8.0,
            symrel,
            ncell=(1, 1, 1),
            rprimd=lattice,
            atom_types=(0, 1, 2),
            include_strain_coupling=False,
            include_pure_strain=False,
            max_nbody=999,
        ),
        xred=xred,
        symrel=symrel,
    )
    comparisons = {
        anchor: compare_basis_sets(fixture_groups[anchor], python_groups[anchor])
        for anchor in (0, 1, 2)
    }

    assert [len(python_groups[anchor]) for anchor in (0, 1, 2)] == [308, 246, 384]
    assert [
        (len(comparisons[anchor].missing), len(comparisons[anchor].extra))
        for anchor in (0, 1, 2)
    ] == [(0, 286), (0, 222), (0, 352)]
    assert len({canonical_identity(item) for entries in python_groups.values() for item in entries}) == sum(
        len(entries) for entries in python_groups.values()
    )
