"""
pymultibinit: Python interface to MULTIBINIT effective potential.

This package provides Python bindings to ABINIT's MULTIBINIT library for
computing energies, forces, and stresses using effective potentials derived
from DFT calculations.

Main classes:
    - MultibinitPotential: High-level potential interface
    - MultibinitCalculator: ASE calculator interface
    - MultibinitWrapperCFFI: Low-level CFFI wrapper (advanced users)

Example:
    >>> from pymultibinit import MultibinitCalculator
    >>> from ase import Atoms
    >>>
    >>> # Create calculator from .abi file
    >>> calc = MultibinitCalculator.from_abi("input.abi")
    >>> atoms.calc = calc
    >>> energy = atoms.get_potential_energy()
    >>> forces = atoms.get_forces()

    >>> # Or from parameters
    >>> calc = MultibinitCalculator.from_params(
    ...     ddb_file="system_DDB",
    ...     sys_file="system.xml",
    ...     ncell=(2, 2, 2)
    ... )
"""

from .potential import MultibinitPotential
from .calculator import MultibinitCalculator, Contributions
from .wrapper_cffi import MultibinitWrapperCFFI
from .pyeffpot import PhonopyDdbExportResult, write_phonopy_from_ddb

# Atom matching utilities
from . import atom_matching

# Configuration file support
from .config import MultibinitConfig

# Binary-based model building support
from .features import (
    evaluate_basis_features_vectorized,
    evaluate_basis_features_jax,
    evaluate_basis_features_auto,
    compile_basis,
    compile_term,
    CompiledTerm,
)
from .training import (
    MultibinitTrainingResult,
    FitDiagnostics,
    FitFeatureMatrices,
    FORTRAN_ANCHORED_GENERATOR_TAG,
    FortranPairList,
    GreedySelectionResult,
    GoalFunctionComponents,
    LinearFitResult,
    MonomialKey,
    PairKey,
    PythonFitConfig,
    PythonFitResult,
    TrainingDataset,
    TrainingFrame,
    XmlBasisFunction,
    basis_to_coefficients,
    build_factor_action_map,
    build_training_dataset,
    canonicalize_monomial_orbit,
    compute_goal_function,
    count_fortran_displacement_coefficients,
    count_fortran_irreducible_pair_combinations,
    displacement_pair_diagnostics,
    evaluate_basis_features,
    fit_multibinit_model_python,
    generate_displacement_basis,
    generate_fortran_anchored_basis,
    generate_fortran_displacement_combination_keys,
    generate_fortran_pair_list,
    load_xml_basis,
    normalize_pair_key,
    read_hist_frames,
    select_greedy_coefficients,
    train_multibinit_model,
    with_fortran_text_labels,
    write_fitted_xml,
)

__version__ = "0.3.12"

__all__ = [
    "CompiledTerm",
    "compile_basis",
    "compile_term",
    "evaluate_basis_features_auto",
    "evaluate_basis_features_jax",
    "evaluate_basis_features_vectorized",
    "MultibinitPotential",
    "MultibinitCalculator",
    "MultibinitWrapperCFFI",
    "MultibinitConfig",
    "PhonopyDdbExportResult",
    "MultibinitTrainingResult",
    "FitDiagnostics",
    "FitFeatureMatrices",
    "FortranPairList",
    "GreedySelectionResult",
    "GoalFunctionComponents",
    "LinearFitResult",
    "MonomialKey",
    "PairKey",
    "PythonFitConfig",
    "PythonFitResult",
    "TrainingDataset",
    "FORTRAN_ANCHORED_GENERATOR_TAG",
    "TrainingFrame",
    "XmlBasisFunction",
    "atom_matching",
    "basis_to_coefficients",
    "build_factor_action_map",
    "build_training_dataset",
    "canonicalize_monomial_orbit",
    "compute_goal_function",
    "count_fortran_displacement_coefficients",
    "count_fortran_irreducible_pair_combinations",
    "displacement_pair_diagnostics",
    "evaluate_basis_features",
    "fit_multibinit_model_python",
    "generate_displacement_basis",
    "generate_fortran_anchored_basis",
    "generate_fortran_displacement_combination_keys",
    "generate_fortran_pair_list",
    "load_xml_basis",
    "normalize_pair_key",
    "read_hist_frames",
    "select_greedy_coefficients",
    "train_multibinit_model",
    "with_fortran_text_labels",
    "write_phonopy_from_ddb",
    "write_fitted_xml",
    "Contributions",
]
