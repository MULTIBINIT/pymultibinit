"""
ASE Calculator interface for MULTIBINIT effective potential.

Allows seamless integration with the Atomic Simulation Environment (ASE)
for structure optimization, molecular dynamics, phonon calculations, etc.
"""
import multiprocessing
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from ase.calculators.calculator import Calculator, all_changes
import numpy as np

from .potential import MultibinitPotential


@dataclass
class Contributions:
    """Energy / force / stress decomposition of the effective potential.

    Each term's arrays are in ASE units: energy in eV, forces in eV/Angstrom,
    stress in eV/Angstrom^3 in Voigt order ``[xx, yy, zz, yz, xz, xy]``.
    Summed over :attr:`terms` the arrays reproduce ``get_potential_energy()``,
    ``get_forces()`` and ``get_stress()`` exactly.

    Attributes
    ----------
    energy : dict[str, float]
        Per-term energy (eV).
    forces : dict[str, np.ndarray]
        Per-term forces, shape (natom, 3), in the input atom order (eV/Angstrom).
    stress : dict[str, np.ndarray]
        Per-term stress, shape (6,) Voigt (eV/Angstrom^3).
    terms : tuple[str, ...]
        Ordered contribution names.

    Examples
    --------
    >>> contrib = calc.get_contributions(atoms)
    >>> contrib.energy["dipdip"]          # dipole-dipole energy
    >>> contrib.forces["harmonic_local"]  # local harmonic IFC forces
    >>> contrib.total_energy() == atoms.get_potential_energy()
    """
    energy: Dict[str, float] = field(default_factory=dict)
    forces: Dict[str, np.ndarray] = field(default_factory=dict)
    stress: Dict[str, np.ndarray] = field(default_factory=dict)
    terms: Tuple[str, ...] = ()

    @classmethod
    def from_terms(cls, terms: Dict[str, Tuple[float, np.ndarray, np.ndarray]]) -> "Contributions":
        """Build from a ``{name: (energy, forces, stress)}`` mapping."""
        energy: Dict[str, float] = {}
        forces: Dict[str, np.ndarray] = {}
        stress: Dict[str, np.ndarray] = {}
        for name, (e, f, s) in terms.items():
            energy[name] = float(e)
            forces[name] = np.asarray(f, dtype=float)
            stress[name] = np.asarray(s, dtype=float)
        return cls(energy=energy, forces=forces, stress=stress,
                   terms=tuple(terms.keys()))

    def total_energy(self) -> float:
        """Sum of all per-term energies (eV)."""
        return float(sum(self.energy.values()))

    def total_forces(self) -> np.ndarray:
        """Sum of all per-term forces, shape (natom, 3) (eV/Angstrom)."""
        if not self.forces:
            raise ValueError("no force contributions available")
        return sum(self.forces.values())

    def total_stress(self) -> np.ndarray:
        """Sum of all per-term stresses, shape (6,) Voigt (eV/Angstrom^3)."""
        if not self.stress:
            raise ValueError("no stress contributions available")
        return sum(self.stress.values())


def _abi_spawned_worker(
    connection,
    abi_file: str,
    lib_path: Optional[str],
    auto_match_atoms: bool,
    match_tolerance: float,
) -> None:
    potential = None
    try:
        potential = MultibinitPotential.from_abi(
            abi_file=abi_file,
            lib_path=lib_path,
            auto_match_atoms=auto_match_atoms,
            match_tolerance=match_tolerance,
        )
        connection.send(("ok", None))
        while True:
            request = connection.recv()
            if request is None:
                potential_to_free = potential
                potential = None
                potential_to_free.free()
                return
            operation = request[0]
            if operation == "evaluate":
                result = potential.evaluate(request[1], request[2])
            elif operation == "reference":
                result = potential.export_supercell_to_ase()
            else:
                raise ValueError(f"unknown spawned potential operation: {operation}")
            connection.send(("ok", result))
    except EOFError:
        return
    except BaseException as error:
        try:
            connection.send(("error", f"{type(error).__name__}: {error}"))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        if potential is not None:
            potential.free()
        connection.close()


class _SpawnedPotential:
    def __init__(
        self,
        abi_file: str,
        lib_path: Optional[str],
        *,
        auto_match_atoms: bool,
        match_tolerance: float,
    ):
        context = multiprocessing.get_context("spawn")
        self._connection, child_connection = context.Pipe()
        self._process = context.Process(
            target=_abi_spawned_worker,
            args=(
                child_connection,
                abi_file,
                lib_path,
                auto_match_atoms,
                match_tolerance,
            ),
        )
        self._closed = False
        self._started = False
        try:
            self._process.start()
            self._started = True
        except BaseException:
            child_connection.close()
            self._reap()
            raise
        child_connection.close()
        try:
            self._receive("initialization")
        except BaseException:
            self._reap()
            raise

    def _worker_exit_error(self, operation: str) -> RuntimeError:
        return RuntimeError(
            "Spawned libabinit worker exited during "
            f"{operation} with exit code {self._process.exitcode}"
        )

    def _receive(self, operation: str):
        try:
            response = self._connection.recv()
        except EOFError as error:
            raise self._worker_exit_error(operation) from error
        if response[0] == "ok":
            return response[1]
        if response[0] == "error":
            raise RuntimeError(
                f"Spawned libabinit worker failed during {operation}: {response[1]}"
            )
        raise RuntimeError(f"unknown spawned potential response: {response!r}")

    def _request(self, request):
        operation = request[0]
        if self._closed:
            raise RuntimeError("Spawned MultibinitPotential is closed")
        try:
            if not self._process.is_alive():
                raise self._worker_exit_error(operation)
            self._connection.send(request)
            return self._receive(operation)
        except (BrokenPipeError, EOFError, OSError) as error:
            self._reap()
            raise self._worker_exit_error(operation) from error
        except BaseException:
            self._reap()
            raise

    def evaluate(self, positions: np.ndarray, lattice: np.ndarray):
        return self._request(("evaluate", positions, lattice))

    def export_supercell_to_ase(self):
        return self._request(("reference",))

    def _reap(self):
        self._closed = True
        try:
            self._connection.close()
        except OSError:
            pass
        if self._started:
            self._process.join(5)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(5)
                if self._process.is_alive():
                    self._process.kill()
                    self._process.join(5)

    def free(self):
        if self._closed:
            return
        try:
            if self._process.is_alive():
                self._connection.send(None)
        except (BrokenPipeError, EOFError, OSError):
            pass
        finally:
            self._reap()


class MultibinitCalculator(Calculator):
    """
    ASE calculator for ABINIT's MULTIBINIT effective potential.

    This calculator wraps the MULTIBINIT C API and provides a standard ASE
    interface for energy, force, and stress calculations.

    Properties implemented: energy, forces, stress

    Two backends are available:
        - 'cffi'   : Requires libabinit.so/dylib (Fortran). Use from_abi(),
                     from_params(), or from_config_file().
        - 'pyeffpot': Pure Python, no Fortran dependency. Use from_pyeffpot().
                       Reads a DDB file, an optional XML coefficient file, and
                       activates dipole-dipole / supercell via keyword args.

    Example:
        >>> from ase import Atoms
        >>> from pymultibinit import MultibinitCalculator
        >>>
        >>> # --- CFFI (Fortran) backend ---
        >>> # Using .abi file
        >>> calc = MultibinitCalculator.from_abi("input.abi")
        >>> atoms.calc = calc
        >>> energy = atoms.get_potential_energy()
        >>>
        >>> # Using parameters
        >>> calc = MultibinitCalculator.from_params(
        ...     ddb_file="system_DDB",
        ...     sys_file="system.xml",
        ...     ncell=(2, 2, 2)
        ... )
        >>> atoms.calc = calc
        >>>
        >>> # --- Pure Python (pyeffpot) backend, no Fortran needed ---
        >>> calc = MultibinitCalculator.from_pyeffpot(
        ...     ddb_file="system.DDB",
        ...     xml_file="coeffs.xml",   # optional, None to skip
        ...     ncell=(4, 4, 4),         # supercell size
        ...     dipdip=True,             # dipole-dipole (long-range Coulomb)
        ... )
        >>> atoms.calc = calc
    """

    implemented_properties = ['energy', 'forces', 'stress']

    def __init__(self, potential: MultibinitPotential | _SpawnedPotential, **kwargs):
        """
        Initialize the calculator with an existing potential.

        Args:
            potential: Initialized MultibinitPotential instance
            **kwargs: Additional arguments for ASE Calculator
        """
        super().__init__(**kwargs)
        self.potential = potential
        self._closed = False

    @classmethod
    def from_abi(cls, abi_file: str, lib_path: Optional[str] = None,
                 *, auto_match_atoms: bool = True, match_tolerance: float = 0.1,
                 **kwargs) -> 'MultibinitCalculator':
        """
        Create calculator from a .abi input file.

        Args:
            abi_file: Path to the .abi input file
            lib_path: Path to libabinit.so/dylib (optional)
            auto_match_atoms: Automatically reorder input atoms to match the
                MULTIBINIT reference on the first evaluation.
            match_tolerance: Atom-matching tolerance in Angstrom.
            **kwargs: Additional arguments for ASE Calculator

        Returns:
            Initialized MultibinitCalculator instance
        """
        potential = MultibinitPotential.from_abi(
            abi_file=abi_file,
            lib_path=lib_path,
            auto_match_atoms=auto_match_atoms,
            match_tolerance=match_tolerance,
        )
        return cls(potential=potential, **kwargs)

    @classmethod
    def from_abi_spawned(
        cls,
        abi_file: str,
        lib_path: Optional[str] = None,
        *,
        auto_match_atoms: bool = True,
        match_tolerance: float = 0.1,
        **kwargs,
    ) -> 'MultibinitCalculator':
        """Create a CFFI calculator whose native state lives in a child process."""
        potential = _SpawnedPotential(
            abi_file,
            lib_path,
            auto_match_atoms=auto_match_atoms,
            match_tolerance=match_tolerance,
        )
        return cls(potential=potential, **kwargs)

    @classmethod
    def from_params(cls, ddb_file: str, sys_file: str = "", coeff_file: str = "",
                   ncell: Tuple[int, int, int] = (1, 1, 1),
                   ngqpt: Tuple[int, int, int] = (1, 1, 1),
                   dipdip: int = 1,
                   lib_path: Optional[str] = None,
                   **kwargs) -> 'MultibinitCalculator':
        """
        Create calculator from direct parameters (no .abi file).

        Args:
            ddb_file: Path to DDB file
            sys_file: Path to system XML file (optional)
            coeff_file: Path to coefficient XML file (optional)
            ncell: Supercell dimensions [nx, ny, nz]
            ngqpt: q-point grid [nqx, nqy, nqz]
            dipdip: Dipole-dipole interactions (0=off, 1=on)
            lib_path: Path to libabinit.so/dylib (optional)
            **kwargs: Additional arguments for ASE Calculator

        Returns:
            Initialized MultibinitCalculator instance
        """
        potential = MultibinitPotential.from_params(
            ddb_file=ddb_file,
            sys_file=sys_file,
            coeff_file=coeff_file,
            ncell=ncell,
            ngqpt=ngqpt,
            dipdip=dipdip,
            lib_path=lib_path
        )
        return cls(potential=potential, **kwargs)

    @classmethod
    def from_pyeffpot(cls, ddb_file: str, xml_file: Optional[str] = None,
                      ncell: Tuple[int, int, int] = (4, 4, 4),
                      dipdip: bool = True,
                      asr: bool = True,
                      auto_match_atoms: bool = True,
                      match_tolerance: float = 0.1,
                      reference_stress_ha_bohr3: Optional[np.ndarray] = None,
                      **kwargs) -> 'MultibinitCalculator':
        """
        Create a calculator using the **pure Python** backend (no Fortran/libabinit required).

        This is the recommended path when libabinit.so is not available.
        It parses the DDB file directly in Python, optionally overlays anharmonic
        coefficients from an XML file, builds the supercell, applies the
        dipole-dipole long-range correction and the acoustic sum rule (ASR),
        and returns a fully functional ASE calculator.

        Args:
            ddb_file: Path to the ABINIT DDB (derivative database) file.
            xml_file: Path to an XML coefficient file with anharmonic terms.
                Optional - pass None or omit to use the harmonic DDB-only model.
            ncell: Supercell dimensions (nx, ny, nz). Defaults to (4, 4, 4).
            dipdip: If True (default), activate dipole-dipole (long-range
                Coulomb) correction using Born effective charges and the
                dielectric tensor read from the DDB. Set False to disable.
            asr: If True (default), enforce the acoustic sum rule on the
                interatomic force constants.
            auto_match_atoms: If True, automatically reorder input atoms to
                match the MULTIBINIT reference on the first evaluate() call.
            match_tolerance: Tolerance (Angstrom) for atom matching.
            reference_stress_ha_bohr3: Optional (3,3) reference stress tensor
                in Hartree/Bohr^3 to subtract (used for parity tests).
            **kwargs: Additional arguments forwarded to the ASE Calculator base.

        Returns:
            Initialized MultibinitCalculator instance backed by pyeffpot.

        Example:
            >>> from ase import Atoms
            >>> from pymultibinit import MultibinitCalculator
            >>>
            >>> # Pure-Python ASE calculator, no libabinit needed
            >>> calc = MultibinitCalculator.from_pyeffpot(
            ...     ddb_file="BTO.DDB",
            ...     xml_file="model.xml",   # optional
            ...     ncell=(2, 2, 2),
            ...     dipdip=True,
            ... )
            >>> atoms = Atoms(...)
            >>> atoms.calc = calc
            >>> energy = atoms.get_potential_energy()   # eV
            >>> forces = atoms.get_forces()             # eV/Angstrom
            >>> stress = atoms.get_stress()             # eV/Angstrom^3
        """
        potential = MultibinitPotential.from_pyeffpot(
            ddb_file=ddb_file,
            xml_file=xml_file,
            ncell=ncell,
            dipdip=dipdip,
            asr=asr,
            auto_match_atoms=auto_match_atoms,
            match_tolerance=match_tolerance,
            reference_stress_ha_bohr3=reference_stress_ha_bohr3,
        )
        return cls(potential=potential, **kwargs)

    @classmethod
    def from_config_file(cls, config_file: str, **kwargs) -> 'MultibinitCalculator':
        """
        Create calculator from a configuration file.

        The configuration file can specify either:
        1. abi_file: Path to .abi input file
        2. ddb_file + sys_file: Direct initialization

        Simple format example:
            ```
            ddb_file: system_DDB
            sys_file: system.xml
            ncell: 2 2 2
            ```

        INI-like format example:
            ```
            [files]
            ddb_file = system_DDB
            sys_file = system.xml

            [parameters]
            ncell = 2 2 2
            ngqpt = 4 4 4
            ```

        Args:
            config_file: Path to the configuration file
            **kwargs: Additional arguments for ASE Calculator

        Returns:
            Initialized MultibinitCalculator instance

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If required parameters are missing or invalid
        """
        potential = MultibinitPotential.from_config_file(config_file)
        return cls(potential=potential, **kwargs)

    def get_reference_atoms(self):
        return self.potential.export_supercell_to_ase()

    def get_contributions(self, atoms=None) -> "Contributions":
        """
        Return the energy/force/stress decomposition by physical term.

        This is the ASE-compatible entry point for inspecting how the total
        energy, forces, and stress split into contributions. The returned
        :class:`Contributions` object exposes per-term dicts plus
        ``total_energy()`` / ``total_forces()`` / ``total_stress()`` that
        reproduce the standard ASE getters exactly.

        Terms: ``reference``, ``harmonic_local`` (local harmonic IFCs),
        ``dipdip`` (dipole-dipole / long-range Coulomb), ``anharmonic``,
        and, when the model contains them, ``elastic`` and
        ``strain_coupling``.

        Parameters
        ----------
        atoms : ase.Atoms, optional
            Structure to evaluate. If omitted, the calculator's currently
            attached atoms are used.

        Returns
        -------
        Contributions

        Raises
        ------
        NotImplementedError
            If the backend does not support decomposition (CFFI / Fortran or
            spawned-process backends). Only the pure-Python ``pyeffpot``
            backend exposes per-term contributions.

        Example
        -------
        >>> calc = MultibinitCalculator.from_pyeffpot("BTO.DDB", "model.xml")
        >>> atoms.calc = calc
        >>> contrib = calc.get_contributions()
        >>> contrib.energy["dipdip"]            # dipole-dipole energy (eV)
        >>> contrib.forces["anharmonic"]        # anharmonic forces
        >>> contrib.total_forces() - atoms.get_forces()  # ~0
        """
        if self._closed:
            raise RuntimeError("MultibinitCalculator is closed")

        evaluate_contrib = getattr(self.potential, "evaluate_contributions", None)
        if evaluate_contrib is None:
            raise NotImplementedError(
                "Energy decomposition is only supported by the pure-Python "
                "(pyeffpot) backend. The CFFI (Fortran) and spawned-process "
                "backends do not expose per-term contributions."
            )

        if atoms is not None:
            self.atoms = atoms
        if self.atoms is None:
            raise RuntimeError("MultibinitCalculator requires ASE atoms")

        positions = self.atoms.get_positions()
        cell = self.atoms.get_cell().array
        raw = evaluate_contrib(positions, cell)
        contributions = Contributions.from_terms(raw)
        self.results['contributions'] = contributions
        return contributions

    def close(self):
        if not self._closed:
            self.potential.free()
            self.results.clear()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        """
        Calculate properties for the given atoms object.

        Args:
            atoms: ASE Atoms object (if None, uses self.atoms)
            properties: List of properties to calculate
            system_changes: List of changed properties since last calculation
        """
        if self._closed:
            raise RuntimeError("MultibinitCalculator is closed")

        super().calculate(atoms, properties, system_changes)

        if self.atoms is None:
            raise RuntimeError("MultibinitCalculator requires ASE atoms")

        # Get positions and cell from atoms (in Angstrom)
        positions = self.atoms.get_positions()  # (natom, 3) in Angstrom
        cell = self.atoms.get_cell().array      # (3, 3) in Angstrom

        # Evaluate using potential (handles unit conversion internally)
        energy, forces, stress = self.potential.evaluate(positions, cell)

        # Store results in ASE format
        self.results['energy'] = energy  # eV
        self.results['forces'] = forces  # eV/Angstrom

        # ASE stress order is [xx, yy, zz, yz, xz, xy] in eV/Angstrom^3.
        self.results['stress'] = stress  # eV/Angstrom^3

    def get_analytic_blocks(self, atoms=None):
        """Exact second-derivative blocks for an ASE Atoms object.

        Thin ASE surface over
        :meth:`MultibinitPotential.analytic_blocks`; returns a
        :class:`~pymultibinit.pyeffpot.second_derivatives.HessianBlocks`
        in eV/Angstrom units with atom rows in the Atoms (input) order:
        ``ifc`` (3N,3N) eV/A^2, ``elastic_fixed_u`` (6,6) eV,
        ``coupling`` (6,3N) eV/A, ``forces`` (N,3) eV/A,
        ``strain_voigt`` (6,). Requires the pyeffpot backend.
        """
        if self._closed:
            raise RuntimeError("MultibinitCalculator is closed")
        if atoms is None:
            atoms = self.atoms
        if atoms is None:
            raise RuntimeError("MultibinitCalculator requires ASE atoms")
        positions = atoms.get_positions()
        cell = atoms.get_cell().array
        return self.potential.analytic_blocks(positions, cell)
