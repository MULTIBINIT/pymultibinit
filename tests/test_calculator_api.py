import inspect

from ase import Atoms
import numpy as np
import pytest

import pymultibinit.calculator as calculator_module
from pymultibinit import MultibinitCalculator
from pymultibinit.potential import MultibinitPotential


class _FakePotential(MultibinitPotential):
    def __init__(self):
        self.reference_atoms = Atoms()
        self.export_calls = 0
        self.free_calls = 0

    def export_supercell_to_ase(self) -> Atoms:
        self.export_calls += 1
        return self.reference_atoms

    def free(self):
        self.free_calls += 1


class _FakeSpawnConnection:
    def __init__(self, responses=()):
        self.responses = list(responses)
        self.sent = []
        self.closed = False

    def send(self, request):
        self.sent.append(request)

    def recv(self):
        return self.responses.pop(0)

    def close(self):
        self.closed = True


class _FakeSpawnProcess:
    def __init__(self, target, args):
        self.target = target
        self.args = args
        self.exitcode: int | None = None
        self.started = False
        self.terminated = False
        self._alive = True

    def start(self):
        self.started = True

    def is_alive(self):
        return self._alive

    def join(self, _timeout=None):
        self._alive = False

    def terminate(self):
        self.terminated = True
        self._alive = False


class _FakeSpawnContext:
    def __init__(self, responses):
        self.parent = _FakeSpawnConnection(responses)
        self.child = _FakeSpawnConnection()
        self.process = _FakeSpawnProcess(None, ())

    def Pipe(self):
        return self.parent, self.child

    def Process(self, target, args):
        self.process = _FakeSpawnProcess(target, args)
        return self.process


def test_get_reference_atoms_delegates_to_supercell_exporter():
    potential = _FakePotential()
    calculator = MultibinitCalculator(potential)

    assert calculator.get_reference_atoms() is potential.reference_atoms
    assert potential.export_calls == 1


def test_close_is_idempotent():
    potential = _FakePotential()
    calculator = MultibinitCalculator(potential)

    calculator.close()
    calculator.close()

    assert potential.free_calls == 1


def test_context_manager_closes_calculator_once():
    potential = _FakePotential()
    calculator = MultibinitCalculator(potential)

    with calculator as managed:
        assert managed is calculator

    calculator.close()
    assert potential.free_calls == 1


def test_calculate_after_close_raises_without_evaluating():
    potential = _FakePotential()
    calculator = MultibinitCalculator(potential)
    calculator.close()

    with pytest.raises(RuntimeError, match="closed"):
        calculator.calculate(Atoms("H", positions=[[0.0, 0.0, 0.0]]))


@pytest.mark.parametrize("getter", ("get_potential_energy", "get_forces", "get_stress"))
def test_ase_getters_after_close_do_not_return_cached_results(getter):
    potential = _FakePotential()
    calculator = MultibinitCalculator(potential)
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    atoms.calc = calculator
    calculator.atoms = atoms.copy()
    calculator.results.update(
        energy=0.0,
        forces=[[0.0, 0.0, 0.0]],
        stress=[0.0] * 6,
    )
    calculator.close()

    with pytest.raises(RuntimeError, match="closed"):
        getattr(atoms, getter)()


def test_from_abi_forwards_keyword_only_matching_options(monkeypatch):
    from pymultibinit.wrapper_cffi import MultibinitWrapperCFFI

    class FakeWrapper:
        def __init__(self):
            self.abi_file = None

        def init_from_abi_file(self, abi_file):
            self.abi_file = abi_file

    wrapper = FakeWrapper()

    def ensure_cffi_wrapper(potential):
        potential.wrapper = wrapper
        return wrapper

    monkeypatch.setattr(MultibinitPotential, "_ensure_cffi_wrapper", ensure_cffi_wrapper)
    monkeypatch.setattr(MultibinitPotential, "_load_znucl_from_ddb", lambda *_: None)
    monkeypatch.setattr(
        MultibinitPotential, "_fetch_internal_supercell_reference", lambda *_: None
    )
    monkeypatch.setattr(
        MultibinitWrapperCFFI,
        "_parse_abi_file",
        staticmethod(lambda *_: ("input.ddb", "", "")),
    )

    calculator = MultibinitCalculator.from_abi(
        "input.abi",
        "libabinit.so",
        auto_match_atoms=False,
        match_tolerance=1.0,
    )

    assert wrapper.abi_file == "input.abi"
    potential = calculator.potential
    assert isinstance(potential, MultibinitPotential)
    assert potential.auto_match_atoms is False
    assert potential.match_tolerance == 1.0

    signature = inspect.signature(MultibinitCalculator.from_abi)
    assert signature.parameters["auto_match_atoms"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["match_tolerance"].kind is inspect.Parameter.KEYWORD_ONLY


def test_from_abi_spawned_constructs_proxy_without_parent_initialization(monkeypatch):
    created = []

    class FakeSpawnedPotential:
        def __init__(
            self,
            abi_file,
            lib_path,
            *,
            auto_match_atoms,
            match_tolerance,
        ):
            created.append(
                (abi_file, lib_path, auto_match_atoms, match_tolerance)
            )
            self.free_calls = 0

        def free(self):
            self.free_calls += 1

    def fail_parent_initialization(*_, **__):
        pytest.fail("from_abi_spawned must not initialize libabinit in the parent")

    monkeypatch.setattr(calculator_module, "_SpawnedPotential", FakeSpawnedPotential)
    monkeypatch.setattr(MultibinitPotential, "from_abi", fail_parent_initialization)

    calculator = MultibinitCalculator.from_abi_spawned(
        "input.abi",
        "libabinit.so",
        auto_match_atoms=False,
        match_tolerance=1.0,
    )

    assert created == [("input.abi", "libabinit.so", False, 1.0)]
    potential = calculator.potential
    assert isinstance(potential, FakeSpawnedPotential)
    calculator.close()
    assert potential.free_calls == 1

    signature = inspect.signature(MultibinitCalculator.from_abi_spawned)
    assert signature.parameters["auto_match_atoms"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["match_tolerance"].kind is inspect.Parameter.KEYWORD_ONLY


def test_spawned_calculator_returns_results_reference_and_closes_proxy(monkeypatch):
    class FakeSpawnedPotential:
        def __init__(self, *_args, **_kwargs):
            self.reference_atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
            self.evaluations = []
            self.export_calls = 0
            self.free_calls = 0

        def evaluate(self, positions, lattice):
            self.evaluations.append((positions.copy(), lattice.copy()))
            return (
                1.25,
                np.array([[2.0, 3.0, 4.0]]),
                np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
            )

        def export_supercell_to_ase(self):
            self.export_calls += 1
            return self.reference_atoms

        def free(self):
            self.free_calls += 1

    monkeypatch.setattr(calculator_module, "_SpawnedPotential", FakeSpawnedPotential)
    calculator = MultibinitCalculator.from_abi_spawned("input.abi")
    atoms = Atoms(
        "H",
        positions=[[0.1, 0.2, 0.3]],
        cell=np.eye(3) * 4.0,
        pbc=True,
    )
    atoms.calc = calculator
    potential = calculator.potential
    assert isinstance(potential, FakeSpawnedPotential)

    assert atoms.get_potential_energy() == 1.25
    np.testing.assert_allclose(atoms.get_forces(), [[2.0, 3.0, 4.0]])
    np.testing.assert_allclose(
        atoms.get_stress(), [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    )
    assert calculator.get_reference_atoms() is potential.reference_atoms
    assert potential.export_calls == 1
    assert len(potential.evaluations) == 1

    calculator.close()
    calculator.close()
    assert potential.free_calls == 1


def test_spawned_proxy_propagates_structured_child_errors(monkeypatch):
    context = _FakeSpawnContext(
        [
            ("ok", None),
            ("error", "ValueError: invalid potential"),
        ]
    )

    def get_context(method):
        assert method == "spawn"
        return context

    monkeypatch.setattr(calculator_module.multiprocessing, "get_context", get_context)
    potential = calculator_module._SpawnedPotential(
        "input.abi", None, auto_match_atoms=True, match_tolerance=0.1
    )

    with pytest.raises(RuntimeError, match="ValueError: invalid potential"):
        potential.evaluate(np.zeros((1, 3)), np.eye(3))

    request = context.parent.sent[0]
    assert request[0] == "evaluate"
    np.testing.assert_array_equal(request[1], np.zeros((1, 3)))
    np.testing.assert_array_equal(request[2], np.eye(3))


def test_spawned_proxy_reaps_after_structured_child_error(monkeypatch):
    context = _FakeSpawnContext(
        [
            ("ok", None),
            ("error", "ValueError: invalid potential"),
        ]
    )
    monkeypatch.setattr(
        calculator_module.multiprocessing, "get_context", lambda _method: context
    )
    potential = calculator_module._SpawnedPotential(
        "input.abi", None, auto_match_atoms=True, match_tolerance=0.1
    )

    with pytest.raises(RuntimeError, match="ValueError: invalid potential"):
        potential.evaluate(np.zeros((1, 3)), np.eye(3))

    assert potential._closed
    assert context.parent.closed
    assert not context.process.is_alive()


def test_spawned_proxy_closes_pipes_when_process_start_fails(monkeypatch):
    class StartFailProcess(_FakeSpawnProcess):
        def start(self):
            raise RuntimeError("process start failed")

    class StartFailContext(_FakeSpawnContext):
        def Process(self, target, args):
            self.process = StartFailProcess(target, args)
            return self.process

    context = StartFailContext([])
    monkeypatch.setattr(
        calculator_module.multiprocessing, "get_context", lambda _method: context
    )

    with pytest.raises(RuntimeError, match="process start failed"):
        calculator_module._SpawnedPotential(
            "input.abi", None, auto_match_atoms=True, match_tolerance=0.1
        )

    assert context.parent.closed
    assert context.child.closed


def test_spawned_proxy_reports_child_abort(monkeypatch):
    context = _FakeSpawnContext([("ok", None)])
    monkeypatch.setattr(
        calculator_module.multiprocessing, "get_context", lambda _method: context
    )
    potential = calculator_module._SpawnedPotential(
        "input.abi", None, auto_match_atoms=True, match_tolerance=0.1
    )
    context.process._alive = False
    context.process.exitcode = 17

    with pytest.raises(RuntimeError, match="exit code 17"):
        potential.evaluate(np.zeros((1, 3)), np.eye(3))

    potential.free()


def test_spawned_worker_uses_tuple_requests_and_none_close(monkeypatch):
    class Connection:
        def __init__(self):
            self.requests = [
                ("evaluate", np.array([[0.1, 0.2, 0.3]]), np.eye(3)),
                ("reference",),
                None,
            ]
            self.sent = []
            self.closed = False

        def send(self, response):
            self.sent.append(response)

        def recv(self):
            return self.requests.pop(0)

        def close(self):
            self.closed = True

    class Potential:
        def __init__(self):
            self.evaluations = []
            self.reference = Atoms("H", positions=[[0.0, 0.0, 0.0]])
            self.free_calls = 0

        def evaluate(self, positions, lattice):
            self.evaluations.append((positions, lattice))
            return 1.25, np.array([[2.0, 3.0, 4.0]]), np.arange(6.0)

        def export_supercell_to_ase(self):
            return self.reference

        def free(self):
            self.free_calls += 1

    connection = Connection()
    potential = Potential()
    monkeypatch.setattr(
        MultibinitPotential, "from_abi", lambda **_kwargs: potential
    )

    calculator_module._abi_spawned_worker(
        connection, "input.abi", None, True, 0.1
    )

    assert connection.sent[0] == ("ok", None)
    energy, forces, stress = connection.sent[1][1]
    assert connection.sent[1][0] == "ok"
    assert energy == 1.25
    np.testing.assert_array_equal(forces, [[2.0, 3.0, 4.0]])
    np.testing.assert_array_equal(stress, np.arange(6.0))
    assert connection.sent[2] == ("ok", potential.reference)
    assert potential.free_calls == 1
    assert connection.closed


def test_spawned_worker_reports_close_errors(monkeypatch):
    class Connection:
        def __init__(self):
            self.sent = []
            self.closed = False

        def send(self, response):
            self.sent.append(response)

        def recv(self):
            return None

        def close(self):
            self.closed = True

    class FailingPotential:
        def free(self):
            raise RuntimeError("native free failed")

    connection = Connection()
    monkeypatch.setattr(
        MultibinitPotential, "from_abi", lambda **_kwargs: FailingPotential()
    )

    calculator_module._abi_spawned_worker(
        connection, "input.abi", None, True, 0.1
    )

    assert connection.sent == [
        ("ok", None),
        ("error", "RuntimeError: native free failed"),
    ]
    assert connection.closed


def test_spawned_proxy_terminates_an_unresponsive_child_on_close(monkeypatch):
    class HungProcess(_FakeSpawnProcess):
        def __init__(self, target, args):
            super().__init__(target, args)
            self.killed = False

        def join(self, _timeout=None):
            pass

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.killed = True
            self._alive = False

    class HungContext(_FakeSpawnContext):
        def Process(self, target, args):
            self.process = HungProcess(target, args)
            return self.process

    context = HungContext([("ok", None)])
    monkeypatch.setattr(
        calculator_module.multiprocessing, "get_context", lambda _method: context
    )
    potential = calculator_module._SpawnedPotential(
        "input.abi", None, auto_match_atoms=True, match_tolerance=0.1
    )

    potential.free()

    assert context.parent.closed
    assert context.process.terminated
    assert context.process.killed
