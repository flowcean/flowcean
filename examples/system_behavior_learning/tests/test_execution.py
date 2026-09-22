from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import FrozenInstanceError, replace
from io import BytesIO
from multiprocessing import get_context
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
from execution import IncompleteSimulationError, execute_simulations
from experiment import SystemSpec, simulate_scenarios
from settings import ParameterRange, SystemSettings

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from numpy.typing import NDArray

    FloatArray = NDArray[np.float64]


@pytest.mark.parametrize("failed_rows", [(0,), (1,), (0, 1), (0, 1, 2)])
def test_failures_preserve_alignment_and_attempt_every_original_row(
    failed_rows: tuple[int, ...],
) -> None:
    attempted: list[int] = []
    updates: list[int] = []

    def simulate(scenario: FloatArray) -> FloatArray:
        index = int(scenario[0])
        attempted.append(index)
        if index in failed_rows:
            message = f"failure {index}"
            raise RuntimeError(message)
        return np.array([index, index + 0.25])

    scenarios = np.arange(3, dtype=np.float64).reshape(-1, 1)
    batch = execute_simulations(scenarios, simulate, 2, updates.append)

    assert attempted == [0, 1, 2]
    assert updates == [1, 1, 1]
    assert batch.attempted_count == batch.generated_count == 3
    assert batch.successful_count == 3 - len(failed_rows)
    assert batch.unique_valid_trajectory_count == batch.successful_count
    assert batch.trajectories.shape == (3, 2)
    np.testing.assert_array_equal(batch.scenarios, scenarios)
    for index in range(3):
        assert batch.valid[index] == (index not in failed_rows)
        if index in failed_rows:
            assert np.isnan(batch.trajectories[index]).all()
            assert batch.errors[index] == f"RuntimeError: failure {index}"
        else:
            np.testing.assert_array_equal(
                batch.trajectories[index],
                [index, index + 0.25],
            )
            assert batch.errors[index] == ""

    # A dependent analysis must not receive just the surviving targets.
    with pytest.raises(IncompleteSimulationError, match="fitting") as caught:
        batch.require_complete("fitting")
    assert caught.value.batch is batch
    assert caught.value.batch.trajectories.shape == (3, 2)


@pytest.mark.parametrize(
    ("output", "reason"),
    [
        (np.array([[1.0, 2.0]]), "trajectory shape (1, 2); expected (2,)"),
        (np.array([1.0]), "trajectory shape (1,); expected (2,)"),
        (np.array(1.0), "trajectory shape (); expected (2,)"),
        (
            np.array([1.0, np.nan]),
            "trajectory must contain only finite values",
        ),
        (
            np.array([np.inf, 1.0]),
            "trajectory must contain only finite values",
        ),
        (
            np.array([-np.inf, 1.0]),
            "trajectory must contain only finite values",
        ),
    ],
)
def test_invalid_outputs_are_unavailable(
    output: FloatArray,
    reason: str,
) -> None:
    batch = execute_simulations([[0.0]], lambda _: output, 2)
    assert not batch.valid.any()
    assert np.isnan(batch.trajectories).all()
    assert batch.errors == (f"ValueError: {reason}",)


def test_output_conversion_failure_does_not_abort_batch() -> None:
    outputs = iter([["not a number"], [2.0]])
    batch = execute_simulations([[0.0], [1.0]], lambda _: next(outputs), 1)
    np.testing.assert_array_equal(batch.valid, [False, True])
    assert batch.errors[0].startswith("ValueError:")
    assert batch.trajectories[1, 0] == 2.0


@pytest.mark.parametrize(
    "error",
    [ValueError("invalid"), ArithmeticError("arithmetic")],
)
def test_other_recoverable_simulator_errors(error: Exception) -> None:
    def simulate(_: FloatArray) -> FloatArray:
        raise error

    batch = execute_simulations([[0.0], [1.0]], simulate, 1)
    assert batch.attempted_count == 2
    assert batch.successful_count == 0
    assert batch.errors == (f"{type(error).__name__}: {error}",) * 2


@pytest.mark.parametrize(
    "error",
    [
        TypeError("programming"),
        IndexError("programming"),
        MemoryError("resources"),
        KeyboardInterrupt(),
        SystemExit(),
    ],
)
def test_nonrecoverable_errors_escape(error: BaseException) -> None:
    attempted: list[float] = []
    updates: list[int] = []

    def simulate(scenario: FloatArray) -> FloatArray:
        attempted.append(float(scenario[0]))
        raise error

    with pytest.raises(type(error)) as caught:
        execute_simulations([[0.0], [1.0]], simulate, 1, updates.append)
    assert caught.value is error
    assert attempted == [0.0]
    assert updates == []


@pytest.mark.parametrize("simulation_fails", [False, True])
def test_progress_errors_escape_even_after_recoverable_failures(
    *,
    simulation_fails: bool,
) -> None:
    attempted: list[float] = []
    updates: list[int] = []
    callback_error = RuntimeError("progress failed")

    def simulate(scenario: FloatArray) -> FloatArray:
        attempted.append(float(scenario[0]))
        if simulation_fails:
            message = "simulation failed"
            raise ValueError(message)
        return scenario

    def progress(count: int) -> None:
        updates.append(count)
        raise callback_error

    with pytest.raises(RuntimeError, match="progress failed") as caught:
        execute_simulations([[0.0], [1.0]], simulate, 1, progress)
    assert caught.value is callback_error
    assert attempted == [0.0]
    assert updates == [1]


@pytest.mark.parametrize(
    "scenarios",
    [
        np.array([]),
        np.empty((0, 1)),
        np.empty((1, 0)),
        np.zeros((1, 1, 1)),
        np.array([[np.nan]]),
        np.array([[np.inf]]),
    ],
)
def test_malformed_scenarios_rejected_before_simulation(
    scenarios: FloatArray,
) -> None:
    attempted: list[FloatArray] = []

    def simulate(scenario: FloatArray) -> FloatArray:
        attempted.append(scenario)
        return scenario

    with pytest.raises(ValueError, match="finite, nonempty 2D"):
        execute_simulations(scenarios, simulate, 1)
    assert attempted == []


@pytest.mark.parametrize("width", [0, -1, 1.5, True])
def test_malformed_width_rejected_before_simulation(width: object) -> None:
    attempted: list[FloatArray] = []

    def simulate(scenario: FloatArray) -> FloatArray:
        attempted.append(scenario)
        return scenario

    with pytest.raises(ValueError, match="positive integer"):
        execute_simulations([[0.0]], simulate, cast("int", width))
    assert attempted == []


def test_duplicates_are_counted_exactly_and_not_removed() -> None:
    scenarios = np.array([[0.0], [-0.0], [1.0], [np.nextafter(1.0, 2.0)]])
    attempted: list[float] = []

    def simulate(scenario: FloatArray) -> FloatArray:
        attempted.append(float(scenario[0]))
        return scenario

    batch = execute_simulations(scenarios, simulate, 1)
    assert attempted == scenarios[:, 0].tolist()
    assert batch.generated_count == batch.attempted_count == 4
    assert batch.successful_count == 4
    assert batch.unique_scenario_count == 3
    assert batch.unique_valid_trajectory_count == 3
    assert batch.require_complete("duplicates") is batch.trajectories
    assert batch.trajectories.tobytes() == scenarios.tobytes()


def test_identical_physical_traces_at_distinct_scenarios_are_valid() -> None:
    batch = execute_simulations([[0.0], [1.0]], lambda _: [4.0], 1)
    assert batch.unique_scenario_count == 2
    assert batch.unique_valid_trajectory_count == 1
    assert batch.successful_count == 2
    np.testing.assert_array_equal(
        batch.require_complete("constant"),
        [[4], [4]],
    )


def test_batch_evidence_is_immutable_and_does_not_alias_inputs() -> None:
    scenarios = np.array([[1.0]])
    output = np.array([2.0])
    batch = execute_simulations(scenarios, lambda _: output, 1)
    scenarios[:] = 100.0
    output[:] = 200.0
    np.testing.assert_array_equal(batch.scenarios, [[1.0]])
    np.testing.assert_array_equal(batch.trajectories, [[2.0]])
    for values in (batch.scenarios, batch.trajectories, batch.valid):
        with pytest.raises(ValueError, match=r"cannot set .* flag to True"):
            values.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        setattr(batch, "errors", ())  # noqa: B010


def test_evidence_can_be_saved_without_pickle() -> None:
    batch = execute_simulations(
        [[0.0], [1.0]],
        lambda row: [np.nan] if row[0] == 0 else [1.0],
        1,
    )
    stream = BytesIO()
    np.savez(
        stream,
        trajectories=batch.trajectories,
        valid=batch.valid,
        errors=np.asarray(batch.errors, dtype=np.str_),
    )
    stream.seek(0)
    with np.load(stream, allow_pickle=False) as saved:
        np.testing.assert_array_equal(
            saved["trajectories"],
            batch.trajectories,
        )
        np.testing.assert_array_equal(saved["valid"], batch.valid)
        assert saved["errors"].tolist() == list(batch.errors)


def _spec(attempted: list[float], *, fail: bool = False) -> SystemSpec:
    settings = SystemSettings(
        name="Toy",
        scenario_parameters=(ParameterRange("x", 0.0, 1.0),),
        fixed_parameters=(),
        horizon=(0.0, 1.0),
        state_names=("z",),
    )

    def simulate(
        _settings: SystemSettings,
        scenario: Mapping[str, float],
        times: FloatArray,
    ) -> FloatArray:
        x = scenario["x"]
        attempted.append(x)
        if fail and x == 0.0:
            message = "unavailable"
            raise RuntimeError(message)
        return x + times

    return SystemSpec(settings, simulate)


def test_strict_wrapper_preserves_successful_results_exactly() -> None:
    spec = _spec([])
    scenarios = np.array([[0.0], [0.25], [0.75]])
    expected = np.stack([spec.simulate_scenario(row, 4) for row in scenarios])
    actual = simulate_scenarios(spec, scenarios, 4)
    assert actual.dtype == np.float64
    assert actual.tobytes() == expected.tobytes()


def test_strict_wrapper_raises_with_full_evidence_not_survivors() -> None:
    attempted: list[float] = []
    updates: list[int] = []
    spec = _spec(attempted, fail=True)
    with pytest.raises(IncompleteSimulationError, match="Toy") as caught:
        simulate_scenarios(spec, np.array([[0.0], [1.0]]), 4, updates.append)
    assert attempted == [0.0, 1.0]
    assert updates == [1, 1]
    batch = caught.value.batch
    assert batch.trajectories.shape == (2, 4)
    np.testing.assert_array_equal(batch.valid, [False, True])
    assert np.isnan(batch.trajectories[0]).all()
    np.testing.assert_array_equal(
        batch.trajectories[1],
        1 + np.linspace(0, 1, 4),
    )
    assert batch.errors == ("RuntimeError: unavailable", "")


@pytest.mark.parametrize(
    ("scenarios", "sample_count"),
    [(np.zeros((1, 2)), 4), (np.zeros((2, 1)), 1)],
)
def test_wrapper_validates_global_arguments_before_simulation(
    scenarios: FloatArray,
    sample_count: int,
) -> None:
    attempted: list[float] = []
    with pytest.raises(
        ValueError,
        match=r"invalid scenario matrix|at least two",
    ):
        simulate_scenarios(_spec(attempted), scenarios, sample_count)
    assert attempted == []


@pytest.mark.parametrize("output", [np.zeros((2, 2)), np.full(4, np.nan)])
def test_wrapper_validates_custom_simulator_outputs(
    output: FloatArray,
) -> None:
    spec = replace(_spec([]), simulator=lambda *_: output)
    with pytest.raises(IncompleteSimulationError) as caught:
        simulate_scenarios(spec, np.array([[0.0], [1.0]]), 4)
    batch = caught.value.batch
    assert batch.attempted_count == 2
    assert batch.successful_count == 0
    assert batch.trajectories.shape == (2, 4)
    assert np.isnan(batch.trajectories).all()


def test_incomplete_batch_evidence_survives_process_boundary() -> None:
    batch = execute_simulations(
        [[0.0], [1.0]],
        lambda row: [np.nan] if row[0] == 0 else [1.0],
        1,
    )
    with ProcessPoolExecutor(
        max_workers=1,
        mp_context=get_context("spawn"),
    ) as executor:
        future = executor.submit(batch.require_complete, "worker")
        with pytest.raises(
            IncompleteSimulationError,
            match="worker",
        ) as caught:
            future.result(timeout=30)
    batch = caught.value.batch
    assert caught.value.label == "worker"
    assert batch.attempted_count == 2
    np.testing.assert_array_equal(batch.valid, [False, True])
    np.testing.assert_array_equal(batch.scenarios, [[0.0], [1.0]])
    np.testing.assert_array_equal(batch.trajectories, [[np.nan], [1.0]])
    assert batch.errors[0].startswith("ValueError:")
    assert batch.errors[1] == ""
    for values in (batch.scenarios, batch.trajectories, batch.valid):
        with pytest.raises(ValueError, match=r"cannot set .* flag to True"):
            values.setflags(write=True)


def test_invalid_progress_rejected_before_simulation() -> None:
    attempted: list[FloatArray] = []

    def simulate(scenario: FloatArray) -> FloatArray:
        attempted.append(scenario)
        return scenario

    with pytest.raises(TypeError, match="must be callable"):
        execute_simulations(
            [[0.0]],
            simulate,
            1,
            cast("Callable[[int], object]", 1),
        )
    assert attempted == []


def test_invalid_simulator_rejected_before_progress() -> None:
    updates: list[int] = []
    with pytest.raises(TypeError, match="must be callable"):
        execute_simulations(
            [[0.0]],
            cast("Callable[[FloatArray], FloatArray]", None),
            1,
            updates.append,
        )
    assert updates == []
