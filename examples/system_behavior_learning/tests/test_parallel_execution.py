from __future__ import annotations

import os
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import replace
from multiprocessing import get_context
from pathlib import Path
from typing import TYPE_CHECKING, cast
from unittest.mock import Mock

import experiment
import numpy as np
import pytest
from experiment import (
    SystemSpec,
    _simulate_scenario_batch_parallel,
    _simulate_scenario_chunk,
    run_experiment,
    simulate_scenario_batch,
)
from settings import ParameterRange, Settings, SystemSettings

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from execution import SimulationBatch
    from experiment import FloatArray, SimulationProgress


def _simulate(
    _settings: SystemSettings,
    scenario: Mapping[str, float],
    times: FloatArray,
) -> FloatArray:
    x = scenario["x"]
    if x in (0, 7, 16):
        message = f"unavailable {x}"
        raise RuntimeError(message)
    if x < 0:
        message = "unexpected worker error"
        raise TypeError(message)
    return x + times


def _pid_simulator(
    _settings: SystemSettings,
    _scenario: Mapping[str, float],
    times: FloatArray,
) -> FloatArray:
    return np.full_like(times, os.getpid())


@pytest.fixture
def spec(monkeypatch: pytest.MonkeyPatch) -> SystemSpec:
    # Pytest's importlib mode does not put the repository on child sys.path.
    monkeypatch.syspath_prepend(str(Path(__file__).parents[3]))
    return SystemSpec(
        SystemSettings(
            name="Toy",
            scenario_parameters=(ParameterRange("x", 0.0, 20.0),),
            fixed_parameters=(),
            horizon=(0.0, 1.0),
            state_names=("z",),
        ),
        _simulate,
    )


def _assert_batches_equal(
    left: SimulationBatch,
    right: SimulationBatch,
) -> None:
    assert left.errors == right.errors
    for name in ("scenarios", "trajectories", "valid"):
        expected = getattr(left, name)
        actual = getattr(right, name)
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        assert actual.tobytes() == expected.tobytes()
        with pytest.raises(ValueError, match=r"cannot set .* flag to True"):
            actual.setflags(write=True)
    assert left.unique_scenario_count == right.unique_scenario_count
    assert (
        left.unique_valid_trajectory_count
        == right.unique_valid_trajectory_count
    )


@pytest.mark.parametrize("row_count", [1, 5, 19])
def test_chunk_scheduling_restores_order(
    spec: SystemSpec,
    monkeypatch: pytest.MonkeyPatch,
    row_count: int,
) -> None:
    scenarios = np.arange(row_count, dtype=np.float64).reshape(-1, 1)
    scenarios[-1] = scenarios[0]  # Duplicates must be attempted, not removed.
    expected = simulate_scenario_batch(spec, scenarios, 4)
    chunks: list[FloatArray] = []
    completed: list[Future[SimulationBatch]] = []
    updates: list[int] = []
    serial = Mock(wraps=simulate_scenario_batch)
    monkeypatch.setattr(experiment, "simulate_scenario_batch", serial)

    def submit(
        function: Callable[[SystemSpec, FloatArray, int], SimulationBatch],
        system: SystemSpec,
        chunk: FloatArray,
        samples: int,
    ) -> Future[SimulationBatch]:
        assert function is _simulate_scenario_chunk
        chunks.append(chunk.copy())
        future: Future[SimulationBatch] = Future()
        future.set_result(function(system, chunk, samples))
        completed.append(future)
        return future

    def reverse_completion(futures: object) -> list[Future[SimulationBatch]]:
        assert len(chunks) == min(row_count, 8)  # All submitted first.
        assert updates == []
        assert (
            list(cast("Mapping[Future[SimulationBatch], int]", futures))
            == completed
        )
        return completed[::-1]

    monkeypatch.setattr(experiment, "as_completed", reverse_completion)
    executor = Mock(spec=ProcessPoolExecutor, submit=Mock(side_effect=submit))
    actual = _simulate_scenario_batch_parallel(
        spec,
        scenarios,
        4,
        updates.append,
        executor=executor,
        workers=2,
    )
    _assert_batches_equal(expected, actual)
    assert updates == [1] * row_count
    np.testing.assert_array_equal(np.concatenate(chunks), scenarios)
    assert [len(chunk) for chunk in chunks] == [
        len(chunk) for chunk in np.array_split(scenarios, min(row_count, 8))
    ]
    assert all(call.args[3] is None for call in serial.call_args_list)


@pytest.mark.parametrize(
    ("scenarios", "samples", "progress"),
    [
        (
            np.zeros((1, 2)),
            1,
            None,
        ),  # Shape takes precedence over sample count.
        (np.array([[0.0], [np.nan]]), 1, None),
        (np.empty((0, 1)), 4, None),
        (np.array([[0.0], [np.nan]]), 4, None),
        (np.array([[0.0], [np.inf]]), 4, None),
        (np.zeros((9, 1)), 1.5, None),
        (np.zeros((9, 1)), 4, 1),
    ],
)
def test_invalid_global_inputs_submit_nothing(
    spec: SystemSpec,
    scenarios: FloatArray,
    samples: int,
    progress: SimulationProgress | None,
) -> None:
    executor = Mock(spec=ProcessPoolExecutor)
    with pytest.raises((ValueError, TypeError)) as serial_error:
        simulate_scenario_batch(spec, scenarios, samples, progress)
    with pytest.raises(type(serial_error.value)) as parallel_error:
        _simulate_scenario_batch_parallel(
            spec,
            scenarios,
            samples,
            progress,
            executor=executor,
            workers=2,
        )
    assert str(parallel_error.value) == str(serial_error.value)
    executor.submit.assert_not_called()


@pytest.mark.parametrize("workers", [0, -1, True, 1.5])
def test_invalid_worker_count_submits_nothing(
    spec: SystemSpec,
    workers: int,
) -> None:
    executor = Mock(spec=ProcessPoolExecutor)
    with pytest.raises(ValueError, match="workers"):
        _simulate_scenario_batch_parallel(
            spec,
            np.ones((9, 1)),
            4,
            executor=executor,
            workers=workers,
        )
    executor.submit.assert_not_called()


def test_spawn_failures_progress_and_pool_reuse(spec: SystemSpec) -> None:
    scenarios = np.arange(19, dtype=np.float64).reshape(-1, 1)
    scenarios[-1] = scenarios[0]
    expected = simulate_scenario_batch(spec, scenarios, 4)
    updates: list[tuple[int, int]] = []

    def progress(increment: int) -> None:
        updates.append((os.getpid(), increment))

    def broken_progress(_: int) -> None:
        message = "unexpected progress error"
        raise RuntimeError(message)

    with ProcessPoolExecutor(
        max_workers=2,
        mp_context=get_context("spawn"),
    ) as pool:
        actual = _simulate_scenario_batch_parallel(
            spec,
            scenarios,
            4,
            progress,
            executor=pool,
            workers=2,
        )
        _assert_batches_equal(expected, actual)
        assert updates == [(os.getpid(), 1)] * len(scenarios)
        with pytest.raises(TypeError, match="unexpected worker error"):
            _simulate_scenario_batch_parallel(
                spec,
                np.array([[1.0], [-1.0]]),
                4,
                executor=pool,
                workers=2,
            )
        with pytest.raises(RuntimeError, match="unexpected progress error"):
            _simulate_scenario_batch_parallel(
                spec,
                scenarios,
                4,
                broken_progress,
                executor=pool,
                workers=2,
            )
        # Both errors propagate without destroying the caller-owned pool.
        again = _simulate_scenario_batch_parallel(
            spec,
            scenarios,
            4,
            executor=pool,
            workers=2,
        )
        _assert_batches_equal(expected, again)

    # A one-process executor makes PID reuse across batches unambiguous.
    with ProcessPoolExecutor(
        max_workers=1,
        mp_context=get_context("spawn"),
    ) as pool:
        pids = [
            _simulate_scenario_batch_parallel(
                replace(spec, simulator=_pid_simulator),
                scenarios,
                4,
                executor=pool,
                workers=1,
            ).trajectories
            for _ in range(2)
        ]
        np.testing.assert_array_equal(pids[0], pids[1])
        assert np.all(pids[0] != os.getpid())


@pytest.mark.parametrize("workers", [1, 2])
def test_single_system_uses_one_run_scoped_pool(
    spec: SystemSpec,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    workers: int,
) -> None:
    factory = Mock(wraps=ProcessPoolExecutor)
    parallel_batch = Mock(wraps=_simulate_scenario_batch_parallel)
    monkeypatch.setattr(
        experiment,
        "_simulate_scenario_batch_parallel",
        parallel_batch,
    )
    monkeypatch.setattr(experiment, "ProcessPoolExecutor", factory)
    settings = Settings(
        systems=(spec.settings,),
        workers=workers,
        replicates=2,
        fitting_size=9,
        assessment_size=9,
        reference_size=9,
        capacities=(2,),
        geometric_repetitions=1,
        trajectory_samples=4,
    )
    run_experiment(
        settings,
        (replace(spec, simulator=_pid_simulator),),
        raw_output_dir=tmp_path,
    )
    assert factory.call_count == (0 if workers == 1 else 1)
    if workers == 1:
        parallel_batch.assert_not_called()
    else:
        assert parallel_batch.call_count >= 5  # Reference and both histories.
        executors = [
            call.kwargs["executor"] for call in parallel_batch.call_args_list
        ]
        assert all(executor is executors[0] for executor in executors)
        assert factory.call_args.kwargs["max_workers"] == workers
        assert (
            factory.call_args.kwargs["mp_context"].get_start_method()
            == "spawn"
        )
    with np.load(tmp_path / "00_toy/shared.npz", allow_pickle=False) as raw:
        pids = raw["reference_trajectories"]
        assert (
            np.all(pids == os.getpid())
            if workers == 1
            else np.all(pids != os.getpid())
        )
