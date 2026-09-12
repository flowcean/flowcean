from __future__ import annotations

import json
import os
import runpy
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from multiprocessing import get_context
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

import inference_run
import numpy as np
import pytest
import report
import run
import settings
from experiment import ExperimentRecords
from numerical_threads import (
    DYNAMIC_THREAD_VARIABLES,
    NUMERICAL_THREAD_VARIABLES,
    THREAD_VARIABLES,
)
from threadpoolctl import threadpool_info

if TYPE_CHECKING:
    from pathlib import Path

EMPTY = ExperimentRecords((), (), (), (), (), (), (), (), ())


def _limits() -> dict[str, int]:
    return {
        pool["filepath"]: pool["num_threads"] for pool in threadpool_info()
    }


def _assert_scope(count: int) -> None:
    assert _limits()
    assert set(_limits().values()) == {count}
    assert all(
        os.environ[name] == str(count) for name in NUMERICAL_THREAD_VARIABLES
    )
    assert all(
        os.environ[name] == "FALSE" for name in DYNAMIC_THREAD_VARIABLES
    )


def _expected_environment(count: int) -> dict[str, str]:
    return {
        **dict.fromkeys(NUMERICAL_THREAD_VARIABLES, str(count)),
        **dict.fromkeys(DYNAMIC_THREAD_VARIABLES, "FALSE"),
    }


@pytest.mark.parametrize("count", [True, False, 0, -1, 1.5, "8", None])
def test_invalid_threads_fail_before_work_output_or_environment_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    count: Any,
) -> None:
    configuration = replace(settings.DEVELOPMENT, output_dir=tmp_path / "run")
    forbidden = Mock(side_effect=AssertionError("work before validation"))
    monkeypatch.setattr(run, "run_experiment", forbidden)
    monkeypatch.setattr(report, "load_records", forbidden)
    before = dict(os.environ)
    with pytest.raises(ValueError, match="numerical_threads"):
        run.run_and_write(configuration, numerical_threads=count)
    with pytest.raises(ValueError, match="numerical_threads"):
        report.report_saved_run(
            configuration.output_dir,
            numerical_threads=count,
        )
    with pytest.raises(ValueError, match="numerical_threads"):
        inference_run.run_inference(
            configuration,
            EMPTY,
            data_dir=tmp_path / "data",
            results_dir=tmp_path / "results",
            numerical_threads=count,
        )
    forbidden.assert_not_called()
    assert list(tmp_path.iterdir()) == []
    assert dict(os.environ) == before


def test_cli_threads_reach_whole_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configuration = replace(settings.DEVELOPMENT, output_dir=tmp_path / "run")
    monkeypatch.setattr(settings, "DEVELOPMENT", configuration)
    generate = Mock(return_value=EMPTY)
    infer = Mock(return_value=())
    monkeypatch.setattr("experiment.run_experiment", generate)
    monkeypatch.setattr("report.write_reports", Mock(return_value=()))
    monkeypatch.setattr("inference_run.run_inference", infer)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run.py", "--workers", "3", "--numerical-threads", "4"],
    )
    runpy.run_path(run.__file__, run_name="__main__")
    assert generate.call_args.args[0].workers == 3
    assert infer.call_args.kwargs["numerical_threads"] == 4
    applied = infer.call_args.args[0]
    assert (
        replace(
            applied,
            output_dir=configuration.output_dir,
            workers=configuration.workers,
        )
        == configuration
    )


@pytest.mark.parametrize("fails", [False, True])
def test_run_scope_covers_all_phases_and_restores_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    fails: bool,
) -> None:
    for name in THREAD_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OMP_NUM_THREADS", "caller-omp")
    monkeypatch.setenv("MKL_DYNAMIC", "TRUE")
    environment = dict(os.environ)
    original_limits = _limits()
    configuration = replace(
        settings.DEVELOPMENT,
        systems=settings.DEVELOPMENT.systems[:1],
        output_dir=tmp_path / "run",
    )
    phases: list[str] = []

    def generate(*_args: Any, **_kwargs: Any) -> ExperimentRecords:
        _assert_scope(8)
        metadata = json.loads(
            (configuration.output_dir / "environment.json").read_text(),
        )
        assert metadata["thread_environment"] == _expected_environment(8)
        phases.append("experiment")
        if fails:
            message = "experiment failed"
            raise RuntimeError(message)
        return EMPTY

    def write(_records: Any, results_dir: Path) -> tuple[Path, ...]:
        _assert_scope(8)
        results_dir.mkdir()
        phases.append("report")
        return ()

    inference_data = Mock()
    inference_result = Mock()

    def infer(actual_data: Any, actual_settings: Any) -> Any:
        _assert_scope(8)
        assert actual_data is inference_data
        assert actual_settings is configuration.statistics
        phases.append("inference")
        return inference_result

    monkeypatch.setattr(run, "run_experiment", generate)
    monkeypatch.setattr(run, "write_reports", write)
    monkeypatch.setattr(
        inference_run,
        "load_coverage_input",
        Mock(return_value=inference_data),
    )
    monkeypatch.setattr(inference_run, "infer_coverage", infer)
    inference_writer = Mock(return_value=())
    monkeypatch.setattr(
        inference_run,
        "write_coverage_inference",
        inference_writer,
    )
    monkeypatch.setattr(
        inference_run,
        "write_coverage_ratios",
        Mock(return_value=tmp_path / "coverage.png"),
    )
    if fails:
        with pytest.raises(RuntimeError, match="experiment failed"):
            run.run_and_write(configuration)
        assert phases == ["experiment"]
    else:
        run.run_and_write(configuration)
        assert phases == ["experiment", "report", "inference"]
        assert inference_writer.call_args.kwargs == {"numerical_threads": 8}
    assert dict(os.environ) == environment
    assert _limits() == original_limits


def test_saved_reporting_loads_and_writes_inside_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "saved"
    run_dir.mkdir()
    configuration = replace(settings.DEVELOPMENT, output_dir=run_dir)
    (run_dir / "settings.json").write_text(json.dumps(configuration.to_dict()))
    phases: list[str] = []

    def load(_path: Path) -> ExperimentRecords:
        _assert_scope(4)
        phases.append("load")
        return EMPTY

    def write(_records: ExperimentRecords, _path: Path) -> tuple[Path, ...]:
        _assert_scope(4)
        phases.append("write")
        return ()

    def infer(*_args: Any, **kwargs: Any) -> tuple[Path, ...]:
        _assert_scope(4)
        assert kwargs["numerical_threads"] == 4
        phases.append("inference")
        return ()

    monkeypatch.setattr(report, "load_records", load)
    monkeypatch.setattr(report, "write_reports", write)
    monkeypatch.setattr(report, "run_inference", infer)
    report.report_saved_run(
        run_dir,
        tmp_path / "reported",
        numerical_threads=4,
    )
    assert phases == ["load", "write", "inference"]


def test_spawned_worker_inherits_actual_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configuration = replace(
        settings.DEVELOPMENT,
        systems=settings.DEVELOPMENT.systems[:1],
        statistics=None,
        output_dir=tmp_path / "run",
    )

    def generate(*_args: Any, **_kwargs: Any) -> ExperimentRecords:
        _assert_scope(8)
        with ProcessPoolExecutor(
            max_workers=1,
            mp_context=get_context("spawn"),
        ) as executor:
            executor.submit(
                np.dot,
                np.ones((2, 2)),
                np.ones((2, 2)),
            ).result()
            environment = {
                name: executor.submit(os.getenv, name).result()
                for name in THREAD_VARIABLES
            }
            pools = executor.submit(threadpool_info).result()
        assert environment == _expected_environment(8)
        assert pools
        assert {pool["num_threads"] for pool in pools} == {8}
        return EMPTY

    monkeypatch.setattr(run, "run_experiment", generate)
    monkeypatch.setattr(run, "write_reports", Mock(return_value=()))
    run.run_and_write(configuration)
