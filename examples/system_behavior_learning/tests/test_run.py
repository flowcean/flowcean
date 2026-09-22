from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
import run
from experiment import (
    ExecutionStatusRow,
    ExperimentRecords,
    LeafBoxRow,
    PrototypePlotData,
)
from record_io import ROW_TYPES, load_records, save_records
from report import write_reports
from run import _environment
from settings import DEVELOPMENT, FULL, Settings

EMPTY = ExperimentRecords((), (), (), (), (), (), (), (), ())


def test_main_selects_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = Mock()
    monkeypatch.setattr(run, "run_and_write", runner)
    run.main()
    run.main(full=True)
    for call, preset in zip(
        runner.call_args_list,
        (DEVELOPMENT, FULL),
        strict=True,
    ):
        (settings,) = call.args
        assert call.kwargs == {"numerical_threads": 8}
        assert replace(settings, output_dir=preset.output_dir) == preset
        assert settings.output_dir.parent == preset.output_dir
        assert settings.output_dir.name.startswith("experiment-")


@pytest.mark.parametrize("full", [False, True])
def test_main_workers_override(
    monkeypatch: pytest.MonkeyPatch,
    *,
    full: bool,
) -> None:
    runner = Mock()
    monkeypatch.setattr(run, "run_and_write", runner)
    preset = FULL if full else DEVELOPMENT
    run.main(full=full, workers=3, numerical_threads=4)
    assert runner.call_args.kwargs == {"numerical_threads": 4}
    (settings,) = runner.call_args.args
    assert settings.workers == 3
    assert settings.output_dir.parent == preset.output_dir
    assert (
        replace(settings, workers=preset.workers, output_dir=preset.output_dir)
        == preset
    )
    assert runner.call_count == 1


@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("workers", [1.5, True, False, 0, -1])
def test_main_rejects_workers_before_running_or_creating_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    workers: Any,
    *,
    full: bool,
) -> None:
    destination = tmp_path / "not-created"
    runner = Mock(side_effect=AssertionError("must not start execution"))
    monkeypatch.setattr(run, "run_and_write", runner)
    for name, preset in (("FULL", FULL), ("DEVELOPMENT", DEVELOPMENT)):
        monkeypatch.setattr(run, name, replace(preset, output_dir=destination))
    with pytest.raises(
        ValueError,
        match="workers must be a positive built-in integer",
    ):
        run.main(full=full, workers=workers)
    runner.assert_not_called()
    assert not destination.exists()


@pytest.mark.parametrize("nonempty", [False, True])
def test_existing_run_is_untouched(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    nonempty: bool,
) -> None:
    destination = tmp_path / "existing"
    destination.mkdir()
    marker = destination / "keep.txt"
    if nonempty:
        marker.write_text("keep")
    generate = Mock()
    monkeypatch.setattr(run, "run_experiment", generate)
    with pytest.raises(FileExistsError):
        run.run_and_write(Settings(output_dir=destination))
    generate.assert_not_called()
    assert list(destination.iterdir()) == ([marker] if nonempty else [])
    if nonempty:
        assert marker.read_text() == "keep"


def test_startup_metadata_allows_dirty_source_and_preserves_partial_raw(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = replace(DEVELOPMENT, output_dir=tmp_path / "run")
    git = Mock(side_effect=["a" * 40 + "\n", "?? untracked.py\n"])
    monkeypatch.setattr(run.subprocess, "check_output", git)

    def generate(*_args: object, **kwargs: Any) -> ExperimentRecords:
        root = settings.output_dir
        assert json.loads((root / "settings.json").read_text()) == json.loads(
            json.dumps(settings.to_dict()),
        )
        environment = json.loads((root / "environment.json").read_text())
        assert environment["source_head"] == "a" * 40
        assert environment["source_dirty"] is True
        assert (root / "uv.lock").read_bytes() == (
            Path(run.__file__).resolve().parents[2] / "uv.lock"
        ).read_bytes()
        (kwargs["raw_output_dir"] / "partial.npz").write_bytes(b"partial")
        message = "interrupted generation"
        raise RuntimeError(message)

    monkeypatch.setattr(run, "run_experiment", generate)
    with pytest.raises(RuntimeError, match="interrupted generation"):
        run.run_and_write(settings)
    assert "--untracked-files=all" in git.call_args_list[1].args[0]
    assert (
        settings.output_dir / "data/raw/partial.npz"
    ).read_bytes() == b"partial"
    assert not (settings.output_dir / "data/records.json").exists()
    assert not (settings.output_dir / "results").exists()
    assert "Completed" not in capsys.readouterr().out


def test_git_unavailable_is_unknown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        run.subprocess,
        "check_output",
        Mock(side_effect=FileNotFoundError),
    )
    metadata = _environment(tmp_path)
    assert metadata["source_head"] is None
    assert metadata["source_dirty"] is None


def test_report_failure_keeps_records_without_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = Settings(output_dir=tmp_path / "run")
    monkeypatch.setattr(run, "run_experiment", Mock(return_value=EMPTY))

    def fail(
        _records: ExperimentRecords,
        results_dir: Path,
    ) -> tuple[Path, ...]:
        results_dir.mkdir()
        (results_dir / "partial.csv").write_text("partial")
        message = "report failure"
        raise OSError(message)

    monkeypatch.setattr(run, "write_reports", fail)
    with pytest.raises(OSError, match="report failure"):
        run.run_and_write(settings)
    assert (settings.output_dir / "data/records.json").exists()
    assert (
        settings.output_dir / "results/partial.csv"
    ).read_text() == "partial"
    assert "Completed" not in capsys.readouterr().out
    with pytest.raises(FileExistsError):
        write_reports(EMPTY, settings.output_dir / "results")


def test_records_preserve_rows_and_prototype_arrays(tmp_path: Path) -> None:
    plot = PrototypePlotData(
        system="Toy",
        replicate=0,
        capacity=2,
        state_names=("z", "a"),
        sample_times=np.array([0.0, 1.0]),
        leaf_ids=np.array([2, 1], dtype=np.int64),
        relative_volumes=np.array([0.25, 0.75]),
        fitting_assignments=np.array([1, 2], dtype=np.int64),
        fitting_trajectories=np.arange(8.0).reshape(2, 4),
        leaf_prototypes=np.arange(8.0).reshape(2, 4),
        midpoint_trajectories=np.arange(8.0).reshape(2, 4),
    )
    records = replace(
        EMPTY,
        leaf_boxes=(
            LeafBoxRow(
                "Toy",
                0,
                2,
                1,
                "x",
                0,
                1,
                lower_inclusive=True,
                upper_inclusive=False,
                midpoint=0.5,
            ),
        ),
        execution_statuses=(
            ExecutionStatusRow("Toy", None, None, "reference", "valid"),
            ExecutionStatusRow("Toy", 0, 2, "fit", "failed"),
        ),
        prototype_plots=(plot, replace(plot, replicate=1)),
    )
    save_records(records, tmp_path)
    saved = json.loads((tmp_path / "records.json").read_text())
    loaded = load_records(tmp_path)
    assert set(saved) == {*ROW_TYPES, "prototype_plots"}
    # Finite records retain the pre-tagging JSON format byte for byte.
    expected = {
        name: [asdict(row) for row in getattr(records, name)]
        for name in ROW_TYPES
    }
    expected["prototype_plots"] = saved["prototype_plots"]
    assert (tmp_path / "records.json").read_text() == (
        json.dumps(expected, indent=2, allow_nan=False) + "\n"
    )
    for name, schema in ROW_TYPES.items():
        rows = getattr(records, name)
        assert all(isinstance(row, schema) for row in rows)
        assert saved[name] == [asdict(row) for row in rows]
        assert getattr(loaded, name) == rows
        for original, restored in zip(
            rows,
            getattr(loaded, name),
            strict=True,
        ):
            for field, value in asdict(original).items():
                assert type(getattr(restored, field)) is type(value)
    for index, metadata in enumerate(saved["prototype_plots"]):
        assert metadata == {
            "system": "Toy",
            "replicate": index,
            "capacity": 2,
            "state_names": ["z", "a"],
            "npz": f"prototypes/prototype_{index}.npz",
        }
        restored = loaded.prototype_plots[index]
        assert restored.system == plot.system
        assert restored.replicate == index
        assert restored.capacity == plot.capacity
        assert restored.state_names == plot.state_names
        with np.load(tmp_path / metadata["npz"], allow_pickle=False) as arrays:
            assert len(arrays.files) == 7
            for name in arrays.files:
                expected = getattr(plot, name)
                assert arrays[name].dtype == expected.dtype
                np.testing.assert_array_equal(arrays[name], expected)
                actual = getattr(restored, name)
                assert actual.dtype == expected.dtype
                assert actual.shape == expected.shape
                np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_record_json_tags_nonfinite_floats_not_literal_strings(
    tmp_path: Path,
    value: float,
) -> None:
    literals = (
        "NaN",
        "nan",
        "inf",
        "+inf",
        "-inf",
        "Infinity",
        "__nonfinite_float__",
    )
    records = replace(
        EMPTY,
        leaf_boxes=(
            LeafBoxRow(
                "Toy",
                0,
                2,
                1,
                "x",
                0,
                1,
                lower_inclusive=True,
                upper_inclusive=False,
                midpoint=value,
            ),
        ),
        execution_statuses=tuple(
            ExecutionStatusRow(
                "Toy",
                None,
                None,
                "reference",
                "failed",
                reason=literal,
            )
            for literal in literals
        ),
    )
    save_records(records, tmp_path)
    text = (tmp_path / "records.json").read_text()

    def reject_constant(token: str) -> None:
        pytest.fail(f"nonstandard JSON constant: {token}")

    saved = json.loads(text, parse_constant=reject_constant)
    token = "nan" if np.isnan(value) else ("+inf" if value > 0 else "-inf")
    assert saved["leaf_boxes"][0]["midpoint"] == {"__nonfinite_float__": token}
    restored = load_records(tmp_path)
    actual = restored.leaf_boxes[0].midpoint
    assert type(actual) is float
    assert np.isnan(actual) if np.isnan(value) else actual == value
    assert restored.execution_statuses == records.execution_statuses
    assert [row.reason for row in restored.execution_statuses] == list(
        literals,
    )
