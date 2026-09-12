from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import Mock

import execution
import experiment
import methods
import metrics
import numpy as np
import pytest
import report
import run
from experiment import ExperimentRecords, SystemSpec, TargetTransform
from record_io import ROW_TYPES, load_records, save_records
from report import report_saved_run
from settings import (
    DEVELOPMENT,
    FULL,
    InferenceSettings,
    ParameterRange,
    Settings,
    SystemSettings,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

EMPTY = ExperimentRecords((), (), (), (), (), (), (), (), ())


@pytest.mark.parametrize("preset", [FULL, DEVELOPMENT, Settings()])
def test_settings_json_roundtrip_without_randomness(
    preset: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forbidden = Mock(side_effect=AssertionError("unexpected RNG call"))
    monkeypatch.setattr(np.random, "SeedSequence", forbidden)
    monkeypatch.setattr(np.random, "default_rng", forbidden)
    saved = json.loads(json.dumps(preset.to_dict()))
    restored = Settings.from_dict(saved)
    assert restored == preset
    assert json.loads(json.dumps(restored.to_dict())) == saved
    assert isinstance(restored.output_dir, Path)
    for system in restored.systems:
        assert isinstance(system.scenario_parameters, tuple)
        assert isinstance(system.fixed_parameters, tuple)
        assert all(isinstance(pair, tuple) for pair in system.fixed_parameters)
        assert isinstance(system.horizon, tuple)
        assert isinstance(system.state_names, tuple)
    forbidden.assert_not_called()


def test_settings_require_saved_fields_and_validate_domains() -> None:
    saved = json.loads(json.dumps(FULL.to_dict()))
    for key in saved:
        incomplete = saved.copy()
        del incomplete[key]
        with pytest.raises(KeyError, match=key):
            Settings.from_dict(incomplete)
    del saved["statistics"]["bootstrap_draws"]
    with pytest.raises(KeyError, match="bootstrap_draws"):
        Settings.from_dict(saved)
    saved = json.loads(json.dumps(FULL.to_dict()))
    saved["systems"][0]["scenario_parameters"][0]["upper"] = -1
    with pytest.raises(ValueError, match="lower < upper"):
        Settings.from_dict(saved)


@pytest.fixture
def empty_run(tmp_path: Path) -> Path:
    root = tmp_path / "run"
    data = root / "data"
    data.mkdir(parents=True)
    save_records(EMPTY, data)
    (root / "settings.json").write_text(json.dumps(Settings().to_dict()))
    return root


def test_empty_prototypes_and_fresh_default_destination(
    empty_run: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forbidden = Mock(side_effect=AssertionError("unexpected inference"))
    monkeypatch.setattr(report, "run_inference", forbidden)
    assert load_records(empty_run / "data") == EMPTY
    assert not (empty_run / "data/prototypes").exists()
    paths = report_saved_run(empty_run)
    assert paths
    assert all(path.is_file() for path in paths)
    before = hashes(empty_run)
    with pytest.raises(FileExistsError):
        report_saved_run(empty_run)
    assert hashes(empty_run) == before
    forbidden.assert_not_called()


@pytest.mark.parametrize("collection", [*ROW_TYPES, "prototype_plots"])
def test_missing_collection_fails_before_creating_results(
    empty_run: Path,
    collection: str,
) -> None:
    path = empty_run / "data/records.json"
    saved = json.loads(path.read_text())
    del saved[collection]
    path.write_text(json.dumps(saved))
    with pytest.raises(KeyError, match=collection):
        report_saved_run(empty_run)
    assert not (empty_run / "results").exists()


@pytest.mark.parametrize("missing", ["settings.json", "data/records.json"])
def test_missing_file_fails_before_creating_results(
    empty_run: Path,
    missing: str,
) -> None:
    (empty_run / missing).unlink()
    with pytest.raises(FileNotFoundError):
        report_saved_run(empty_run)
    assert not (empty_run / "results").exists()


@pytest.mark.parametrize("corrupt", ["{", '{"tree_metrics": {}}'])
def test_malformed_records_do_not_become_empty(
    empty_run: Path,
    corrupt: str,
) -> None:
    (empty_run / "data/records.json").write_text(corrupt)
    with pytest.raises((ValueError, TypeError)):
        report_saved_run(empty_run)
    assert not (empty_run / "results").exists()


@pytest.mark.parametrize(
    "problem",
    ["missing_file", "missing_array", "pickle"],
)
def test_unreadable_prototype_fails_before_creating_results(
    empty_run: Path,
    problem: str,
) -> None:
    path = empty_run / "data/records.json"
    saved = json.loads(path.read_text())
    saved["prototype_plots"] = [
        {
            "system": "Toy",
            "replicate": 0,
            "capacity": 2,
            "state_names": ["z", "a"],
            "npz": "prototypes/selected.npz",
        },
    ]
    path.write_text(json.dumps(saved))
    if problem != "missing_file":
        destination = empty_run / "data/prototypes"
        destination.mkdir()
        arrays = (
            {}
            if problem == "missing_array"
            else {
                "sample_times": np.array([object()], dtype=object),
            }
        )
        np.savez_compressed(
            destination / "selected.npz",
            **cast("dict[str, Any]", arrays),
        )
    with pytest.raises((FileNotFoundError, KeyError, ValueError)):
        report_saved_run(empty_run)
    assert not (empty_run / "results").exists()


@pytest.mark.parametrize(
    "destination",
    ["data", "data/nested/results", "link/new"],
)
def test_data_destination_is_protected(
    empty_run: Path,
    destination: str,
) -> None:
    (empty_run / "link").symlink_to(
        empty_run / "data",
        target_is_directory=True,
    )
    before = hashes(empty_run)
    with pytest.raises(ValueError, match="inside recorded data"):
        report_saved_run(empty_run, empty_run / destination)
    assert hashes(empty_run) == before
    assert not (empty_run / "data/nested").exists()
    assert not (empty_run / "data/new").exists()


def hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(
            path.read_bytes(),
        ).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


@pytest.fixture(params=[False, True], ids=["descriptive", "statistics"])
def synthetic_run(request: pytest.FixtureRequest, tmp_path: Path) -> Path:
    system = SystemSettings(
        "Toy",
        (ParameterRange("x", 0, 1), ParameterRange("y", -1, 1)),
        (),
        (0, 1),
        ("linear", "quadratic"),
    )

    def simulator(
        _settings: SystemSettings,
        scenario: Mapping[str, float],
        times: np.ndarray,
    ) -> np.ndarray:
        return np.column_stack(
            (
                scenario["x"] + times * scenario["y"],
                scenario["y"] + times**2 * scenario["x"],
            ),
        ).reshape(-1)

    settings = Settings(
        systems=(system,),
        root_seed=419,
        replicates=2,
        fitting_size=2,
        assessment_size=1,
        reference_size=8,
        capacities=(2, 4),
        geometric_repetitions=3,
        trajectory_samples=4,
        prototype_plot_replicate=0,
        prototype_plot_capacity=2,
        workers=1,
        output_dir=tmp_path / "original",
        statistics=InferenceSettings(
            root_seed=317,
            bootstrap_draws=19,
            batch_size=4,
        )
        if request.param
        else None,
    )
    records, _ = run.run_and_write(settings, (SystemSpec(system, simulator),))
    restored = load_records(settings.output_dir / "data")
    for name in ROW_TYPES:
        assert getattr(records, name)
        assert getattr(restored, name) == getattr(records, name)
    assert len(restored.prototype_plots) == 1
    # Only the directory survives; no in-memory evidence is reused.
    return settings.output_dir


def test_relocated_disk_reporting_preserves_evidence_and_results(
    synthetic_run: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = hashes(synthetic_run)
    relocated = synthetic_run.rename(tmp_path / "relocated")
    assert not synthetic_run.exists()
    forbidden = Mock(side_effect=AssertionError("evidence regeneration"))
    for module, names in (
        (run, ("run_experiment",)),
        (execution, ("execute_simulations",)),
        (
            experiment,
            (
                "run_experiment",
                "build_system_specs",
                "simulate",
                "simulate_scenario_batch",
                "execute_simulations",
                "fit_target_transform",
                "fit_behavior_tree",
                "fit_unbounded_behavior_tree",
                "fit_input_only_tree",
                "pam_medoids",
                "sample_unit_suite",
                "tree_midpoints",
                "pairwise_coordinate_rms",
                "reference_coverage_metrics",
                "_prototype_plot_data",
            ),
        ),
        (
            methods,
            (
                "fit_behavior_tree",
                "fit_unbounded_behavior_tree",
                "fit_input_only_tree",
                "pam_medoids",
                "sample_unit_suite",
            ),
        ),
        (
            metrics,
            (
                "pairwise_coordinate_rms",
                "cross_coordinate_rms",
                "reference_coverage_metrics",
                "_distance_matrix",
            ),
        ),
        (TargetTransform, ("transform",)),
        (SystemSpec, ("simulate_scenario",)),
    ):
        for name in names:
            monkeypatch.setattr(module, name, forbidden)
    original_inference = report.run_inference
    observed_settings = []

    def inference(
        settings: Settings,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[Path, ...]:
        assert settings.output_dir == relocated
        assert kwargs["numerical_threads"] == 4
        observed_settings.append(settings)
        return original_inference(settings, *args, **kwargs)

    monkeypatch.setattr(report, "run_inference", inference)
    destination = relocated / "restyled"
    paths = report_saved_run(relocated, destination, numerical_threads=4)
    initial = relocated / "results"
    assert {path.relative_to(destination) for path in paths} == {
        path.relative_to(initial)
        for path in initial.rglob("*")
        if path.is_file()
    }
    for path in initial.rglob("*"):
        if not path.is_file():
            continue
        actual = destination / path.relative_to(initial)
        if path.name == "metadata.json":
            expected_metadata = json.loads(path.read_text())
            actual_metadata = json.loads(actual.read_text())
            assert expected_metadata.pop("numerical_threads") == 8
            assert actual_metadata.pop("numerical_threads") == 4
            assert actual_metadata == expected_metadata
        elif path.suffix in {".csv", ".json"}:
            assert actual.read_bytes() == path.read_bytes()
        elif path.suffix == ".npz":
            with (
                np.load(path, allow_pickle=False) as expected,
                np.load(actual, allow_pickle=False) as restored,
            ):
                assert restored.files == expected.files
                curves = restored["sobol__log_curves"]
                assert curves.shape == (19, 2)
                assert np.all(np.isfinite(curves))
                assert np.ptp(curves) > 0
                for name in expected.files:
                    assert restored[name].dtype == expected[name].dtype
                    np.testing.assert_array_equal(
                        restored[name],
                        expected[name],
                    )
    configured = json.loads((relocated / "settings.json").read_text())[
        "statistics"
    ]
    assert bool(observed_settings) == (configured is not None)
    assert {
        name: digest
        for name, digest in hashes(relocated).items()
        if not name.startswith("restyled/")
    } == before
    forbidden.assert_not_called()
