from __future__ import annotations

import json
import shutil
import weakref
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import experiment
import inference_run
import numpy as np
import plots
import pytest
import run
from execution import execute_simulations
from experiment import ExperimentRecords, SystemSpec, _raw_system_name
from inference import CoverageInference, CoverageInput, infer_coverage
from inference_io import load_coverage_input, write_coverage_inference
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from settings import (
    DEVELOPMENT,
    FULL,
    INFERENCE_PRIMARY_COMPARATORS,
    InferenceSettings,
    ParameterRange,
    Settings,
    SystemSettings,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from execution import SimulationBatch
    from experiment import SimulationProgress

CONFIG = InferenceSettings(
    root_seed=82301,
    bootstrap_draws=17,
    confidence_level=0.8,
    batch_size=4,
)
SYSTEM = SystemSettings(
    "Toy",
    (ParameterRange("x", 0, 1),),
    (),
    (0, 1),
    ("value",),
)


def _tiny(path: Path, systems: int = 1) -> Settings:
    return Settings(
        systems=tuple(
            replace(SYSTEM, name=f"Toy {i}") for i in range(systems)
        ),
        root_seed=83201,
        replicates=2,
        fitting_size=6,
        assessment_size=4,
        reference_size=8,
        capacities=(2, 4),
        geometric_repetitions=2,
        trajectory_samples=3,
        workers=1,
        output_dir=path,
        statistics=CONFIG,
    )


def _simulate(
    _settings: SystemSettings,
    scenario: Mapping[str, float],
    times: np.ndarray,
) -> np.ndarray:
    return scenario["x"] + scenario["x"] ** 2 * times


def _runner(settings: Settings) -> tuple[ExperimentRecords, tuple[Path, ...]]:
    return run.run_and_write(
        settings,
        tuple(SystemSpec(s, _simulate) for s in settings.systems),
    )


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


@dataclass
class Published:
    settings: Settings
    records: ExperimentRecords
    path: Path


@pytest.fixture(scope="module")
def published(tmp_path_factory: pytest.TempPathFactory) -> Published:
    root = tmp_path_factory.mktemp("integrated")
    settings = _tiny(root / "outputs", systems=5)
    records: list[ExperimentRecords] = []
    order: list[int] = []
    dense: list[weakref.ReferenceType[np.ndarray]] = []
    original_load = inference_run.load_coverage_input

    with pytest.MonkeyPatch.context() as monkeypatch:

        def load(
            settings: Settings,
            records: ExperimentRecords,
            raw: Path,
            *,
            system_index: int,
        ) -> CoverageInput:
            assert all(reference() is None for reference in dense)
            order.append(system_index)
            data = original_load(
                settings,
                records,
                raw,
                system_index=system_index,
            )
            dense.extend(
                weakref.ref(e.distances) for e in data.methods.values()
            )
            return data

        monkeypatch.setattr(inference_run, "load_coverage_input", load)

        original_experiment = run.run_experiment

        def generate(*args: Any, **kwargs: Any) -> ExperimentRecords:
            saved = _json(settings.output_dir / "settings.json")
            assert saved["statistics"] == {
                "root_seed": 82301,
                "bootstrap_draws": 17,
                "confidence_level": 0.8,
                "batch_size": 4,
            }
            result = original_experiment(*args, **kwargs)

            def forbidden(*_args: object, **_kwargs: object) -> None:
                pytest.fail("simulation after raw generation")

            monkeypatch.setattr(
                experiment,
                "simulate_scenario_batch",
                forbidden,
            )
            return result

        monkeypatch.setattr(run, "run_experiment", generate)
        records.append(_runner(settings)[0])
        path = settings.output_dir
    assert order == [0, 1, 2, 3, 4]
    assert all(reference() is None for reference in dense)
    return Published(settings, records[0], path)


@pytest.fixture
def copied(published: Published, tmp_path: Path) -> Path:
    return Path(shutil.copytree(published.path, tmp_path / "relocated"))


def test_integrated_outputs_reconstruct_and_relocate(
    published: Published,
    copied: Path,
    tmp_path: Path,
) -> None:
    bound = {
        path.relative_to(copied / "results").as_posix()
        for path in (copied / "results").rglob("*")
        if path.is_file()
    }
    assert "coverage_ratios.png" in bound
    assert not plt.get_fignums()
    for index, system in enumerate(published.settings.systems):
        directory = "inference/" + _raw_system_name(index, system.name)
        assert {
            f"{directory}/{name}" for name in inference_run.INFERENCE_FILES
        } <= bound
        loaded = load_coverage_input(
            published.settings,
            published.records,
            copied / "data/raw",
            system_index=index,
        )
        result = infer_coverage(loaded, CONFIG)
        assert result.settings == CONFIG
        for method in ("random", "lhs"):
            assert result.comparisons[method].band.status == "supporting"
            assert result.comparisons[
                method
            ].band.bootstrap_log_effects.shape == (0, 2)
        metadata = _json(copied / "results" / directory / "metadata.json")
        assert metadata["format_version"] == 2
        assert metadata["label"] == "EXPERIMENT"
        assert metadata["numerical_threads"] == 8
        expected = write_coverage_inference(
            result,
            tmp_path / f"reconstruct-{index}",
            numerical_threads=8,
        )
        for path in expected:
            actual = copied / "results" / directory / path.name
            if path.suffix == ".npz":
                with (
                    np.load(path, allow_pickle=False) as left,
                    np.load(actual, allow_pickle=False) as right,
                ):
                    assert left.files == right.files
                    for key in left.files:
                        np.testing.assert_array_equal(left[key], right[key])
            else:
                assert actual.read_bytes() == path.read_bytes()


@pytest.mark.parametrize("failure", ["load", "write", "geometry"])
def test_inference_errors_retain_raw_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    settings = _tiny(tmp_path / "outputs")

    def fail(*_args: object, **_kwargs: object) -> None:
        message = "injected inference error"
        raise ValueError(message)

    if failure != "geometry":
        monkeypatch.setattr(
            inference_run,
            "load_coverage_input"
            if failure == "load"
            else "write_coverage_inference",
            fail,
        )

    original_experiment = run.run_experiment

    def generate(*args: Any, **kwargs: Any) -> ExperimentRecords:
        result = original_experiment(*args, **kwargs)
        if failure == "geometry":
            path = settings.output_dir / "data/raw/00_toy_0/shared.npz"
            with np.load(path, allow_pickle=False) as archive:
                arrays = {name: archive[name] for name in archive.files}
            del arrays["suite_sobol__budget_002__repetition_000__scenarios"]
            np.savez_compressed(path, **arrays)
        return result

    monkeypatch.setattr(run, "run_experiment", generate)
    with pytest.raises(ValueError, match=r"injected inference|raw|physical"):
        _runner(settings)
    assert (
        settings.output_dir / "data/raw/00_toy_0/replicate_000.npz"
    ).exists()
    assert (settings.output_dir / "data/records.json").exists()
    assert (settings.output_dir / "results/summary.csv").exists()


def test_configuration_seeds() -> None:
    assert FULL.root_seed == 2237213407657127
    assert FULL.statistics is not None
    assert FULL.statistics.root_seed == 2210151554901251
    assert DEVELOPMENT.root_seed == 2027
    assert DEVELOPMENT.statistics is not None
    assert DEVELOPMENT.statistics.root_seed == 944701
    assert Settings().root_seed == 2027


def _declare_failure(monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    batches = 0
    failures = {"reference": 0, "assessment": 2, "known_b": 4, "geometric": 6}

    def adapter(
        spec: SystemSpec,
        scenarios: np.ndarray,
        samples: int,
        progress: SimulationProgress | None = None,
    ) -> SimulationBatch:
        nonlocal batches
        index = batches
        batches += 1
        row_index = 0

        def simulate(row: np.ndarray) -> np.ndarray:
            nonlocal row_index
            row_index += 1
            if index == failures.get(failure) and row_index == 1:
                message = "declared synthetic failure"
                raise RuntimeError(message)
            return spec.simulate_scenario(row, samples)

        return execute_simulations(scenarios, simulate, samples, progress)

    monkeypatch.setattr(experiment, "simulate_scenario_batch", adapter)
    if failure == "unknown_b":
        original = experiment.fit_behavior_tree
        fits = 0

        def fit(*args: Any, **kwargs: Any) -> Any:
            nonlocal fits
            fits += 1
            if fits == 1:
                message = "declared synthetic fit failure"
                raise ValueError(message)
            return original(*args, **kwargs)

        monkeypatch.setattr(experiment, "fit_behavior_tree", fit)


@pytest.mark.parametrize(
    "failure",
    ["reference", "assessment", "unknown_b", "known_b", "geometric"],
)
def test_declared_failures_complete_independent_analyses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    settings = _tiny(tmp_path / "outputs")
    _declare_failure(monkeypatch, failure)
    captured: list[CoverageInference] = []
    original_infer = inference_run.infer_coverage

    def infer(
        data: CoverageInput,
        cfg: InferenceSettings,
    ) -> CoverageInference:
        result = original_infer(data, cfg)
        captured.append(result)
        return result

    monkeypatch.setattr(inference_run, "infer_coverage", infer)
    _runner(settings)
    path = settings.output_dir / "results"
    (result,) = captured
    assert (path / "coverage_ratios.png").is_file()
    assert not plt.get_fignums()
    if failure == "assessment":
        assert all(
            point.log_effect is not None
            for comparison in result.comparisons.values()
            for point in comparison.points
        )
        assert all(
            result.comparisons[method].band.status == "available"
            for method in INFERENCE_PRIMARY_COMPARATORS
        )
    elif failure == "reference":
        assert all(
            point.log_effect is None
            for comparison in result.comparisons.values()
            for point in comparison.points
        )
        assert (path / "tree_metrics.csv").stat().st_size > 500
    else:
        comparison = result.comparisons["sobol"]
        assert comparison.band.status == "incomplete_points"
        assert comparison.points[0].log_effect is None
        assert comparison.points[1].log_effect is not None
        assert comparison.band.bootstrap_log_effects.shape == (0, 2)
        if failure == "unknown_b":
            assert result.realized_budgets[0, 0] == 0
        elif failure == "known_b":
            assert result.realized_budgets[0, 0] == 2
            assert np.isfinite(result.methods["archive_pam"].history_q).all()
        else:
            assert result.comparisons["archive_pam"].band.status == "available"
            assert np.isfinite(result.methods["sobol"].repetition_q).any()


@pytest.fixture
def result(published: Published) -> CoverageInference:
    data = load_coverage_input(
        published.settings,
        published.records,
        published.path / "data/raw",
        system_index=0,
    )
    return infer_coverage(data, CONFIG)


def _plot_case(result: CoverageInference, case: str) -> CoverageInference:
    comparisons = dict(result.comparisons)
    for method, comparison in comparisons.items():
        if case in {"partial", "allmissing", "supporting_only"}:
            affected = (
                case != "supporting_only"
                or method in INFERENCE_PRIMARY_COMPARATORS
            )
            if affected:
                points = tuple(
                    replace(
                        point,
                        log_effect=None,
                        ratio=None,
                        reason="missing evidence",
                    )
                    if case != "partial" or i == 0
                    else point
                    for i, point in enumerate(comparison.points)
                )
                comparisons[method] = replace(
                    comparison,
                    points=points,
                    band=replace(comparison.band, status="incomplete_points"),
                )
        elif case == "overflow":
            comparisons[method] = replace(
                comparison,
                points=tuple(
                    replace(point, ratio=None, ratio_reason="ratio_overflow")
                    for point in comparison.points
                ),
                band=replace(
                    comparison.band,
                    ratio_lower=np.full(2, np.nan),
                    ratio_upper=np.full(2, np.nan),
                ),
            )
        elif case == "no_bands":
            comparisons[method] = replace(
                comparison,
                band=replace(comparison.band, status="undefined_bootstrap"),
            )
    return replace(result, comparisons=comparisons)


@pytest.mark.parametrize(
    "case",
    [
        "normal",
        "partial",
        "allmissing",
        "supporting_only",
        "overflow",
        "no_bands",
    ],
)
def test_ratio_plot_gaps_labels_and_closed_files(
    result: CoverageInference,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    result = _plot_case(result, case)
    figures: list[Any] = []
    close = plt.close

    def capture(figure: Any = None) -> None:
        if figure is not None and not isinstance(figure, str):
            figures.append(figure)
        close(figure)

    monkeypatch.setattr(plt, "close", capture)
    path = plots.write_coverage_ratios([result], tmp_path)
    assert path.name == "coverage_ratios.png"
    assert path.stat().st_size > 1000
    assert not plt.get_fignums()
    (figure,) = figures
    axis = figure.axes[0]
    texts = "\n".join(
        [
            *(t.get_text() for t in figure.texts),
            axis.get_title(),
            *(t.get_text() for t in axis.texts),
        ],
    )
    assert "DEVELOPMENT ONLY" not in texts
    assert "80%" in texts
    assert "95%" not in texts
    assert "Not jointly simultaneous" in texts
    assert "nan" not in texts.lower()
    assert "inf" not in texts.lower()
    assert axis.get_xticks().tolist() == list(result.capacities)
    for line in axis.lines[:3]:
        values = np.asarray(line.get_ydata(), dtype=float)
        if case in {"allmissing", "supporting_only", "overflow"}:
            assert np.isnan(values).all()
        elif case == "partial":
            assert np.isnan(values[0])
            assert np.isfinite(values[1])
        else:
            assert np.isfinite(values).all()
    if case == "overflow":
        assert "ratio out of range" in texts
    if case == "no_bands":
        assert not axis.collections
        assert "band unavailable" in texts


def test_plot_closes_after_write_error(
    result: CoverageInference,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*_args: object, **_kwargs: object) -> None:
        message = "synthetic write failure"
        raise OSError(message)

    monkeypatch.setattr(Figure, "savefig", fail)
    with pytest.raises(OSError, match="synthetic write"):
        plots.write_coverage_ratios([result], tmp_path)
    assert not plt.get_fignums()
