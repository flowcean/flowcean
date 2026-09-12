"""Inference phase over recorded raw data."""

from __future__ import annotations

from typing import TYPE_CHECKING

from experiment import _raw_system_name
from inference import infer_coverage
from inference_io import load_coverage_input, write_coverage_inference
from numerical_threads import validate_numerical_threads
from plots import write_coverage_ratios

if TYPE_CHECKING:
    from pathlib import Path

    from experiment import ExperimentRecords
    from inference import CoverageInference
    from settings import Settings

INFERENCE_FILES = (
    "method_scores.csv",
    "paired_effects.csv",
    "capacity_effects.csv",
    "bootstrap.npz",
    "metadata.json",
)


def run_inference(
    settings: Settings,
    records: ExperimentRecords,
    *,
    data_dir: Path,
    results_dir: Path,
    numerical_threads: int,
) -> tuple[Path, ...]:
    """Load, infer, persist and release one dense system before the next."""
    validate_numerical_threads(numerical_threads)
    configuration = settings.statistics
    if configuration is None:
        message = "statistical analysis requires settings.statistics"
        raise ValueError(message)
    destination = results_dir / "inference"
    destination.mkdir()
    results: list[CoverageInference] = []
    paths: list[Path] = []
    for index, system in enumerate(settings.systems):
        print(
            f"Inference {index + 1}/{len(settings.systems)}: {system.name}",
            flush=True,
        )
        data = load_coverage_input(
            settings,
            records,
            data_dir / "raw",
            system_index=index,
        )
        result = infer_coverage(data, configuration)
        # Results contain scores/draws, never dense reference vectors.
        del data
        paths.extend(
            write_coverage_inference(
                result,
                destination / _raw_system_name(index, system.name),
                numerical_threads=numerical_threads,
            ),
        )
        results.append(result)
    paths.append(write_coverage_ratios(results, results_dir))
    return tuple(paths)
