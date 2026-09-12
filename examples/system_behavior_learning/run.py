from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from experiment import ExperimentRecords, run_experiment
from inference_run import run_inference
from numerical_threads import (
    THREAD_VARIABLES,
    numerical_thread_limit,
    validate_numerical_threads,
)
from record_io import save_records
from report import write_reports
from settings import DEVELOPMENT, FULL, Settings

if TYPE_CHECKING:
    from collections.abc import Sequence

    from experiment import SystemSpec


def _environment(source_dir: Path) -> dict[str, object]:
    head: str | None = None
    dirty: bool | None = None
    git = shutil.which("git")
    if git is not None:
        try:
            head = subprocess.check_output(  # noqa: S603 - fixed Git arguments
                [git, "rev-parse", "HEAD"],
                cwd=source_dir,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            dirty = bool(
                subprocess.check_output(  # noqa: S603 - fixed Git arguments
                    [git, "status", "--porcelain", "--untracked-files=all"],
                    cwd=source_dir,
                    stderr=subprocess.DEVNULL,
                    text=True,
                ).strip(),
            )
        except (OSError, subprocess.CalledProcessError):
            pass
    packages: dict[str, str | None] = {}
    for package in (
        "flowcean",
        "numpy",
        "scikit-learn",
        "scipy",
        "matplotlib",
    ):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "thread_environment": {
            name: os.environ[name]
            for name in THREAD_VARIABLES
            if name in os.environ
        },
        "source_head": head,
        "source_dirty": dirty,
    }


def run_and_write(
    settings: Settings,
    systems: Sequence[SystemSpec] | None = None,
    *,
    numerical_threads: int = 8,
) -> tuple[ExperimentRecords, tuple[Path, ...]]:
    """Use an exact new destination and preserve partial evidence on errors."""
    validate_numerical_threads(numerical_threads)
    with numerical_thread_limit(numerical_threads):
        return _run_and_write(settings, systems, numerical_threads)


def _run_and_write(
    settings: Settings,
    systems: Sequence[SystemSpec] | None,
    numerical_threads: int,
) -> tuple[ExperimentRecords, tuple[Path, ...]]:
    start = time.monotonic()
    run_dir = settings.output_dir
    run_dir.mkdir(parents=True, exist_ok=False)
    settings_path = run_dir / "settings.json"
    settings_path.write_text(
        json.dumps(settings.to_dict(), indent=2, allow_nan=False) + "\n",
    )
    source_dir = Path(__file__).resolve().parents[2]
    environment_path = run_dir / "environment.json"
    environment_path.write_text(
        json.dumps(_environment(source_dir), indent=2, allow_nan=False) + "\n",
    )
    lock_paths: tuple[Path, ...] = ()
    lock = source_dir / "uv.lock"
    if lock.is_file():
        lock_path = run_dir / "uv.lock"
        shutil.copyfile(lock, lock_path)
        lock_paths = (lock_path,)
    data_dir = run_dir / "data"
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True)
    print(
        "Running system-behavior experiments: "
        f"{len(settings.systems)} systems, {settings.replicates} replicates, "
        f"capacities {settings.capacities}, {settings.workers} workers",
        flush=True,
    )
    records = run_experiment(
        settings,
        systems,
        show_progress=True,
        raw_output_dir=raw_dir,
    )
    save_records(records, data_dir)
    results_dir = run_dir / "results"
    report_paths = write_reports(records, results_dir)
    inference_paths = (
        run_inference(
            settings,
            records,
            data_dir=data_dir,
            results_dir=results_dir,
            numerical_threads=numerical_threads,
        )
        if settings.statistics is not None
        else ()
    )
    paths = (
        settings_path,
        environment_path,
        *lock_paths,
        data_dir,
        *report_paths,
        *inference_paths,
    )
    elapsed = time.monotonic() - start
    print(f"Completed in {elapsed:.1f}s. Outputs:")
    for path in paths:
        print(f"  {path.resolve()}")
    return records, paths


def main(
    *,
    full: bool = False,
    workers: int | None = None,
    numerical_threads: int = 8,
) -> None:
    settings = FULL if full else DEVELOPMENT
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    settings = replace(
        settings,
        output_dir=settings.output_dir / f"experiment-{timestamp}",
        workers=settings.workers if workers is None else workers,
    )
    run_and_write(settings, numerical_threads=numerical_threads)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run system-behavior learning (development by default).",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="run all 30 histories and 31 geometric repetitions (expensive)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="number of simulation processes (1 for serial execution)",
    )
    parser.add_argument(
        "--numerical-threads",
        type=int,
        default=8,
        help="overall numerical-runtime thread cap (default: 8)",
    )
    args = parser.parse_args()
    main(
        full=args.full,
        workers=args.workers,
        numerical_threads=args.numerical_threads,
    )
