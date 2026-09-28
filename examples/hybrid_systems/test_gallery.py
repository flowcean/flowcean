"""Verify gallery summaries and sampled exports with metadata sidecars."""

import json
import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent))

import export as benchmark_export
import run as benchmark_run
from scenarios import SCENARIOS

from flowcean.hybrid import simulate


def test_summary_counts_locations_between_sampled_rows() -> None:
    spec = next(s for s in SCENARIOS if s.name == "Buck Converter")
    trajectory = simulate(spec.factory(), spec.t_span)

    assert trajectory.sample(dt=0.01)["location_id"].n_unique() < 3
    summary = benchmark_run.summarize_benchmark(spec, trajectory)
    assert summary.location_count == 3


@pytest.mark.parametrize("has_input", [False, True])
def test_export_writes_sampled_parquet_and_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, has_input: bool
) -> None:
    spec = next(
        scenario
        for scenario in SCENARIOS
        if (scenario.input_stream is not None) == has_input
    )
    monkeypatch.setattr(benchmark_export, "SCENARIOS", (spec,))
    monkeypatch.chdir(tmp_path)

    benchmark_export.main()

    path = tmp_path / "output" / spec.name
    frame = pl.read_parquet(path / "trace_0.parquet")
    assert frame["t"][0] == spec.t_span[0]
    assert frame["t"][-1] == spec.t_span[1]
    assert ("u0" in frame.columns) == has_input
    assert frame["location_time"][0] == 0
    assert json.loads(
        (path / "trace_0.meta.json").read_text(encoding="utf-8")
    ) == {
        "benchmark": spec.name,
        "tags": list(spec.tags),
        "description": spec.description,
    }
