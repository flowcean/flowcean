"""Batch export benchmark traces to Parquet."""

import json
from pathlib import Path

from scenarios import SCENARIOS

from flowcean.hybrid import simulate


def main() -> None:
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    for spec in SCENARIOS:
        trajectory = simulate(
            spec.factory(),
            t_span=spec.t_span,
            input_stream=spec.input_stream,
        )
        path = output_dir / spec.name
        path.mkdir(parents=True, exist_ok=True)
        trajectory.sample(
            dt=0.01, include_inputs=spec.input_stream is not None
        ).write_parquet(path / "trace_0.parquet")
        (path / "trace_0.meta.json").write_text(
            json.dumps(
                {
                    "benchmark": spec.name,
                    "tags": list(spec.tags),
                    "description": spec.description,
                }
            ),
            encoding="utf-8",
        )
        print(f"exported {spec.name} -> {path}")


if __name__ == "__main__":
    main()
