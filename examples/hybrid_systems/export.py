"""Batch export benchmark traces to Parquet."""

from pathlib import Path

from benchmarks import all_specs

from flowcean.ode import simulate
from flowcean.ode.io import save_traces_parquet


def main() -> None:
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    for spec in all_specs():
        trace = simulate(
            spec.factory(),
            t_span=spec.t_span,
            input_stream=spec.input_stream,
            sample_dt=0.01,
        )
        path = output_dir / spec.name
        save_traces_parquet(
            [trace],
            str(path),
            trace_metadata=[
                {
                    "benchmark": spec.name,
                    "tags": list(spec.tags),
                    "description": spec.description,
                },
            ],
        )
        print(f"exported {spec.name} -> {path}")


if __name__ == "__main__":
    main()
