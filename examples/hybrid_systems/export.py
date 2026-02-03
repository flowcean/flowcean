"""Batch export benchmark traces to Parquet."""

from pathlib import Path

import polars as pl
from benchmarks import all_specs

from flowcean.ode import simulate
from flowcean.ode.io import traces_to_polars


def main() -> None:
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    for spec in all_specs():
        trace = simulate(spec.factory(), t_span=spec.t_span, sample_dt=0.01)
        df = traces_to_polars([trace])
        df = df.with_columns(
            pl.lit(spec.name).alias("benchmark"),
            pl.lit(",".join(spec.tags)).alias("tags"),
            pl.lit(spec.description).alias("description"),
        )
        path = output_dir / f"{spec.name}.parquet"
        df.write_parquet(path)
        print(f"exported {spec.name} -> {path}")


if __name__ == "__main__":
    main()
