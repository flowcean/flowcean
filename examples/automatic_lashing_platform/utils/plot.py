from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
from matplotlib.ticker import FuncFormatter

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
MAX_CYLINDER_NUMBER = 4
DEFAULT_TIME_LIMIT = 4.0


RESULTS_FILE = RESULTS_DIR / "results_regression_tree.csv"
RESULTS_SOURCES = [
    (
        RESULTS_DIR / "results_regression_tree_^p_accumulator_[0-9]*$.csv",
        "p_{acc}",
    ),
    (
        RESULTS_DIR
        / (
            "results_regression_tree_^p_accumulator_[0-9]*$_T_active_valve_count.csv"
        ),
        "p_{acc}, T, n_{valves}",
    ),
    (
        RESULTS_DIR
        / "results_regression_tree_^p_accumulator_derivative_[0-9]*$.csv",
        "p'",
    ),
    (
        RESULTS_DIR
        / (
            "results_regression_tree_p_accumulator_0_"
            "^p_accumulator_derivative_[0-9]*$.csv"
        ),
        "p', p_0",
    ),
    (
        RESULTS_DIR
        / (
            "results_regression_tree_^p_accumulator_derivative_[0-9]*$"
            "_T_active_valve_count.csv"
        ),
        "p', T, n_{valves}",
    ),
]

yfmt = FuncFormatter(lambda value, _: f"{value / 1000:.0f}")


def as_mathtext(label: str) -> str:
    stripped = label.strip()
    if stripped.startswith("$") and stripped.endswith("$"):
        return stripped
    return rf"${stripped}$"


def build_regression_tree_results_csv(
    output_file: Path = RESULTS_FILE,
) -> pl.DataFrame:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    rows: list[pl.DataFrame] = []
    for source_file, features in RESULTS_SOURCES:
        if not source_file.exists():
            continue
        rows.append(
            pl.read_csv(source_file.open("rb"))
            .with_columns(pl.lit(features).alias("features"))
            .select(["features", "depth", "MAE", "MSE", "MAPE"]),
        )

    combined = (
        pl.concat(rows, how="vertical")
        if rows
        else pl.DataFrame(
            schema={
                "features": pl.String,
                "depth": pl.Int64,
                "MAE": pl.Float64,
                "MSE": pl.Float64,
                "MAPE": pl.Float64,
            },
        )
    )
    combined.write_csv(output_file)
    return combined


def plot_alp_pressures(
    weight: float,
    cylinders: list[int] | None = None,
    time_limit: float = DEFAULT_TIME_LIMIT,
) -> None:
    """Plot the pressure curves for a given container weight.

    Args:
        weight: The container weight to filter the data by.
        cylinders: A list of cylinder numbers to include in the plot (1-4).
        If None, only the accumulator pressure will be plotted.
        time_limit: The maximum time (in seconds) to include.
    """
    data = (
        pl.scan_parquet(DATA_DIR / "alp_sim_data.parquet")
        .filter(pl.col("active_valve_count") > 0)
        .filter(pl.col("container_weight").is_close(weight, abs_tol=100))
        .head(1)
        .collect(engine="streaming")
        .lazy()
    )
    p_accumulator = (
        data.select(
            pl.col("p_accumulator").list.explode().struct.unnest(),
        )
        .collect()
        .filter(pl.col("time") <= time_limit)
    )
    p_accumulator.write_csv(RESULTS_DIR / "pressure_curve.csv")
    print(f"container_weight: {data.select('container_weight').collect()}")
    print(f"T: {data.select('T').collect()}")
    print(f"Active Valve Count: {data.select('active_valve_count').collect()}")
    p_init = data.select(pl.col("p_accumulator").list.first()).collect()
    print(f"p_init: {p_init}")

    _, ax = plt.subplots(layout="constrained")
    ax.plot(
        p_accumulator["time"],
        p_accumulator["value"],
        label="Accumulator",
    )
    if cylinders is not None:
        for cylinder in cylinders:
            if not (1 <= cylinder <= MAX_CYLINDER_NUMBER):
                msg = f"Cylinder must be between 1 and {MAX_CYLINDER_NUMBER}"
                raise ValueError(msg)
            ax.plot(
                data.select(
                    pl.col(f"p_cylinder{cylinder}")
                    .list.explode()
                    .struct.unnest(),
                )
                .collect()
                .filter(pl.col("time") <= time_limit)["time"],
                data.select(
                    pl.col(f"p_cylinder{cylinder}")
                    .list.explode()
                    .struct.unnest(),
                )
                .collect()
                .filter(pl.col("time") <= time_limit)["value"],
                label=f"Cylinder {cylinder}",
            )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pressure [bar]")
    ax.yaxis.set_major_formatter(yfmt)
    ax.grid(visible=True, which="both", linestyle="--", linewidth=0.5)
    ax.legend()
    plt.show()


palette = ["tab:blue", "tab:orange", "tab:red", "tab:green", "tab:purple"]
markers = [".", ".", ".", "1", "2"]
styles = ["-", "-", "-", "--", "-."]


def plot_performances() -> None:
    results = build_regression_tree_results_csv()
    _, ax = plt.subplots(layout="constrained")
    for i, features in enumerate(
        results["features"].unique(maintain_order=True),
    ):
        subset = results.filter(pl.col("features") == features)
        ax.plot(
            subset["depth"],
            subset["MAE"],
            color=palette[i % len(palette)],
            marker=markers[i % len(markers)],
            linestyle=styles[i % len(styles)],
            markersize=14,
            label=as_mathtext(features),
        )
    ax.yaxis.set_major_formatter(yfmt)
    ax.set_xlabel("Tree Depth")
    ax.set_ylabel("Mean Absolute Error (MAE) [t]")
    ax.set_title("Regression Tree Performance on ALP Data")
    ax.grid(visible=True, which="both", linestyle="--", linewidth=0.5)
    ax.legend()
    plt.show()


if __name__ == "__main__":
    plot_alp_pressures(weight=45000, cylinders=[1])
    plot_performances()
