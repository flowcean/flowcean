from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from collections.abc import Sequence

if __package__:
    from .experiment import SystemEvaluation, run_experiment
else:
    from experiment import (  # pyright: ignore[reportImplicitRelativeImport]
        SystemEvaluation,
        run_experiment,
    )

DEFAULT_OUTPUT_DIR = Path(__file__).parent / "outputs"


def _capacity_label(capacity: int | None) -> str:
    return "unbounded" if capacity is None else str(capacity)


def save_plot(
    evaluations: Sequence[SystemEvaluation],
    path: Path,
) -> None:
    unbounded_position = 17
    figure, axis = plt.subplots(figsize=(9, 4.5))
    for evaluation in evaluations:
        bounded = [
            result
            for result in evaluation.capacity_results
            if result.max_leaf_nodes is not None
        ]
        unbounded = next(
            result
            for result in evaluation.capacity_results
            if result.max_leaf_nodes is None
        )
        capacities = [
            result.max_leaf_nodes
            for result in bounded
            if result.max_leaf_nodes is not None
        ]
        (line,) = axis.plot(
            capacities,
            [result.assessment_rmse_ratio for result in bounded],
            marker="o",
            markersize=4,
            label=evaluation.prepared.spec.name,
        )
        axis.plot(
            [capacities[-1], unbounded_position],
            [
                bounded[-1].assessment_rmse_ratio,
                unbounded.assessment_rmse_ratio,
            ],
            color=line.get_color(),
            linestyle=":",
        )
        axis.scatter(
            [unbounded_position],
            [unbounded.assessment_rmse_ratio],
            color=line.get_color(),
            marker="X",
            s=70,
            zorder=3,
        )

    axis.scatter(
        [],
        [],
        color="0.25",
        marker="X",
        s=70,
        label="Unbounded tree",
    )
    axis.axhline(
        1.0,
        color="0.35",
        linestyle="--",
        label="Historical mean (ratio 1.0)",
    )
    positions = [*range(2, 17), unbounded_position]
    axis.set_xticks(
        positions,
        [*(str(value) for value in range(2, 17)), "unbounded"],
        rotation=30,
        ha="right",
    )
    axis.set_xlabel("Configured max leaf nodes")
    axis.set_ylabel("Assessment RMSE ratio")
    axis.set_title("Complete-state behavior learning by tree capacity")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(ncol=2)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def print_summaries(evaluations: Sequence[SystemEvaluation]) -> None:
    for evaluation in evaluations:
        summary = evaluation.summary
        print(
            f"{summary.system}: lowest observed ratio "
            f"{summary.lowest_observed_rmse_ratio:.4f} at "
            f"{_capacity_label(summary.lowest_observed_max_leaf_nodes)}; "
            f"saturation at "
            f"{_capacity_label(summary.saturation_max_leaf_nodes)} "
            f"({summary.saturation_fitted_leaf_count} fitted leaves)",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Learn complete simulated behavior with decision trees.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="directory for report.json and performance.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    plot_path = output_dir / "performance.png"

    report, evaluations = run_experiment()
    report_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    save_plot(evaluations, plot_path)
    print_summaries(evaluations)
    print(f"Report: {report_path}")
    print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()
