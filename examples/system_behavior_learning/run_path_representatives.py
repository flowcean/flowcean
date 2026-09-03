from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.figure import Figure

if __package__:
    from .path_representatives import (
        PathRepresentativeEvaluation,
        PathRepresentativeStudy,
        inverse_transform_complete_targets,
        run_path_representatives,
    )
else:
    from path_representatives import (  # pyright: ignore[reportImplicitRelativeImport]
        PathRepresentativeEvaluation,
        PathRepresentativeStudy,
        inverse_transform_complete_targets,
        run_path_representatives,
    )

REPORT_NAME = "path_representatives.json"
PLOT_NAME = "path_representatives.png"
PROTOTYPE_NAMES = {
    "Thermostat": "path_prototypes_thermostat.png",
    "Bouncing Ball": "path_prototypes_bouncing_ball.png",
    "Hybrid Oscillator": "path_prototypes_hybrid_oscillator.png",
    "Tank Valves": "path_prototypes_tank_valves.png",
}
MIDPOINT_COLOR = "#e36c0a"
PROTOTYPE_COLOR = "#1f77b4"


def create_combined_figure(study: PathRepresentativeStudy) -> Figure:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8.5), squeeze=False)
    for axis, evaluation in zip(
        axes.flat,
        study.evaluations,
        strict=False,
    ):
        leaves = evaluation.leaves
        positions = list(range(1, len(leaves) + 1))
        boxes = axis.boxplot(
            [leaf.held_residuals for leaf in leaves],
            positions=positions,
            orientation="horizontal",
            tick_labels=[str(leaf.region.leaf_id) for leaf in leaves],
            patch_artist=True,
            widths=0.6,
        )
        for box in boxes["boxes"]:
            box.set_facecolor("0.82")
            box.set_edgecolor("0.35")
        axis.scatter(
            [leaf.midpoint_error for leaf in leaves],
            positions,
            color=MIDPOINT_COLOR,
            edgecolor="white",
            marker="D",
            s=42,
            zorder=3,
        )
        axis.axvline(
            evaluation.volume_weighted_midpoint_error,
            color=MIDPOINT_COLOR,
            linestyle=":",
        )
        axis.axvline(
            evaluation.volume_weighted_held_mean,
            color="0.2",
            linestyle="--",
        )
        axis.set_xlabel(
            "Standardized-coordinate RMS distance\nto fitting prototype",
        )
        axis.set_ylabel("Tree leaf ID")
        axis.set_title(evaluation.prepared.spec.name)
        axis.grid(axis="x", alpha=0.2)

    for axis in axes.flat[len(study.evaluations) :]:
        axis.set_visible(False)
    handles = [
        Patch(facecolor="0.82", edgecolor="0.35", label="Held residuals"),
        Line2D(
            [],
            [],
            color=MIDPOINT_COLOR,
            marker="D",
            linestyle="None",
            label="Path midpoint",
        ),
        Line2D(
            [],
            [],
            color=MIDPOINT_COLOR,
            linestyle=":",
            label="Volume-weighted midpoint",
        ),
        Line2D(
            [],
            [],
            color="0.2",
            linestyle="--",
            label="Volume-weighted held mean",
        ),
    ]
    figure.legend(handles=handles, loc="lower center", ncols=4, frameon=False)
    figure.suptitle("Path-box representatives across systems")
    figure.tight_layout(rect=(0.0, 0.07, 1.0, 0.96))
    return figure


def save_plot(study: PathRepresentativeStudy, path: Path) -> None:
    figure = create_combined_figure(study)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _physical_trajectories(
    evaluation: PathRepresentativeEvaluation,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prepared = evaluation.prepared
    spec = prepared.spec
    transform = prepared.transform
    shape = (-1, spec.sample_count, spec.state_count)
    fitting = inverse_transform_complete_targets(
        prepared.fitting_targets,
        transform,
    ).reshape(shape)
    prototypes = inverse_transform_complete_targets(
        np.stack([leaf.prototype for leaf in evaluation.leaves]),
        transform,
    ).reshape(shape)
    midpoints = inverse_transform_complete_targets(
        np.stack([leaf.midpoint_target for leaf in evaluation.leaves]),
        transform,
    ).reshape(shape)
    return fitting, prototypes, midpoints


def create_prototype_figure(
    evaluation: PathRepresentativeEvaluation,
) -> Figure:
    leaves = evaluation.leaves
    spec = evaluation.prepared.spec
    fitting, prototypes, midpoints = _physical_trajectories(evaluation)
    assignments = evaluation.tree.apply(evaluation.prepared.fitting_features)
    figure, axes = plt.subplots(
        len(leaves),
        spec.state_count,
        figsize=(5.2 * spec.state_count, 1.55 * len(leaves)),
        sharex=True,
        squeeze=False,
    )
    times = spec.sample_times()
    for row, leaf in enumerate(leaves):
        members = fitting[assignments == leaf.region.leaf_id]
        for column, state_name in enumerate(spec.state_names):
            axis = axes[row, column]
            axis.plot(
                times,
                members[:, :, column].T,
                color="0.45",
                alpha=0.16,
                lw=0.6,
            )
            axis.plot(
                times,
                prototypes[row, :, column],
                color=PROTOTYPE_COLOR,
                lw=2.2,
            )
            axis.plot(
                times,
                midpoints[row, :, column],
                color=MIDPOINT_COLOR,
                linestyle="--",
                lw=1.8,
            )
            axis.grid(alpha=0.15)
            if row == 0:
                axis.set_title(state_name)
            if row == len(leaves) - 1:
                axis.set_xlabel("Time")
        axes[row, 0].set_ylabel(
            f"leaf {leaf.region.leaf_id}\nn={leaf.fitting_occupancy}, "
            f"vol={leaf.relative_volume:.3f}",
            fontsize="small",
        )

    for column in range(spec.state_count):
        values = np.concatenate(
            (
                fitting[:, :, column].reshape(-1),
                prototypes[:, :, column].reshape(-1),
                midpoints[:, :, column].reshape(-1),
            ),
        )
        lower = float(values.min())
        upper = float(values.max())
        padding = 0.05 * (upper - lower) if upper > lower else 0.5
        for row in range(len(leaves)):
            axes[row, column].set_ylim(lower - padding, upper + padding)

    handles = [
        Line2D(
            [],
            [],
            color="0.45",
            alpha=0.5,
            lw=0.8,
            label="Fitting trajectories",
        ),
        Line2D([], [], color=PROTOTYPE_COLOR, lw=2.2, label="Leaf prototype"),
        Line2D(
            [],
            [],
            color=MIDPOINT_COLOR,
            linestyle="--",
            lw=1.8,
            label="Path-midpoint trajectory",
        ),
    ]
    figure.legend(handles=handles, loc="lower center", ncols=3, frameon=False)
    figure.suptitle(
        f"{spec.name}: leaf prototypes in physical coordinates",
    )
    figure.tight_layout(rect=(0.0, 0.07, 1.0, 0.98))
    return figure


def save_prototype_plot(
    evaluation: PathRepresentativeEvaluation,
    path: Path,
) -> None:
    figure = create_prototype_figure(evaluation)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Describe all four systems with path-box representatives.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="directory for the JSON report and five PNG figures",
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> None:
    args = parse_args(arguments)
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / REPORT_NAME
    plot_path = output_dir / PLOT_NAME

    report, study = run_path_representatives()
    report_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    save_plot(study, plot_path)
    prototype_paths: list[Path] = []
    for evaluation in study.evaluations:
        name = evaluation.prepared.spec.name
        try:
            filename = PROTOTYPE_NAMES[name]
        except KeyError as error:
            msg = f"no prototype output filename configured for {name}"
            raise ValueError(msg) from error
        prototype_path = output_dir / filename
        save_prototype_plot(evaluation, prototype_path)
        prototype_paths.append(prototype_path)
        print(
            f"{name}: volume-weighted midpoint "
            f"{evaluation.volume_weighted_midpoint_error:.4f}; held mean "
            f"{evaluation.volume_weighted_held_mean:.4f}",
        )

    print(f"Report: {report_path}")
    print(f"Combined plot: {plot_path}")
    for path in prototype_paths:
        print(f"Prototype plot: {path}")


if __name__ == "__main__":
    main()
