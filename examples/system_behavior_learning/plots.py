from __future__ import annotations

import re
from typing import TYPE_CHECKING

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from settings import INFERENCE_PRIMARY_COMPARATORS

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from experiment import (
        PrototypePlotData,
        SuiteMetricRow,
        SuiteStatusRow,
        TreeMetricRow,
    )
    from inference import CoverageInference


METHOD_COLORS = {
    "behavior_midpoint": "tab:blue",
    "input_only_midpoint": "tab:orange",
    "archive_pam": "tab:green",
    "sobol": "tab:red",
    "random": "tab:purple",
    "lhs": "tab:brown",
}


def write_plots(
    tree_rows: Sequence[TreeMetricRow],
    suite_rows: Sequence[SuiteMetricRow],
    prototype_plots: Sequence[PrototypePlotData],
    output_dir: Path,
    *,
    planned_statuses: Sequence[SuiteStatusRow] = (),
    pending_groups: frozenset[tuple[str, int, str]] = frozenset(),
) -> tuple[Path, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction = output_dir / "prediction_quality.png"
    realization = output_dir / "representative_realization.png"
    coverage = output_dir / "reference_coverage.png"
    _tree_plot(
        tree_rows,
        "prediction_ratio",
        "Assessment RMSE / historical-mean RMSE",
        prediction,
        planned_statuses,
    )
    _tree_plot(
        tree_rows,
        "realization_ratio",
        "Midpoint realization / held leaf distance",
        realization,
        planned_statuses,
    )
    _coverage_plot(suite_rows, coverage, planned_statuses, pending_groups)
    slugs = tuple(_slug(data.system) for data in prototype_plots)
    if any(not slug for slug in slugs) or len(set(slugs)) != len(slugs):
        message = "prototype plot system names need unique nonempty slugs"
        raise ValueError(message)
    prototype_paths = tuple(
        output_dir / f"trajectory_prototypes_{slug}.png" for slug in slugs
    )
    for path in output_dir.glob("trajectory_prototypes_*.png"):
        if path not in prototype_paths:
            path.unlink()
    for data, path in zip(prototype_plots, prototype_paths, strict=True):
        _prototype_plot(data, path)
    return prediction, realization, coverage, *prototype_paths


def _tree_plot(
    rows: Sequence[TreeMetricRow],
    field: str,
    ylabel: str,
    path: Path,
    planned_statuses: Sequence[SuiteStatusRow] = (),
) -> None:
    systems = list(
        dict.fromkeys(row.system for row in (*planned_statuses, *rows)),
    )
    if not systems:
        _empty_plot(path)
        return
    figure, axes = plt.subplots(
        1,
        len(systems),
        figsize=(4.2 * len(systems), 3.6),
        squeeze=False,
        sharey=False,
    )
    for axis, system in zip(axes[0], systems, strict=True):
        capacities = sorted(
            {
                row.capacity
                for row in (*planned_statuses, *rows)
                if row.system == system
            },
        )
        centers: list[float] = []
        lower: list[float] = []
        upper: list[float] = []
        usable_capacities: list[int] = []
        for capacity in capacities:
            values = np.asarray(
                [
                    float(getattr(row, field))
                    for row in rows
                    if row.system == system
                    and row.capacity == capacity
                    and getattr(row, field) is not None
                ],
            )
            if not len(values):
                continue
            usable_capacities.append(capacity)
            centers.append(float(np.median(values)))
            lower.append(float(np.quantile(values, 0.25)))
            upper.append(float(np.quantile(values, 0.75)))
        axis.plot(usable_capacities, centers, marker="o")
        axis.fill_between(usable_capacities, lower, upper, alpha=0.2)
        if not usable_capacities:
            axis.set_xlim(min(capacities) / 1.2, max(capacities) * 1.2)
            axis.text(
                0.5,
                0.5,
                "no valid results",
                ha="center",
                transform=axis.transAxes,
            )
        axis.axhline(1.0, color="0.5", linestyle="--", linewidth=1)
        axis.set_title(system)
        axis.set_xlabel("Requested capacity")
        axis.set_xscale("log", base=2)
        axis.set_xticks(
            capacities,
            labels=[str(value) for value in capacities],
        )
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _coverage_plot(
    rows: Sequence[SuiteMetricRow],
    path: Path,
    planned_statuses: Sequence[SuiteStatusRow] = (),
    pending_groups: frozenset[tuple[str, int, str]] = frozenset(),
) -> None:
    systems = list(
        dict.fromkeys(row.system for row in (*planned_statuses, *rows)),
    )
    systems.extend(sorted({key[0] for key in pending_groups} - set(systems)))
    if not systems:
        _empty_plot(path)
        return
    figure, axes = plt.subplots(
        1,
        len(systems),
        figsize=(4.8 * len(systems), 3.8),
        squeeze=False,
        sharey=False,
    )
    handles: dict[str, Line2D] = {}
    for axis, system in zip(axes[0], systems, strict=True):
        capacities = sorted(
            {
                row.capacity
                for row in (*planned_statuses, *rows)
                if row.system == system
            }
            | {key[1] for key in pending_groups if key[0] == system},
        )
        for method, color in METHOD_COLORS.items():
            medians = []
            for capacity in capacities:
                replicate_values = (
                    []
                    if (system, capacity, method) in pending_groups
                    else _coverage_replicates(rows, system, capacity, method)
                )
                medians.append(
                    float(np.median(replicate_values))
                    if replicate_values
                    else np.nan,
                )
            if np.any(np.isfinite(medians)):
                (line,) = axis.plot(
                    capacities,
                    medians,
                    marker="o",
                    label=method,
                    color=color,
                )
                handles.setdefault(method, line)
        pending = any(key[0] == system for key in pending_groups)
        if not axis.lines:
            axis.set_xlim(min(capacities) / 1.2, max(capacities) * 1.2)
        if pending or not axis.lines:
            annotation = (
                (
                    "some summaries withheld: incomplete planned evidence"
                    if axis.lines
                    else "incomplete planned evidence"
                )
                if pending
                else "no valid results"
            )
            axis.text(
                0.5,
                0.95 if axis.lines else 0.5,
                annotation,
                ha="center",
                va="top",
                transform=axis.transAxes,
            )
        axis.set_title(system)
        axis.set_xlabel("Requested capacity")
        axis.set_ylabel("Mean nearest-reference distance")
        axis.set_xscale("log", base=2)
        axis.set_xticks(
            capacities,
            labels=[str(value) for value in capacities],
        )
        axis.grid(alpha=0.2)
    if handles:
        figure.legend(
            handles=[
                handles[method]
                for method in METHOD_COLORS
                if method in handles
            ],
            fontsize="small",
            bbox_to_anchor=(1.0, 1.0),
            loc="upper left",
        )
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _coverage_replicates(
    rows: Sequence[SuiteMetricRow],
    system: str,
    capacity: int,
    method: str,
) -> list[float]:
    by_replicate: dict[int, list[float]] = {}
    for row in rows:
        if (
            row.system == system
            and row.capacity == capacity
            and row.method == method
        ):
            by_replicate.setdefault(row.replicate, []).append(
                row.mean_distance,
            )
    return [float(np.median(values)) for values in by_replicate.values()]


def _empty_plot(path: Path) -> None:
    figure, axis = plt.subplots(figsize=(4.2, 3.6))
    axis.text(
        0.5,
        0.5,
        "no valid results",
        ha="center",
        transform=axis.transAxes,
    )
    axis.set_axis_off()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _prototype_plot(data: PrototypePlotData, path: Path) -> None:
    state_count = len(data.state_names)
    sample_count = len(data.sample_times)
    fitting = data.fitting_trajectories.reshape(
        -1,
        sample_count,
        state_count,
    )
    prototypes = data.leaf_prototypes.reshape(
        len(data.leaf_ids),
        sample_count,
        state_count,
    )
    midpoints = data.midpoint_trajectories.reshape(
        len(data.leaf_ids),
        sample_count,
        state_count,
    )
    figure, axes = plt.subplots(
        len(data.leaf_ids),
        state_count,
        figsize=(5.2 * state_count, 1.55 * len(data.leaf_ids)),
        sharex=True,
        squeeze=False,
    )
    for row, (leaf_id, volume) in enumerate(
        zip(data.leaf_ids, data.relative_volumes, strict=True),
    ):
        members = fitting[data.fitting_assignments == leaf_id]
        for column, state_name in enumerate(data.state_names):
            axis = axes[row, column]
            axis.plot(
                data.sample_times,
                members[:, :, column].T,
                color="0.45",
                alpha=0.16,
                linewidth=0.6,
            )
            axis.plot(
                data.sample_times,
                prototypes[row, :, column],
                color="tab:blue",
                linewidth=2.2,
            )
            axis.plot(
                data.sample_times,
                midpoints[row, :, column],
                color="tab:orange",
                linestyle="--",
                linewidth=1.8,
            )
            axis.grid(alpha=0.15)
            if row == 0:
                axis.set_title(state_name)
        axes[row, 0].set_ylabel(
            f"leaf {leaf_id}\nn={len(members)}, vol={volume:.3f}",
            fontsize="small",
        )

    for column in range(state_count):
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
        for row in range(len(data.leaf_ids)):
            axes[row, column].set_ylim(lower - padding, upper + padding)

    handles = (
        Line2D(
            [],
            [],
            color="0.45",
            alpha=0.5,
            linewidth=0.8,
            label="Fitting trajectories",
        ),
        Line2D(
            [],
            [],
            color="tab:blue",
            linewidth=2.2,
            label="Leaf prototype",
        ),
        Line2D(
            [],
            [],
            color="tab:orange",
            linestyle="--",
            linewidth=1.8,
            label="Path-midpoint trajectory",
        ),
    )
    figure.supxlabel("Time", y=0)
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0),
        ncols=3,
        frameon=False,
    )
    figure.suptitle(
        f"{data.system}: behavior-tree leaf prototypes "
        f"(replicate {data.replicate}, capacity {data.capacity})",
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_coverage_ratios(
    results: Sequence[CoverageInference],
    output_dir: Path,
) -> Path:
    """Plot paired points and whole-curve bands, never survivor bands."""
    path = output_dir / "coverage_ratios.png"
    if not results:
        _empty_plot(path)
        return path
    figure, axes = plt.subplots(
        1,
        len(results),
        figsize=(5.4 * len(results), 4.8),
        squeeze=False,
    )
    try:
        for axis, result in zip(axes[0], results, strict=True):
            capacities = result.capacities
            annotations: list[str] = []
            for method in INFERENCE_PRIMARY_COMPARATORS:
                comparison = result.comparisons[method]
                points = comparison.points
                ratios = [
                    point.ratio
                    if point.ratio is not None
                    and np.isfinite(point.ratio)
                    and point.ratio > 0
                    else np.nan
                    for point in points
                ]
                color = METHOD_COLORS[method]
                axis.plot(
                    capacities,
                    ratios,
                    marker="o",
                    label=method,
                    color=color,
                )
                band = comparison.band
                if (
                    band.status == "available"
                    and band.ratio_lower is not None
                    and band.ratio_upper is not None
                ):
                    lower, upper = band.ratio_lower, band.ratio_upper
                    usable = (
                        np.isfinite(lower)
                        & np.isfinite(upper)
                        & (lower > 0)
                        & (upper > 0)
                    )
                    axis.fill_between(
                        capacities,
                        lower,
                        upper,
                        where=usable,
                        color=color,
                        alpha=0.18,
                    )
                    if not np.all(usable):
                        annotations.append(
                            f"{method}: band ratio out of range",
                        )
                else:
                    annotations.append(
                        f"{method}: band unavailable ({band.status})",
                    )
                if any(
                    point.log_effect is not None and point.ratio_reason
                    for point in points
                ):
                    annotations.append(f"{method}: point ratio out of range")
                if any(point.reason for point in points):
                    annotations.append(f"{method}: incomplete points (gaps)")
            axis.axhline(1, color="0.5", linestyle="--", linewidth=1)
            axis.set_xscale("log", base=2)
            axis.set_xticks(capacities, labels=[str(c) for c in capacities])
            axis.set_xlim(min(capacities) / 1.2, max(capacities) * 1.2)
            axis.set_yscale("log")
            axis.set_xlabel("Requested capacity")
            axis.set_ylabel(
                "Behavior / comparator coverage error (lower better)",
            )
            axis.set_title(
                f"{result.system}\n"
                f"Approx. {100 * result.settings.confidence_level:g}% "
                "whole-curve bands",
            )
            axis.grid(alpha=0.2)
            axis.legend(fontsize="x-small")
            if annotations:
                axis.text(
                    0,
                    -0.25,
                    "\n".join(annotations),
                    transform=axis.transAxes,
                    fontsize="x-small",
                    va="top",
                )
        figure.suptitle(
            "Per-curve across planned capacities\n"
            "Not jointly simultaneous across systems or comparators",
            fontsize="small",
        )
        figure.tight_layout()
        figure.savefig(path, dpi=160, bbox_inches="tight")
    finally:
        plt.close(figure)
    return path
