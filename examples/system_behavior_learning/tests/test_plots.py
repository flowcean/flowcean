from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from experiment import PrototypePlotData, SuiteMetricRow, SuiteStatusRow
from matplotlib.figure import Figure
from plots import _prototype_plot, write_plots

if TYPE_CHECKING:
    from pathlib import Path


COLORS = {
    "behavior_midpoint": "tab:blue",
    "input_only_midpoint": "tab:orange",
    "archive_pam": "tab:green",
    "sobol": "tab:red",
    "random": "tab:purple",
    "lhs": "tab:brown",
}


def _metric(system: str, method: str, capacity: int = 2) -> SuiteMetricRow:
    return SuiteMetricRow(
        system,
        0,
        capacity,
        capacity,
        method,
        None,
        1.0,
        1.0,
        1.0,
        1.0,
        capacity,
        1.0,
        1.0,
    )


def _status(system: str, method: str, capacity: int = 2) -> SuiteStatusRow:
    return SuiteStatusRow(
        system,
        0,
        capacity,
        method,
        None,
        capacity,
        0,
        "invalid_execution",
    )


@pytest.fixture
def figures(monkeypatch: pytest.MonkeyPatch) -> dict[str, Figure]:
    saved: dict[str, Figure] = {}

    def capture(self: Figure, path: Path, **_kwargs: object) -> None:
        saved[path.name] = self

    monkeypatch.setattr(Figure, "savefig", capture)
    return saved


@pytest.mark.parametrize("state_count", [1, 2, 3])
def test_prototype_shared_time_label_leaves_space_for_legend(
    tmp_path: Path,
    figures: dict[str, Figure],
    state_count: int,
) -> None:
    times = np.linspace(0, 1, 5)
    values = np.arange(2 * len(times) * state_count, dtype=float).reshape(
        2,
        -1,
    )
    data = PrototypePlotData(
        system="Synthetic",
        replicate=0,
        capacity=2,
        state_names=tuple(f"state_{i}" for i in range(state_count)),
        sample_times=times,
        leaf_ids=np.array([1, 2]),
        relative_volumes=np.array([0.5, 0.5]),
        fitting_assignments=np.array([1, 2]),
        fitting_trajectories=values,
        leaf_prototypes=values,
        midpoint_trajectories=values + 0.1,
    )
    _prototype_plot(data, tmp_path / "prototypes.png")
    figure = figures["prototypes.png"]
    (label,) = [text for text in figure.texts if text.get_text() == "Time"]
    assert all(not axis.get_xlabel() for axis in figure.axes)
    figure.canvas.draw()
    assert (
        figure.legends[0].get_window_extent().y1 < label.get_window_extent().y0
    )
    for index, axis in enumerate(figure.axes):
        row, column = divmod(index, state_count)
        for line, expected in zip(
            axis.lines[-2:],
            (
                values[row, column::state_count],
                values[row, column::state_count] + 0.1,
            ),
            strict=True,
        ):
            np.testing.assert_array_equal(line.get_xdata(), times)
            np.testing.assert_array_equal(line.get_ydata(), expected)


def test_coverage_colors_and_shared_legend_survive_final_empty_system(
    tmp_path: Path,
    figures: dict[str, Figure],
) -> None:
    # Different missing methods on each axis must not shift method colors.
    rows = (
        _metric("First", "behavior_midpoint"),
        _metric("First", "archive_pam"),
        _metric("First", "random"),
        _metric("Second", "input_only_midpoint"),
        _metric("Second", "sobol"),
        _metric("Second", "random"),
        _metric("Second", "lhs"),
    )
    statuses = tuple(
        _status(system, method)
        for system in ("First", "Second", "Empty")
        for method in COLORS
    )
    write_plots((), rows, (), tmp_path, planned_statuses=statuses)
    figure = figures["reference_coverage.png"]
    assert [axis.get_title() for axis in figure.axes] == [
        "First",
        "Second",
        "Empty",
    ]
    for axis in figure.axes:
        assert axis.get_legend() is None
        for line in axis.lines:
            label = line.get_label()
            assert isinstance(label, str)
            assert line.get_color() == COLORS[label]
    assert not figure.axes[-1].lines
    assert [text.get_text() for text in figure.axes[-1].texts] == [
        "no valid results",
    ]
    assert len(figure.legends) == 1
    legend = figure.legends[0]
    assert [text.get_text() for text in legend.get_texts()] == list(COLORS)
    assert [line.get_color() for line in legend.get_lines()] == list(
        COLORS.values(),
    )
    figure.canvas.draw()
    legend_box = legend.get_window_extent()
    assert all(
        legend_box.x0 > axis.get_window_extent().x1 for axis in figure.axes
    )


@pytest.mark.parametrize("other_curve", [False, True])
def test_pending_coverage_annotation_is_not_no_valid_results(
    tmp_path: Path,
    figures: dict[str, Figure],
    *,
    other_curve: bool,
) -> None:
    rows = (_metric("Pending", "sobol"),)
    if other_curve:
        rows += (_metric("Pending", "random"),)
    write_plots(
        (),
        rows,
        (),
        tmp_path,
        planned_statuses=(_status("Pending", "sobol"),),
        pending_groups=frozenset({("Pending", 2, "sobol")}),
    )
    axis = figures["reference_coverage.png"].axes[0]
    assert [line.get_label() for line in axis.lines] == (
        ["random"] if other_curve else []
    )
    assert [text.get_text() for text in axis.texts] == [
        "some summaries withheld: incomplete planned evidence"
        if other_curve
        else "incomplete planned evidence",
    ]
    assert rows[0].mean_distance == 1.0


def test_missing_capacity_breaks_coverage_curve_without_changing_metrics(
    tmp_path: Path,
    figures: dict[str, Figure],
) -> None:
    rows = tuple(_metric("Gaps", "sobol", capacity) for capacity in (2, 8))
    write_plots(
        (),
        rows,
        (),
        tmp_path,
        planned_statuses=(_status("Gaps", "sobol", 4),),
    )
    line = figures["reference_coverage.png"].axes[0].lines[0]
    np.testing.assert_array_equal(line.get_xdata(), [2, 4, 8])
    np.testing.assert_array_equal(line.get_ydata(), [1.0, np.nan, 1.0])
    assert [row.mean_distance for row in rows] == [1.0, 1.0]
