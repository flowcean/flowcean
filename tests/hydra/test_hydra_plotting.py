import matplotlib.pyplot as plt
import polars as pl
import pytest

from flowcean.hydra.plotting import plot_hydra_replay_step
from flowcean.hydra.replay import HyDRAReplay, HyDRAStep


def test_plot_hydra_replay_step_draws_signal_lines_and_segment_overlays() -> (
    None
):
    replay = HyDRAReplay(
        trace_count=1,
        threshold=0.5,
        start_width=2,
        step_width=2,
        final_mode_count=1,
        steps=(
            HyDRAStep("pending_segment_found", 0, 0, 3),
            HyDRAStep(
                "candidate_window_evaluated",
                0,
                0,
                1,
                window_size=2,
                fit=0.0,
                threshold=0.5,
            ),
            HyDRAStep("accurate_segment_found", 0, 0, 3, mode_id=0),
        ),
    )
    trace = pl.DataFrame(
        {
            "t": [0.0, 1.0, 2.0, 3.0],
            "x": [0.0, 0.5, 1.0, 1.5],
            "y": [1.5, 1.0, 0.5, 0.0],
        },
    )
    figure, ax = plt.subplots()

    try:
        result = plot_hydra_replay_step(
            [trace],
            replay,
            step_index=2,
            x_column="t",
            y_columns=("x", "y"),
            ax=ax,
        )

        assert result is ax
        assert len(ax.lines) == 2
        assert len(ax.patches) >= 2
        assert "accurate_segment_found" in ax.get_title()
    finally:
        plt.close(figure)


def test_plot_hydra_replay_step_rejects_missing_plot_columns() -> None:
    replay = HyDRAReplay(
        trace_count=1,
        threshold=0.5,
        start_width=2,
        step_width=2,
        final_mode_count=0,
        steps=(HyDRAStep("pending_segment_found", 0, 0, 1),),
    )

    with pytest.raises(ValueError, match="missing plot column"):
        plot_hydra_replay_step(
            [pl.DataFrame({"x": [0.0, 1.0]})],
            replay,
            step_index=0,
            x_column="t",
            y_columns=("x",),
        )


def test_plot_hydra_replay_step_scopes_segments_to_current_trace() -> None:
    replay = HyDRAReplay(
        trace_count=2,
        threshold=0.5,
        start_width=2,
        step_width=2,
        final_mode_count=1,
        steps=(
            HyDRAStep("accurate_segment_found", 0, 0, 3, mode_id=0),
            HyDRAStep("pending_segment_found", 1, 0, 1),
        ),
    )
    traces = [
        pl.DataFrame({"t": [0.0, 1.0, 2.0, 3.0], "x": [0.0, 0.1, 0.2, 0.3]}),
        pl.DataFrame({"t": [10.0, 11.0], "x": [1.0, 1.1]}),
    ]
    figure, ax = plt.subplots()

    try:
        result = plot_hydra_replay_step(
            traces,
            replay,
            step_index=1,
            x_column="t",
            y_columns=("x",),
            ax=ax,
        )

        assert result is ax
        assert len(ax.lines) == 1
        assert len(ax.patches) == 1
        assert "trace=1" in ax.get_title()
    finally:
        plt.close(figure)


def test_plot_hydra_replay_step_places_legend_above_axes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay = HyDRAReplay(
        trace_count=1,
        threshold=0.5,
        start_width=2,
        step_width=2,
        final_mode_count=1,
        steps=(HyDRAStep("pending_segment_found", 0, 0, 1),),
    )
    trace = pl.DataFrame({"t": [0.0, 1.0], "x": [0.0, 1.0]})
    legend_calls: list[dict[str, object]] = []
    subplots_adjust_calls: list[dict[str, float]] = []

    figure, ax = plt.subplots()
    original_legend = ax.legend

    def capture_legend(*args: object, **kwargs: object) -> object:
        del args
        legend_calls.append(dict(kwargs))
        return original_legend(**kwargs)

    def capture_adjust(**kwargs: float) -> None:
        subplots_adjust_calls.append(kwargs)

    monkeypatch.setattr(ax, "legend", capture_legend)
    monkeypatch.setattr(figure, "subplots_adjust", capture_adjust)

    try:
        plot_hydra_replay_step(
            [trace],
            replay,
            step_index=0,
            x_column="t",
            y_columns=("x",),
            ax=ax,
        )
    finally:
        plt.close(figure)

    assert legend_calls == [
        {
            "loc": "lower center",
            "bbox_to_anchor": (0.5, 1.02),
            "ncol": 2,
            "borderaxespad": 0.0,
        },
    ]
    assert subplots_adjust_calls == [{"top": 0.82}]
