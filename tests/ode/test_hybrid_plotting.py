"""Tests for hybrid trace plotting helpers."""

import matplotlib as mpl
import numpy as np

mpl.use("Agg")

from flowcean.ode import Event, Trace, plot_phase, plot_trace


def test_plot_trace_uses_location_event_labels() -> None:
    """Event annotations use source and target locations."""
    trace = Trace(
        t=np.array([0.0, 1.0], dtype=float),
        x=np.array([[0.0], [1.0]], dtype=float),
        location=np.array(["left", "right"], dtype=object),
        events=(
            Event(
                time=0.5,
                source_location="left",
                target_location="right",
                guard="cross",
                reset=None,
                state=np.array([0.5], dtype=float),
            ),
        ),
    )

    ax = plot_trace(trace, show_location_labels=True)

    assert any("left->right" in text.get_text() for text in ax.texts)


def test_plot_phase_labels_location_segments() -> None:
    """Phase plots label trajectory segments with locations."""
    trace = Trace(
        t=np.array([0.0, 1.0, 2.0], dtype=float),
        x=np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]], dtype=float),
        location=np.array(["left", "left", "right"], dtype=object),
        events=(),
    )

    ax = plot_phase(trace)

    legend = ax.get_legend()
    assert legend is not None
    labels = [text.get_text() for text in legend.get_texts()]
    assert "left" in labels
    assert "right" in labels
