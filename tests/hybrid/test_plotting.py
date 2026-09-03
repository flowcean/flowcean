"""Tests for hybrid trace plotting."""

import matplotlib.pyplot as plt
import numpy as np

from flowcean.hybrid import Event, Trace, plot_trace


def test_plot_trace_breaks_line_at_jump() -> None:
    """A right-continuous trace does not connect reset boundary states."""
    event = Event(
        time=0.5,
        source_location="flight",
        target_location="flight",
        event_surface="ground",
        reset="bounce",
        state_before=np.array([0.0, -4.0]),
        state_after=np.array([0.0, 3.0]),
        microstep=0,
    )
    trace = Trace(
        t=np.array([0.0, 0.5, 1.0]),
        x=np.array([[1.0, 0.0], [0.0, 3.0], [0.5, -2.0]]),
        location=np.array(["flight", "flight", "flight"], dtype=object),
        events=(event,),
    )

    figure, ax = plt.subplots()
    try:
        plot_trace(
            trace,
            show_locations=False,
            show_events=False,
            ax=ax,
        )

        velocity_line = ax.lines[1]
        np.testing.assert_allclose(
            np.asarray(velocity_line.get_xdata(), dtype=float),
            [0.0, 0.5, 0.5, 0.5, 0.5, 1.0],
        )
        np.testing.assert_allclose(
            np.asarray(velocity_line.get_ydata(), dtype=float),
            [0.0, -4.0, np.nan, 3.0, 3.0, -2.0],
        )
    finally:
        plt.close(figure)
