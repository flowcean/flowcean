"""Plot the gallery's driven Duffing oscillator and its velocity resets.

Use --output to choose a figure path, including the documentation SVG path
when regenerating the gallery asset. Ordinary runs write to ignored outputs.
"""

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scenarios import DUFFING_OSCILLATOR

from flowcean.hybrid import (
    plot_locations,
    plot_style,
    plot_trajectory,
    simulate,
)


@plot_style()
def main() -> None:
    mpl.use("Agg")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent
        / "outputs"
        / "duffing_oscillator-trace.svg",
        help="figure path (SVG or another Matplotlib-supported format)",
    )
    args = parser.parse_args()
    system = DUFFING_OSCILLATOR.factory()
    trajectory = simulate(
        system,
        t_span=DUFFING_OSCILLATOR.t_span,
        input_stream=DUFFING_OSCILLATOR.input_stream,
    )
    frame = trajectory.sample(dt=0.02, include_inputs=True)
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(9, 9),
        gridspec_kw={"height_ratios": [1, 1.4, 1.4, 0.8]},
        layout="constrained",
    )
    axes[1].sharex(axes[0])
    axes[2].sharex(axes[0])
    axes[0].plot(frame["t"], frame["u0"], color="0.25")
    plot_locations(trajectory, ax=axes[0])
    for dimension, ax in enumerate(axes[1:3]):
        plot_trajectory(
            trajectory,
            dims=[dimension],
            ax=ax,
            show_locations=False,
            show_events=False,
        )
        ax.get_legend().remove()
        plot_locations(trajectory, ax=ax)
    threshold_lines = []
    for name, style in (("inner_radius", ":"), ("outer_radius", "--")):
        radius = system.parameters[name]
        for sign in (-1, 1):
            line = axes[1].axhline(
                sign * radius,
                color="0.4",
                linestyle=style,
                linewidth=0.9,
                label=name.replace("_", " ") if sign == 1 else "_nolegend_",
            )
            if sign == 1:
                threshold_lines.append(line)
    axes[1].legend(
        handles=threshold_lines,
        loc="lower right",
    )
    for event in trajectory.events:
        axes[2].plot(
            [event.time, event.time],
            [event.state_before[1], event.state_after[1]],
            color="0.25",
            linestyle="--",
            linewidth=1.2,
        )

    # A separate compact zoom keeps the small speed loss visible without
    # covering the main traces. The connector uses exact event states.
    zoom = axes[3]
    if trajectory.events:
        event = trajectory.events[0]
        half_width = 0.04
        plot_trajectory(
            trajectory,
            dims=[1],
            ax=zoom,
            show_locations=False,
            show_events=False,
        )
        zoom.get_legend().remove()
        plot_locations(trajectory, ax=zoom)
        zoom.plot(
            [event.time, event.time],
            [event.state_before[1], event.state_after[1]],
            color="0.25",
            linestyle="--",
            linewidth=1.2,
            marker="o",
            markersize=3,
        )
        zoom.set_xlim(event.time - half_width, event.time + half_width)
        velocities = [
            trajectory.evaluate(event.time - half_width).state[1],
            trajectory.evaluate(event.time + half_width).state[1],
            event.state_before[1],
            event.state_after[1],
        ]
        margin = 0.2 * np.ptp(velocities)
        zoom.set_ylim(min(velocities) - margin, max(velocities) + margin)
        zoom.set_title("First switch: velocity reset (dashed)", fontsize=10)
        zoom.set_xlabel("Time (normalized), expanded scale")
        zoom.set_ylabel("Velocity\n(normalized)")
    else:
        zoom.set_visible(False)

    for ax, label in zip(
        axes[:3],
        (
            "Force per unit mass\n(normalized)",
            "Position\n(normalized)",
            "Velocity\n(normalized)",
        ),
        strict=True,
    ):
        ax.set_ylabel(label)
        ax.set_xlim(trajectory.t_span)
        ax.set_xlabel("")
    axes[0].tick_params(labelbottom=False)
    axes[1].tick_params(labelbottom=False)
    axes[2].set_xlabel("Time (normalized)")
    fig.suptitle("Driven Duffing oscillator")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        [label.replace("_", " ").capitalize() for label in labels],
        loc="outside lower center",
        ncols=2,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"Date": None} if args.output.suffix.lower() == ".svg" else None
    with mpl.rc_context({"svg.hashsalt": "flowcean-duffing-oscillator"}):
        fig.savefig(args.output, metadata=metadata, dpi=150)
    plt.close(fig)
    if args.output.suffix.lower() == ".svg":
        # Match repository whitespace checks as part of actual generation.
        svg = args.output.read_text(encoding="utf-8")
        args.output.write_text(
            "\n".join(line.rstrip() for line in svg.splitlines()) + "\n",
            encoding="utf-8",
        )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
