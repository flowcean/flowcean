"""Run the wind turbine through a smooth 7..15..7 m/s wind cycle."""

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scenarios import WIND_TURBINE

from flowcean.hybrid import plot_locations, plot_style, simulate
from flowcean.hybrid.benchmarks import wind_turbine_power
from flowcean.hybrid.hybrid_system import display_label


def _caption(label: str) -> str:
    """Present a model label as words without changing its identity."""
    return label.replace("_", " ")


@plot_style()
def main() -> None:
    mpl.use("Agg")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "outputs" / "wind_turbine.png",
        help="figure path (SVG or another Matplotlib-supported format)",
    )
    args = parser.parse_args()
    # The default rotor is already turning; this is not startup from rest.
    system = WIND_TURBINE.factory()
    trajectory = simulate(
        system,
        t_span=WIND_TURBINE.t_span,
        input_stream=WIND_TURBINE.input_stream,
    )

    frame = trajectory.sample(
        dt=0.1, include_inputs=True, include_location_label=True
    )
    print(f"Initial mode: {_caption(frame['location_label'][0])}")
    for event in trajectory.events:
        source = _caption(display_label(event.transition.source))
        target = _caption(display_label(event.transition.target))
        print(f"t = {event.time:6.2f} s: {source} -> {target}")

    # State columns: rotor speed, tower displacement/velocity, pitch/rate,
    # and the pitch controller's integral contribution.
    signals = (
        ("Wind speed (m/s)", frame["u0"].to_numpy()),
        ("Rotor speed (rpm)", frame["x0"].to_numpy() * 60 / (2 * np.pi)),
        ("Blade pitch (degrees)", np.rad2deg(frame["x3"].to_numpy())),
        ("Tower displacement (m)", frame["x1"].to_numpy()),
        (
            "Generator mechanical\npower (MW)",
            wind_turbine_power(
                frame["x0"].to_numpy(),
                frame["location_label"].to_list(),
                parameters=system.parameters,
            )
            / 1e6,
        ),
    )
    fig, axes = plt.subplots(
        len(signals),
        1,
        sharex=True,
        figsize=(9, 11),
        layout="constrained",
    )
    for ax, (label, values) in zip(axes, signals, strict=True):
        ax.plot(frame["t"], values, color="0.25")
        plot_locations(trajectory, ax=ax)
        ax.set_ylabel(label)
    rated_power_mw = system.parameters["rated_mechanical_power"] / 1e6
    rated_line = axes[-1].axhline(
        rated_power_mw,
        color="0.25",
        linestyle="--",
        linewidth=1,
        label=f"Rated power: {rated_power_mw:.2f} MW",
    )
    axes[-1].legend(
        handles=[rated_line],
        loc="lower center",
    )
    axes[-1].set_xlabel("Time (s)")
    axes[-1].set_xlim(trajectory.t_span)
    fig.suptitle("Running wind turbine")

    # Every panel uses the same mode colors, so one legend is enough.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        [_caption(label).capitalize() for label in labels],
        loc="outside lower center",
        ncols=2,
        title="Controller mode",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"Date": None} if args.output.suffix.lower() == ".svg" else None
    with mpl.rc_context({"svg.hashsalt": "flowcean-wind-turbine"}):
        fig.savefig(args.output, dpi=150, metadata=metadata)
    plt.close(fig)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
