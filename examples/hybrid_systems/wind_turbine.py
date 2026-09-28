"""Run the wind turbine through a smooth 7..15..7 m/s wind cycle."""

from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")

import matplotlib.pyplot as plt
from scenarios import WIND_TURBINE

from flowcean.hybrid import plot_locations, simulate
from flowcean.hybrid.benchmarks import wind_turbine_power
from flowcean.hybrid.hybrid_system import display_label


def main() -> None:
    # The default rotor is already turning; this is not startup from rest.
    system = WIND_TURBINE.factory()
    trace = simulate(
        system,
        t_span=WIND_TURBINE.t_span,
        input_stream=WIND_TURBINE.input_stream,
    )

    frame = trace.sample(
        dt=0.1, include_inputs=True, include_location_label=True
    )
    print(f"Initial mode: {frame['location_label'][0].replace('_', ' ')}")
    for event in trace.events:
        source = display_label(event.source_location).replace("_", " ")
        target = display_label(event.target_location).replace("_", " ")
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
            wind_turbine_power(frame, parameters=system.parameters) / 1e6,
        ),
    )
    fig, axes = plt.subplots(
        len(signals), 1, sharex=True, figsize=(9, 11), layout="constrained"
    )
    for ax, (label, values) in zip(axes, signals, strict=True):
        ax.plot(frame["t"], values, color="#263238")
        plot_locations(trace, ax=ax, alpha=0.16)
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
    rated_power_mw = system.parameters["rated_mechanical_power"] / 1e6
    rated_line = axes[-1].axhline(
        rated_power_mw,
        color="black",
        linestyle="--",
        linewidth=1,
        label=f"Rated power: {rated_power_mw:.2f} MW",
    )
    axes[-1].legend(handles=[rated_line], loc="lower center")
    axes[-1].set_xlabel("Time (s)")
    axes[-1].set_xlim(trace.t_span)
    fig.suptitle("Running wind turbine")

    # Every panel uses the same mode colors, so one legend is enough.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        [label.replace("_", " ").capitalize() for label in labels],
        loc="outside lower center",
        ncols=2,
        title="Controller mode",
        frameon=False,
    )

    output_path = Path(__file__).parent / "outputs" / "wind_turbine.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
