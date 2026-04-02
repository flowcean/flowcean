"""Run hybrid system benchmarks and visualize traces."""

import matplotlib as mpl
import matplotlib.pyplot as plt
from benchmarks import all_specs, turbine_specs

from flowcean.ode import plot_trace, simulate

mpl.use("Agg")


def main() -> None:
    specs = list(turbine_specs())

    # specs = list(all_specs())

    cols = 3
    rows = (len(specs) + cols - 1) // cols
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(cols * 5.0, rows * 3.0),
        layout="constrained",
    )
    axes_list = axes.flatten()

    for ax, spec in zip(axes_list, specs, strict=False):
        trace = simulate(spec.factory(), t_span=spec.t_span)
        plot_trace(
            trace,
            show_modes=True,
            show_mode_labels=False,
            show_events=True,
            show_event_labels=False,
            ax=ax,
        )
        ax.set_title(spec.name)
        print(f"{spec.name}: {trace.t.size} steps")

    # Turn off any unused axes
    for ax in axes_list[len(specs) :]:
        ax.axis("off")

    fig.savefig("benchmarks.png", dpi=150)


if __name__ == "__main__":
    main()
