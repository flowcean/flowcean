"""Render the reusable hybrid system benchmark gallery.

This runner only renders the gallery defined in ``benchmarks``. It creates the
``outputs`` directory when needed and writes the figure to
``outputs/benchmarks.png``.
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt

try:
    from .benchmarks import BenchmarkSpec, all_specs
except ImportError:
    from benchmarks import BenchmarkSpec, all_specs

from flowcean.ode import Trace, plot_trace, simulate


@dataclass(frozen=True)
class BenchmarkRunSummary:
    name: str
    tags: tuple[str, ...]
    location_count: int
    state_dimension: int
    step_count: int
    event_count: int
    description: str


def summarize_benchmark(
    spec: BenchmarkSpec,
    trace: Trace,
) -> BenchmarkRunSummary:
    return BenchmarkRunSummary(
        name=spec.name,
        tags=spec.tags,
        location_count=len(set(trace.location.tolist())),
        state_dimension=trace.x.shape[1] if trace.x.ndim > 1 else 1,
        step_count=trace.t.size,
        event_count=len(trace.events),
        description=spec.description,
    )


def format_benchmark_summary(summary: BenchmarkRunSummary) -> str:
    tags = ", ".join(summary.tags)
    return (
        f"{summary.name}: tags={tags}; locations={summary.location_count}; "
        f"state_dim={summary.state_dimension}; steps={summary.step_count}; "
        f"events={summary.event_count}; {summary.description}"
    )


def main() -> None:
    specs = list(all_specs())
    output_path = Path("outputs") / "benchmarks.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print("Hybrid systems benchmark gallery")

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
        trace = simulate(
            spec.factory(),
            t_span=spec.t_span,
            input_stream=spec.input_stream,
        )
        plot_trace(
            trace,
            show_locations=True,
            show_location_labels=False,
            show_events=True,
            show_event_labels=False,
            ax=ax,
        )
        ax.set_title(spec.name)
        summary = summarize_benchmark(spec, trace)
        print(format_benchmark_summary(summary))

    # Turn off any unused axes
    for ax in axes_list[len(specs) :]:
        ax.axis("off")

    fig.savefig(output_path, dpi=150)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
