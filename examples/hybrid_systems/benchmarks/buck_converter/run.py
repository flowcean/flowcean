"""Run and visualize the SpaceEx buck-converter conversion in Flowcean."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from flowcean.ode import plot_phase, plot_trace, simulate, trace_to_polars

try:
    from ..buck_converter import buck_converter
except ImportError:
    # Allows running as a script from this folder as well.
    import importlib.util

    module_path = Path(__file__).resolve().parents[1] / "buck_converter.py"
    spec = importlib.util.spec_from_file_location(
        "flowcean_buck_converter_module",
        module_path,
    )
    if spec is None or spec.loader is None:
        message = f"Could not load module from {module_path}."
        raise RuntimeError(message)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    buck_converter = module.buck_converter


def main() -> None:
    """Simulate the converted automaton and write verification plots."""
    t_span = (0.0, 10.0)
    sample_dt = 0.01

    system = buck_converter(initial_state=np.array([0.0, 0.0], dtype=float))
    trace = simulate(system, t_span=t_span, sample_dt=sample_dt)

    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig_trace, ax_trace = plt.subplots(figsize=(9.0, 4.5), layout="constrained")
    plot_trace(
        trace,
        show_locations=True,
        show_location_labels=True,
        show_events=True,
        show_event_labels=True,
        ax=ax_trace,
    )
    ax_trace.set_title("Buck Converter (SpaceEx -> Flowcean): State Trace")
    trace_path = output_dir / "buck_converter_trace.png"
    fig_trace.savefig(trace_path, dpi=150)
    plt.close(fig_trace)

    fig_phase, ax_phase = plt.subplots(figsize=(6.0, 5.5), layout="constrained")
    plot_phase(trace, x_dim=0, y_dim=1, ax=ax_phase)
    ax_phase.set_title("Buck Converter (SpaceEx -> Flowcean): Phase Portrait")
    phase_path = output_dir / "buck_converter_phase.png"
    fig_phase.savefig(phase_path, dpi=150)
    plt.close(fig_phase)

    csv_path = output_dir / "buck_converter_trace.csv"
    trace_to_polars(trace).write_csv(csv_path)

    unique_locations = len(set(trace.location.tolist()))
    print("Buck converter conversion verification")
    print(f"t_span={t_span}, sample_dt={sample_dt}")
    print(f"samples={trace.t.size}, state_dim={trace.x.shape[1]}")
    print(f"events={len(trace.events)}, visited_locations={unique_locations}")
    print(f"wrote {trace_path}")
    print(f"wrote {phase_path}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
