from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from pysr import PySRRegressor
from system import thermostat, thermostat_target_stream

import flowcean.cli
import flowcean.utils
from flowcean.hydra import (
    HybridDecisionTreeLearner,
    HybridDecisionTreeModel,
    HyDRALearner,
    HyDRAModel,
    HyDRATraceSchema,
    PlotCallback,
    SelectorFeatureConfig,
    StateTraceComparison,
    compare_state_traces,
)
from flowcean.ode import (
    Trace,
    plot_trace,
    simulate,
    trace_to_polars,
)
from flowcean.pysr import PySRLearner

EXAMPLE_SEED = 42
HYDRA_LOGGER = "flowcean.hydra.learner"
OUTPUT_DIR = Path("outputs")


def print_selector_outputs(
    selector: HybridDecisionTreeModel,
    output_dir: Path = Path("outputs"),
) -> None:
    print("selector_summary")
    print(selector.summary_text())
    print("selector_mode_summary")
    print(selector.mode_summary_text())
    # print("selector_leaf_summary")
    # print(selector.leaf_summary_text())
    print("selector_tree")
    print(selector.tree_text())

    svg_path = output_dir / "selector_tree.svg"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        selector.save_svg(svg_path)
    except (RuntimeError, OSError) as exc:
        print("selector_svg_skipped", exc)
        return

    print("selector_svg", svg_path)


def compare_learned_model_to_reference(
    model: HyDRAModel,
    reference: Trace,
) -> tuple[Trace, StateTraceComparison]:
    learned_trace = model.simulate(
        (float(reference.t[0]), float(reference.t[-1])),
        reference.x[0],
        sample_times=reference.t,
    )
    return learned_trace, compare_state_traces(reference, learned_trace)


def save_trace_comparison_plot(
    reference: Trace,
    learned: Trace,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    ax.plot(reference.t, reference.x[:, 0], label="reference x")
    ax.plot(learned.t, learned.x[:, 0], label="learned x", linestyle="--")
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def format_comparison_summary(comparison: StateTraceComparison) -> str:
    return "\n".join(
        (
            "learned trace comparison",
            f"mae: {comparison.mae:.6g}",
            f"rmse: {comparison.rmse:.6g}",
            f"max_error: {comparison.max_error:.6g}",
        ),
    )


def main() -> None:
    flowcean.cli.initialize()
    flowcean.utils.initialize_random(EXAMPLE_SEED)

    system = thermostat()
    reference_trace = simulate(
        system,
        t_span=(0.0, 20.0),
        input_stream=thermostat_target_stream,
        capture_derivatives=True,
        sample_dt=0.02,
    )

    print(
        "Plotting reference trace... close plot to continue...",
    )
    plot_trace(
        reference_trace,
        show_locations=True,
        show_location_labels=False,
        show_events=True,
        show_event_labels=False,
        show=True,
    )
    trace_frame = trace_to_polars(
        reference_trace,
        state_names=("x",),
        derivative_names=("dx",),
    )
    schema = HyDRATraceSchema(time="t", state=("x",), derivative=("dx",))
    callback = PlotCallback(reference_trace, dims=[0])
    learner = HyDRALearner(
        regressor_factory=lambda: PySRLearner(
            model=PySRRegressor(
                niterations=10,
                # random_state=flowcean.utils.get_seed(),
            ),
        ),
        threshold=1e-2,
        selector_learner=HybridDecisionTreeLearner(
            SelectorFeatureConfig(state_features=("x",)),
            random_state=7,
        ),
        callback=callback,
        trace_schema=schema,
    )

    model = learner.learn(
        trace_frame.select(schema.input_features).lazy(),
        trace_frame.select(schema.derivative).lazy(),
    )

    print(
        {
            "rows": trace_frame.height,
            "locations": trace_frame["location"].unique().sort().to_list(),
            "modes": len(model.modes),
            "input_features": model.input_features,
            "output_features": model.output_features,
        },
    )
    if model.selector is not None:
        print_selector_outputs(model.selector, output_dir=OUTPUT_DIR)

    learned_trace, comparison = compare_learned_model_to_reference(
        model,
        reference_trace,
    )
    print(format_comparison_summary(comparison))
    comparison_path = OUTPUT_DIR / "learned_vs_reference.png"
    save_trace_comparison_plot(reference_trace, learned_trace, comparison_path)
    print("trace_comparison_plot", comparison_path)


if __name__ == "__main__":
    main()
