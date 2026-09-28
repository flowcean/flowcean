from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from pysr import PySRRegressor

import flowcean.cli
import flowcean.utils
from flowcean.hybrid import plot_trace, simulate
from flowcean.hybrid.benchmarks import thermostat
from flowcean.hybrid.hydra import (
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
from flowcean.pysr import PySRLearner

EXAMPLE_SEED = 42
HYDRA_LOGGER = "flowcean.hybrid.hydra.learner"
OUTPUT_DIR = Path("outputs")


def thermostat_target_stream(t: float) -> np.ndarray:
    """Target temperature for this example's reference trajectory."""
    return np.array([22.0 + 0.8 * np.sin(0.7 * t)], dtype=float)


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
    reference: pl.DataFrame,
) -> tuple[pl.DataFrame, StateTraceComparison]:
    learned_trace = model.simulate(
        (float(reference["t"][0]), float(reference["t"][-1])),
        reference.select("x").row(0),
        sample_times=reference["t"].to_numpy(),
    ).rename({"x0": "x"})
    return learned_trace, compare_state_traces(
        reference, learned_trace, state_columns=["x"]
    )


def save_trace_comparison_plot(
    reference: pl.DataFrame,
    learned: pl.DataFrame,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    ax.plot(reference["t"], reference["x"], label="reference x")
    ax.plot(learned["t"], learned["x"], label="learned x", linestyle="--")
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
    trace_frame = reference_trace.sample(
        dt=0.02, include_derivatives=True
    ).rename({"x0": "x", "dx0": "dx"})
    schema = HyDRATraceSchema(time="t", state=("x",), derivative=("dx",))
    callback = PlotCallback(trace_frame, state_columns=["x"])
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
            "locations": trace_frame["location_id"].unique().sort().to_list(),
            "modes": len(model.modes),
            "input_features": model.input_features,
            "output_features": model.output_features,
        },
    )
    if model.selector is not None:
        print_selector_outputs(model.selector, output_dir=OUTPUT_DIR)

    learned_trace, comparison = compare_learned_model_to_reference(
        model,
        trace_frame,
    )
    print(format_comparison_summary(comparison))
    comparison_path = OUTPUT_DIR / "learned_vs_reference.png"
    save_trace_comparison_plot(trace_frame, learned_trace, comparison_path)
    print("trace_comparison_plot", comparison_path)


if __name__ == "__main__":
    main()
