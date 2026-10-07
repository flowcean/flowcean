from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from pysr import PySRRegressor

import flowcean.cli
import flowcean.utils
from flowcean.hybrid import plot_trajectory, simulate
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
    print("selector_flow_summary")
    print(selector.flow_summary_text())
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
    reference_frame: pl.DataFrame,
) -> tuple[pl.DataFrame, StateTraceComparison]:
    learned_frame = model.simulate(
        (float(reference_frame["t"][0]), float(reference_frame["t"][-1])),
        reference_frame.select("x").row(0),
        sample_times=reference_frame["t"].to_numpy(),
        input_stream=thermostat_target_stream,
    ).rename({"x0": "x"})
    return learned_frame, compare_state_traces(
        reference_frame, learned_frame, state_columns=["x"]
    )


def save_trace_comparison_plot(
    reference_frame: pl.DataFrame,
    learned_frame: pl.DataFrame,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    ax.plot(reference_frame["t"], reference_frame["x"], label="reference x")
    ax.plot(
        learned_frame["t"],
        learned_frame["x"],
        label="learned x",
        linestyle="--",
    )
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
    reference_trajectories = [
        simulate(
            system,
            t_span=(0.0, 20.0),
            x0=[temperature],
            input_stream=thermostat_target_stream,
        )
        for temperature in (20.0, 20.5)
    ]
    reference_trajectory = reference_trajectories[0]

    print(
        "Plotting reference trajectory... close plot to continue...",
    )
    plot_trajectory(
        reference_trajectory,
        show_locations=True,
        show_location_labels=False,
        show_events=True,
        show_event_labels=False,
        show=True,
    )
    frames = [
        trajectory.sample(
            dt=0.02, include_derivatives=True, include_inputs=True
        ).rename({"x0": "x", "dx0": "dx", "u0": "target"})
        for trajectory in reference_trajectories
    ]
    reference_frame = frames[0]
    schema = HyDRATraceSchema(
        time="t", state=("x",), derivative=("dx",), inputs=("target",)
    )
    callback = PlotCallback(reference_frame, columns=["x"], time_column="t")
    learner = HyDRALearner(
        regressor_factory=lambda: PySRLearner(
            model=PySRRegressor(
                niterations=10,
                # random_state=flowcean.utils.get_seed(),
            ),
        ),
        threshold=1e-2,
        callback=callback,
    )

    result = learner.learn(
        frames,
        input_features=schema.input_features,
        output_features=schema.derivative,
    )
    flow_models = [flow.model for flow in result.flows]
    # Previous temperature helps the selector distinguish heating from cooling.
    selector = HybridDecisionTreeLearner(
        SelectorFeatureConfig(
            state_features=("x",), input_features=("target",), state_history=1
        ),
        random_state=7,
    ).learn_from_traces(
        result.to_labeled_frames(frames),
        flow_models_by_id=dict(enumerate(flow_models)),
    )
    model = HyDRAModel(
        flow_models,
        input_features=result.input_features,
        output_features=result.output_features,
        selector=selector,
        trace_schema=schema,
    )
    diagnostics = model.predict_with_diagnostics(
        reference_frame.select(schema.input_features)
    )
    print("batch_prediction_rows_after_warmup", len(diagnostics.row_indices))

    print(
        {
            "rows": reference_frame.height,
            "locations": reference_frame["location_id"]
            .unique()
            .sort()
            .to_list(),
            "flow_count": len(result.flows),
            "input_features": model.input_features,
            "output_features": model.output_features,
        },
    )
    if model.selector is not None:
        print_selector_outputs(model.selector, output_dir=OUTPUT_DIR)

    if len(result.flows) > 1:
        print(
            "Rollout skipped: this selector requires history. Stateful learned-model rollout is not implemented."
        )
        return

    learned_frame, comparison = compare_learned_model_to_reference(
        model,
        reference_frame,
    )
    print(format_comparison_summary(comparison))
    comparison_path = OUTPUT_DIR / "learned_vs_reference.png"
    save_trace_comparison_plot(reference_frame, learned_frame, comparison_path)
    print("trace_comparison_plot", comparison_path)


if __name__ == "__main__":
    main()
