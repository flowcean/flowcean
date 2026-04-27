from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import matplotlib.pyplot as plt
import numpy as np

import flowcean.cli
from flowcean.hydra import (
    HybridDecisionTreeLearner,
    HyDRALearner,
    HyDRALivePlotCallback,
    SelectorFeatureConfig,
)
from flowcean.ode import (
    ContinuousDynamics,
    Guard,
    HybridSystem,
    InputStream,
    Location,
    Transition,
    simulate,
    trace_to_polars,
)
from flowcean.utils import get_seed, initialize_random

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import polars as pl

    from flowcean.hydra.model import HyDRAModel

EXAMPLE_SEED = 42


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the simulated hybrid system HyDRA example.",
    )
    parser.add_argument(
        "--live-plot",
        action="store_true",
        help="show HyDRA live plot callbacks while learning",
    )
    return parser.parse_args(argv)


class SelectorDiagnostics(Protocol):
    def summary_text(self) -> str: ...

    def mode_summary_text(self) -> str: ...

    def leaf_summary_text(self) -> str: ...

    def tree_text(self) -> str: ...

    def save_svg(self, path: Path) -> None: ...


def print_selector_outputs(
    selector: SelectorDiagnostics,
    output_dir: Path = Path("outputs"),
) -> None:
    print("selector_summary")
    print(selector.summary_text())
    print("selector_mode_summary")
    print(selector.mode_summary_text())
    print("selector_leaf_summary")
    print(selector.leaf_summary_text())
    print("selector_tree")
    print(selector.tree_text())

    svg_path = output_dir / "selector_tree.svg"
    try:
        output_dir.mkdir(exist_ok=True)
        selector.save_svg(svg_path)
    except (RuntimeError, OSError) as exc:
        print("selector_svg_skipped", exc)
        return

    print("selector_svg", svg_path)


def switched_affine_1d() -> HybridSystem:
    def left_flow(
        _: float,
        state: np.ndarray,
        _params: Mapping[str, float],
        _input: InputStream,
    ) -> np.ndarray:
        return np.array([-0.6 * state[0] + 1.0], dtype=float)

    def right_flow(
        _: float,
        state: np.ndarray,
        _params: Mapping[str, float],
        _input: InputStream,
    ) -> np.ndarray:
        return np.array([-1.2 * state[0] + 3.0], dtype=float)

    def guard_to_right(
        _: float,
        state: np.ndarray,
        params: Mapping[str, float],
        _input: InputStream,
    ) -> float:
        return state[0] - params["threshold"]

    left = Location(
        name="left",
        dynamics=ContinuousDynamics(name="left_flow", flow=left_flow),
    )
    right = Location(
        name="right",
        dynamics=ContinuousDynamics(name="right_flow", flow=right_flow),
    )

    return HybridSystem(
        locations={"left": left, "right": right},
        transitions=[
            Transition(
                source_location="left",
                target_location="right",
                guard=Guard(
                    name="cross_right",
                    fn=guard_to_right,
                    direction=1,
                ),
            ),
        ],
        initial_location="left",
        initial_state=np.array([-2.0], dtype=float),
        params={"threshold": 0.0},
    )


def simulate_training_data() -> pl.DataFrame:
    trace = simulate(
        switched_affine_1d(),
        t_span=(0.0, 12.0),
        capture_derivatives=True,
        sample_dt=0.02,
    )
    return trace_to_polars(
        trace,
        state_names=("x",),
        derivative_names=("dx",),
    )


def main(*, live_plot: bool = False) -> None:
    from pysr import PySRRegressor

    from flowcean.pysr import PySRLearner

    flowcean.cli.initialize()
    initialize_random(EXAMPLE_SEED)

    trace_frame = simulate_training_data()
    inputs = ["x"]
    outputs = ["dx"]
    selector_learner = HybridDecisionTreeLearner(
        SelectorFeatureConfig(input_columns=tuple(inputs)),
        random_state=7,
    )
    callback = None
    if live_plot:
        callback = HyDRALivePlotCallback(
            traces=[trace_frame.select(inputs + outputs)],
            y_columns=("dx",),
            x_column="x",
        )
    learner = HyDRALearner(
        regressor_factory=lambda: PySRLearner(
            model=PySRRegressor(
                niterations=20,
                binary_operators=["+", "-", "*"],
                unary_operators=[],
                constraints={"*": (1, 1)},
                maxsize=7,
                verbosity=0,
                random_state=get_seed(),
                deterministic=True,
                parallelism="serial",
            ),
        ),
        threshold=1e-2,
        start_width=40,
        step_width=40,
        selector_learner=selector_learner,
        callback=callback,
    )

    model = learner.learn(
        trace_frame.select(inputs).lazy(),
        trace_frame.select(outputs).lazy(),
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
        print_selector_outputs(model.selector)

    if live_plot:
        plt.show()


if __name__ == "__main__":
    args = parse_args()
    main(live_plot=args.live_plot)
