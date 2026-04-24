#!/usr/bin/env python

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import matplotlib.pyplot as plt

import flowcean.cli
from flowcean.hydra import (
    HybridDecisionTreeLearner,
    HyDRALearner,
    HyDRALivePlotCallback,
    SelectorFeatureConfig,
)
from flowcean.polars import DataFrame
from flowcean.utils import get_seed, initialize_random

if TYPE_CHECKING:
    from collections.abc import Sequence

    from flowcean.hydra.model import HyDRAModel

EXAMPLE_SEED = 42


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the passive circuit example.",
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


def main(*, live_plot: bool = False) -> None:
    from pysr import PySRRegressor

    from flowcean.pysr import PySRLearner

    flowcean.cli.initialize()
    initialize_random(EXAMPLE_SEED)

    train = DataFrame.from_uri("file:./data/circuit_data.csv")
    raw_trace = train.data.collect()
    inputs = ["U1", "U2", "U3", "R"]
    outputs = ["I1"]
    selector_learner = HybridDecisionTreeLearner(
        SelectorFeatureConfig(input_columns=tuple(inputs)),
        random_state=7,
    )
    callback = None
    if live_plot:
        callback = HyDRALivePlotCallback(
            traces=[raw_trace.select(inputs + outputs)],
            y_columns=("I1",),
            x_column=None,
        )
    learner = HyDRALearner(
        regressor_factory=lambda: PySRLearner(
            model=PySRRegressor(
                niterations=10,
                verbosity=0,
                random_state=get_seed(),
                deterministic=True,
                parallelism="serial",
            ),
        ),
        threshold=1e-5,
        start_width=400,
        step_width=200,
        selector_learner=selector_learner,
        callback=callback,
    )

    model = cast(
        "HyDRAModel",
        learner.learn(
            raw_trace.select(inputs).lazy(),
            raw_trace.select(outputs).lazy(),
        ),
    )

    print(
        {
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
