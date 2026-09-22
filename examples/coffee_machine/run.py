#!/usr/bin/env python

from pathlib import Path
from typing import override

import polars as pl
from tqdm import tqdm

import flowcean.cli
from flowcean.aalpy import RPNIMealyLearner
from flowcean.core import (
    ChainedOfflineEnvironments,
    Metric,
    evaluate_offline,
    learn_offline,
)
from flowcean.polars import (
    DataFrame,
    ToTimeSeries,
    TrainTestSplit,
    collect,
)


class TraceAccuracy(Metric):
    """Fraction of completely correct output words, ignoring timestamps."""

    @override
    def _compute(self, true: pl.LazyFrame, predicted: pl.LazyFrame) -> float:
        expected = (
            true.select(
                pl.col("output").list.eval(
                    pl.element()
                    .struct.field("value")
                    .sort_by(
                        pl.element().struct.field("time"),
                        maintain_order=True,
                    ),
                ),
            )
            .collect()
            .to_series()
        )
        actual = predicted.collect().to_series()
        return sum((expected == actual).to_list()) / len(expected)


def main() -> None:
    flowcean.cli.initialize()

    paths = sorted(Path("./data").glob("*.csv"))
    if not paths:
        msg = "No Coffee Machine CSV traces found. Run 'uv run dvc pull --recursive examples/coffee_machine' from the repository root."
        raise FileNotFoundError(msg)

    data = ChainedOfflineEnvironments(
        [
            DataFrame.from_uri("file:" + path.as_posix()) | ToTimeSeries("t")
            for path in tqdm(
                paths,
                desc="Loading environments",
            )
        ],
    )
    train, test = TrainTestSplit(ratio=0.8, shuffle=False).split(
        collect(data),
    )

    learner = RPNIMealyLearner()
    inputs = ["input"]
    outputs = ["output"]

    model = learn_offline(
        train,
        learner,
        inputs,
        outputs,
    )

    report = evaluate_offline(
        model,
        test,
        inputs,
        outputs,
        [TraceAccuracy()],
    )
    print(report)


if __name__ == "__main__":
    main()
