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
from flowcean.polars import DataFrame, Lambda, TrainTestSplit, collect


class TraceAccuracy(Metric):
    """Fraction of completely correct output words."""

    @override
    def _compute(self, true: pl.LazyFrame, predicted: pl.LazyFrame) -> float:
        expected = true.collect().to_series()
        actual = predicted.collect().to_series()
        return sum((expected == actual).to_list()) / len(expected)


def _to_words(data: pl.LazyFrame) -> pl.LazyFrame:
    """Sort one synchronized trace and collect its input/output words."""
    return data.sort("t", maintain_order=True).select(
        pl.col("input").implode(),
        pl.col("output").implode(),
    )


def main() -> None:
    flowcean.cli.initialize()

    paths = sorted(Path("./data").glob("*.csv"))
    if not paths:
        msg = "No Coffee Machine CSV traces found. Run 'uv run dvc pull --recursive examples/coffee_machine' from the repository root."
        raise FileNotFoundError(msg)

    data = ChainedOfflineEnvironments(
        [
            DataFrame.from_uri("file:" + path.as_posix()) | Lambda(_to_words)
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
