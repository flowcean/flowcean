#!/usr/bin/env python

import logging

import polars as pl
from sklearn.cluster import KMeans

import flowcean.cli
from flowcean.core import evaluate_offline
from flowcean.core.strategies.offline import learn_offline
from flowcean.ensemble.cluster_learner import ClusterLearner
from flowcean.polars import (
    DataFrame,
)
from flowcean.sklearn import (
    LinearRegression,
    MeanAbsoluteError,
    MeanSquaredError,
)

logger = logging.getLogger(__name__)

N = 1_000


def main() -> None:
    flowcean.cli.initialize()

    df_a = pl.DataFrame(
        {
            "x": pl.arange(0, N, eager=True).cast(pl.Float32) / N,
        },
    )
    df_a = df_a.with_columns(
        (pl.col("x") * 1.0 + 0.1).alias("y"),
    )

    df_b = pl.DataFrame(
        {
            "x": pl.arange(2 * N, 3 * N, eager=True).cast(pl.Float32) / N,
        },
    )
    df_b = df_b.with_columns(
        (pl.col("x") * -1.0 + 3.0).alias("y"),
    )

    data = DataFrame(pl.concat([df_a, df_b]))

    learner = ClusterLearner(
        KMeans(n_clusters=2),
        LinearRegression(),
    )

    inputs = ["x"]
    outputs = ["y"]

    model = learn_offline(
        data,
        learner,
        inputs,
        outputs,
    )

    report = evaluate_offline(
        model,
        data,
        inputs,
        outputs,
        [MeanAbsoluteError(), MeanSquaredError()],
    )
    print(report)


if __name__ == "__main__":
    main()
