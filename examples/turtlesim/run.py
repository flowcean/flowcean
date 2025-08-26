import logging
from collections.abc import Iterable
from os import PathLike

import polars as pl
from custom_metrics.euclidean_distance import MeanEuclideanDistance

import flowcean
import flowcean.cli
from flowcean.core import Lambda, evaluate_offline, learn_offline
from flowcean.polars import DataFrame, ExplodeTimeSeries, ZeroOrderHold
from flowcean.ros import load_rosbag
from flowcean.sklearn import (
    MaxError,
    MeanAbsoluteError,
    MeanSquaredError,
    R2Score,
    RandomForestRegressorLearner,
    RegressionTree,
)
from flowcean.torch import (
    LightningLearner,
    MultilayerPerceptron,
)

logger = logging.getLogger(__name__)


def shift_in_time(df: pl.LazyFrame) -> pl.LazyFrame:
    return df.with_columns(
        pl.col("/turtle1/pose/x", "/turtle1/pose/y", "/turtle1/pose/theta")
        .shift(-1)
        .name.suffix("_next"),
    ).filter(pl.col("/turtle1/pose/x_next").is_not_null())


def load_and_process_rosbag(
    path: str | PathLike,
    message_paths: Iterable[str | PathLike] | None = None,
) -> DataFrame:
    logger.info("Loading rosbag from: %s", path)
    rosbag = load_rosbag(
        path=path,
        topics={
            "/turtle1/cmd_vel": [
                "linear.x",
                "angular.z",
            ],
            "/turtle1/pose": [
                "x",
                "y",
                "theta",
            ],
        },
        message_paths=message_paths,
    )
    return (
        DataFrame(rosbag)
        | ZeroOrderHold(
            features=[
                "/turtle1/cmd_vel",
                "/turtle1/pose",
            ],
            name="measurements",
        )
        | ExplodeTimeSeries("measurements")
        | Lambda(shift_in_time)
    )


def main() -> None:
    config = flowcean.cli.initialize()

    samples_train = load_and_process_rosbag(
        config.rosbag.training_path,
        config.rosbag.message_paths,
    )
    samples_eval = load_and_process_rosbag(
        config.rosbag.evaluation_path,
        config.rosbag.message_paths,
    )

    regression_tree = RegressionTree(**config.training.tree)
    random_forest = RandomForestRegressorLearner(
        **config.training.forest,
    )
    mlp = LightningLearner(
        module=MultilayerPerceptron(
            learning_rate=config.training.mlp.learning_rate,
            # TODO: remove magic value by lazily infering from data
            input_size=5,
            output_size=3,
        ),
        batch_size=config.training.mlp.batch_size,
        max_epochs=config.training.mlp.max_epochs,
    )
    learners = {
        "regression_tree": regression_tree,
        "random_forest": random_forest,
        "multilayer_perceptron": mlp,
    }

    inputs = [
        "/turtle1/pose/x",
        "/turtle1/pose/y",
        "/turtle1/pose/theta",
        "/turtle1/cmd_vel/linear.x",
        "/turtle1/cmd_vel/angular.z",
    ]
    outputs = [
        "/turtle1/pose/x_next",
        "/turtle1/pose/y_next",
        "/turtle1/pose/theta_next",
    ]

    models = {}
    for learner_name, learner in learners.items():
        logger.info("Training model: %s", learner_name)
        model = learn_offline(
            samples_train,
            learner,
            inputs=inputs,
            outputs=outputs,
        )
        models[learner_name] = model

    metrics = [
        MaxError(),
        MeanAbsoluteError(),
        MeanSquaredError(),
        R2Score(),
        MeanEuclideanDistance(
            columns=[
                "/turtle1/pose/x_next",
                "/turtle1/pose/y_next",
            ],
        ),
    ]

    reports = {}
    for model_name, model in models.items():
        logger.info("Evaluating model: %s", model_name)
        report = evaluate_offline(
            model=model,
            environment=samples_eval,
            metrics=metrics,
            inputs=inputs,
            outputs=outputs,
        )
        print(report)
        reports[model_name] = report


if __name__ == "__main__":
    main()
