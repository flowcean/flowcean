import logging
from collections.abc import Iterable
from os import PathLike
from pathlib import Path

import polars as pl
from custom_metrics.euclidean_distance import MeanEuclideanDistance
from matplotlib import pyplot as plt

import flowcean
import flowcean.cli
from flowcean.core import Lambda, evaluate_offline, learn_offline
from flowcean.core.model import Model
from flowcean.core.strategies.offline import (
    print_report_table,
    select_best_model,
)
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
from flowcean.torch import LightningLearner, MultilayerPerceptron

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


def plot_predictions_vs_ground_truth(
    samples_eval: pl.DataFrame,
    models: dict[str, Model],
    input_names: list[str],
    output_names: list[str],
) -> None:
    # check if plots directory exists
    Path("plots").mkdir(exist_ok=True)
    for model_name, model in models.items():
        predictions = model.predict(
            samples_eval.select(input_names).lazy(),
        ).collect()
        # create x-y plot
        plt.figure(figsize=(12, 12))
        plt.scatter(
            samples_eval.select(
                pl.col("/turtle1/pose/x_next"),
            ).to_series(),
            samples_eval.select(
                pl.col("/turtle1/pose/y_next"),
            ).to_series(),
            label="Ground Truth",
            color="red",
        )
        plt.scatter(
            model.predict(samples_eval.select(input_names).lazy())
            .collect()
            .select(pl.col("/turtle1/pose/x_next"))
            .to_series(),
            model.predict(samples_eval.select(input_names).lazy())
            .collect()
            .select(pl.col("/turtle1/pose/y_next"))
            .to_series(),
            label="Predictions",
            color="blue",
        )
        plt.title(f"2D Trajectory - {model_name}")
        plt.xlabel("x position")
        plt.ylabel("y position")
        plt.legend()
        plt.savefig(Path(f"plots/{model_name}_2d_trajectory.png"))
        plt.close()
        # create time series plots
        for output_name in output_names:
            plt.figure(figsize=(12, 6))
            plt.plot(
                samples_eval.select(pl.col(output_name)).to_series(),
                label="Ground Truth",
                color="blue",
            )
            plt.plot(
                predictions.select(pl.col(output_name)).to_series(),
                label="Predictions",
                color="red",
            )
            plt.title(
                f"Predictions vs Ground Truth - {model_name} - {output_name}",
            )
            plt.xlabel("Sample Index")
            plt.ylabel(output_name)
            plt.legend()
            plt.savefig(
                Path(
                    f"plots/{model_name}_{output_name.replace('/', '_')}.png",
                ),
            )
            plt.close()


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
        print_report_table(report)
        reports[model_name] = report

    best_model_name = select_best_model(
        reports,
        output_name="multi_output",
        metric_name="MeanEuclideanDistance",
    )
    logger.info("Best model: %s", best_model_name)
    plot_predictions_vs_ground_truth(
        samples_eval=samples_eval.observe().collect(),
        input_names=inputs,
        output_names=outputs,
        models=models,
    )


if __name__ == "__main__":
    main()
