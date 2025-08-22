import logging
from os import PathLike

import polars as pl
from custom_transforms.zero_order_hold_matching import ZeroOrderHold
from omegaconf import DictConfig, ListConfig

import flowcean
import flowcean.cli
from flowcean.core.strategies import evaluate_offline, learn_offline
from flowcean.polars.environments.dataframe import DataFrame
from flowcean.ros.rosbag import load_rosbag
from flowcean.sklearn.metrics.regression import (
    MaxError,
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MeanSquaredError,
    R2Score,
)
from flowcean.sklearn.regression_tree import RegressionTree

logger = logging.getLogger(__name__)


def explode_and_unnest(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.explode("measurements")
        .unnest("measurements")
        .unnest("value")
        .drop("/turtle1/cmd_vel", "/turtle1/pose", "time")
    )


def shift_columns(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        [
            pl.col("/turtle1/pose/x").shift(-1).alias("/turtle1/pose/x_next"),
            pl.col("/turtle1/pose/y").shift(-1).alias("/turtle1/pose/y_next"),
            pl.col("/turtle1/pose/theta")
            .shift(-1)
            .alias("/turtle1/pose/theta_next"),
        ],
    ).filter(pl.col("/turtle1/pose/x_next").is_not_null())


def load_and_process_rosbag(
    path: str | PathLike,
    config: DictConfig | ListConfig,
) -> pl.DataFrame:
    logger.info("Processing rosbag: %s", path)
    data = load_rosbag(
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
        message_paths=config.rosbag.message_paths,
    )
    transform = ZeroOrderHold(
        features=[
            "/turtle1/cmd_vel",
            "/turtle1/pose",
        ],
        name="measurements",
    )
    transformed_data: pl.DataFrame = transform.apply(data).collect(
        engine="streaming",
    )
    transformed_data = explode_and_unnest(transformed_data)
    logger.info("After exploding and unnesting: %s", transformed_data)
    transformed_data = shift_columns(transformed_data)
    logger.info("After shifting columns: %s", transformed_data)
    return transformed_data


def main() -> None:
    config = flowcean.cli.initialize()
    train_path = config.rosbag.training_paths[0]
    logger.info("Loading training rosbag from: %s", train_path)
    samples_train = load_and_process_rosbag(train_path, config)
    eval_path = config.rosbag.evaluation_paths[0]
    logger.info("Loading evaluation rosbag from: %s", eval_path)
    samples_eval = load_and_process_rosbag(eval_path, config)
    logger.info("Training samples: %s", samples_train)
    logger.info("Evaluation samples: %s", samples_eval)
    input_names = samples_train.drop(
        "/turtle1/pose/x_next",
        "/turtle1/pose/y_next",
        "/turtle1/pose/theta_next",
    ).columns
    output_names = samples_train.select(
        "/turtle1/pose/x_next",
        "/turtle1/pose/y_next",
        "/turtle1/pose/theta_next",
    ).columns
    tree_params = {
        "max_leaf_nodes": 1000,
    }
    regression_tree_learner = RegressionTree(**tree_params)
    regression_tree_model = learn_offline(
        DataFrame(samples_train),
        regression_tree_learner,
        inputs=input_names,
        outputs=output_names,
    )
    metrics = [
        MaxError(),
        MeanAbsoluteError(),
        MeanAbsolutePercentageError(),
        MeanSquaredError(),
        R2Score(),
    ]

    example_output = regression_tree_model.predict(
        samples_eval.select(input_names).limit(10).lazy(),
    )
    logger.info("Example output: %s", example_output.collect())

    for output_name in output_names:
        print(f"\nEvaluating {output_name}:")
        report = evaluate_offline(
            model=regression_tree_model,
            environment=DataFrame(samples_eval),
            metrics=metrics,
            inputs=input_names,
            outputs=[output_name],
        )
        formatted_report = "\n".join(
            f"  {metric_name}: {value:.2e}"
            if metric_name == "MeanAbsolutePercentageError"
            else f"  {metric_name}: {value:.4f}"
            for metric_name, value in report[output_name].items()
        )
        logger.info(
            "Evaluation report for %s:\n%s",
            output_name,
            formatted_report,
        )


if __name__ == "__main__":
    main()
