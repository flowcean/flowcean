#!/usr/bin/env python
import logging
from pathlib import Path

import polars as pl
import torch
from architectures.cnn import CNN
from custom_learners.image_based_lightning_learner import (
    ImageBasedLightningLearner,
)
from custom_transforms.collapse import Collapse
from custom_transforms.detect_delocalizations import DetectDelocalizations
from custom_transforms.localization_status import LocalizationStatus
from custom_transforms.slice_time_series import SliceTimeSeries
from custom_transforms.zero_order_hold_matching import ZeroOrderHold
from omegaconf import DictConfig, ListConfig

import flowcean.cli
from flowcean.core.strategies import evaluate_offline
from flowcean.core.transform import Lambda
from flowcean.polars import DataFrame
from flowcean.polars.transforms.drop import Drop
from flowcean.ros.rosbag import RosbagLoader
from flowcean.sklearn.metrics.classification import (
    Accuracy,
    ClassificationReport,
    FBetaScore,
    PrecisionScore,
    Recall,
)

logger = logging.getLogger(__name__)


def load_and_process_data(
    path: str | list[str],
    config: DictConfig | ListConfig,
) -> pl.LazyFrame:
    processed_data = []
    # Process each ROS2 bag directory if not already processed
    for rosbag_path in path:
        out_path = Path(rosbag_path).with_suffix(".processed.parquet")
        if out_path.exists():
            processed_data.append(pl.scan_parquet(out_path))
            logger.info(
                "Loading already processed rosbag from parquet",
                extra={"path": out_path},
            )
            continue

        rosbag = RosbagLoader(
            path=rosbag_path,
            topics={
                "/amcl_pose": [
                    "pose.pose.position.x",
                    "pose.pose.position.y",
                    "pose.pose.orientation.x",
                    "pose.pose.orientation.y",
                    "pose.pose.orientation.z",
                    "pose.pose.orientation.w",
                ],
                "/momo/pose": [
                    "pose.position.x",
                    "pose.position.y",
                    "pose.orientation.x",
                    "pose.orientation.y",
                    "pose.orientation.z",
                    "pose.orientation.w",
                ],
                "/scan": [
                    "ranges",
                    "angle_min",
                    "angle_max",
                    "angle_increment",
                    "range_min",
                    "range_max",
                ],
                "/map": [
                    "data",
                    "info.resolution",
                    "info.width",
                    "info.height",
                    "info.origin.position.x",
                    "info.origin.position.y",
                    "info.origin.position.z",
                    "info.origin.orientation.x",
                    "info.origin.orientation.y",
                    "info.origin.orientation.z",
                    "info.origin.orientation.w",
                ],
                "/delocalizations": ["data"],
                "/particle_cloud": ["particles"],
            },
            message_paths=config.rosbag.message_paths,
        )

        convert_map_to_bool = Lambda(
            lambda df: df.with_columns(
                pl.col("/map").struct.with_fields(
                    pl.field("data").list.eval(pl.element() != 0),
                ),
            ),
        )
        # Apply preprocessing pipeline
        data = (
            rosbag
            | Collapse("/map", element=0)
            | convert_map_to_bool
            | ZeroOrderHold(
                features=[
                    "/scan",
                    "/particle_cloud",
                    "/momo/pose",
                    "/amcl_pose",
                ],
                name="measurements",
            )
            | Drop("/scan", "/particle_cloud", "/momo/pose", "/amcl_pose")
            | DetectDelocalizations("/delocalizations", name="slice_points")
            | Drop("/delocalizations")
            | SliceTimeSeries(
                time_series="measurements",
                slice_points="slice_points",
            )
            | Drop("slice_points")
            | LocalizationStatus(
                time_series="measurements",
                ground_truth="/momo/pose",
                estimation="/amcl_pose",
                position_threshold=config.localization.position_threshold,
                heading_threshold=config.localization.heading_threshold,
            )
        )
        try:
            logger.info(
                "Processing rosbag",
                extra={"path": out_path},
            )
            data.observe().sink_parquet(out_path)
            processed_data.append(data.observe())
        except Exception:
            logger.exception(
                "Error processing rosbag",
                extra={"path": out_path},
            )
            continue
    return pl.concat(processed_data)


def create_image_dataset(
    data: pl.LazyFrame,
) -> pl.DataFrame:
    processed_data = (
        data.explode("measurements")
        .unnest("measurements")
        .unnest("value")
        .select(
            pl.col("/map"),
            pl.struct(
                [
                    pl.col("/scan/ranges").alias("ranges"),
                    pl.col("/scan/angle_min").alias("angle_min"),
                    pl.col("/scan/angle_max").alias("angle_max"),
                    pl.col("/scan/angle_increment").alias(
                        "angle_increment",
                    ),
                    pl.col("/scan/range_min").alias("range_min"),
                    pl.col("/scan/range_max").alias("range_max"),
                ],
            ).alias("/scan"),
            pl.col("/particle_cloud/particles").alias(
                "/particle_cloud",
            ),
            pl.struct(
                [
                    pl.struct(
                        [
                            pl.col(
                                "/amcl_pose/pose.pose.position.x",
                            ).alias("position.x"),
                            pl.col(
                                "/amcl_pose/pose.pose.position.y",
                            ).alias("position.y"),
                            pl.col(
                                "/amcl_pose/pose.pose.orientation.x",
                            ).alias("orientation.x"),
                            pl.col(
                                "/amcl_pose/pose.pose.orientation.y",
                            ).alias("orientation.y"),
                            pl.col(
                                "/amcl_pose/pose.pose.orientation.z",
                            ).alias("orientation.z"),
                            pl.col(
                                "/amcl_pose/pose.pose.orientation.w",
                            ).alias("orientation.w"),
                        ],
                    ).alias("pose"),
                ],
            ).alias("/amcl_pose"),
            pl.col("is_delocalized"),
        )
    )
    return processed_data.collect(engine="streaming")


def main() -> None:
    # Initialize configuration and logging
    config = flowcean.cli.initialize()

    processed_train_data = load_and_process_data(
        path=config.rosbag.training_paths,
        config=config,
    )
    processed_evaluation_data = load_and_process_data(
        path=config.rosbag.evaluation_paths,
        config=config,
    )
    logger.info("Creating image datasets from processed data")
    train_image_dataset = create_image_dataset(
        processed_train_data,
    )
    test_image_dataset = create_image_dataset(
        processed_evaluation_data,
    )
    out_path = f"models/{config.model_name}.pt"
    train_and_evaluate(
        train_image_dataset,
        test_image_dataset,
        config,
        out_path,
    )


def train_and_evaluate(
    train_data: pl.DataFrame,
    test_data: pl.DataFrame,
    config: DictConfig | ListConfig,
    out_path: str,
) -> None:
    # Create and train the learner
    learner = ImageBasedLightningLearner(
        module=CNN(
            image_size=config.architecture.image_size,
            in_channels=3,
            learning_rate=config.learning.learning_rate,
        ),
        batch_size=config.learning.batch_size,
        max_epochs=config.learning.epochs,
        image_size=config.architecture.image_size,
        width_meters=config.architecture.width_meters,
    )
    logger.info("Training model with %s epochs", config.learning.epochs)
    model = learner.learn(
        inputs=train_data.drop(["is_delocalized"]),
        outputs=train_data.select(["is_delocalized"]),
    )
    # Evaluate the model
    metrics = [
        Accuracy(),
        ClassificationReport(),
        FBetaScore(beta=1.0),
        PrecisionScore(),
        Recall(),
    ]
    logger.info("Evaluating model on test data")
    report = evaluate_offline(
        model,
        DataFrame(test_data),
        inputs=test_data.drop(["is_delocalized"]).columns,
        outputs=["is_delocalized"],
        metrics=metrics,
    )
    print(report)
    # Save the model
    logger.info("Saving model to %s", out_path)
    torch.save(model.module.state_dict(), out_path)


if __name__ == "__main__":
    main()
