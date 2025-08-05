#!/usr/bin/env python
import logging
import sys
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
from feature_images import FeatureImagesData
from omegaconf import DictConfig, ListConfig

import flowcean.cli
from flowcean.core.strategies import evaluate_offline
from flowcean.core.transform import Lambda
from flowcean.polars import DataFrame
from flowcean.polars.environments.train_test_split import TrainTestSplit
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


def main() -> None:
    # Initialize configuration and logging
    config = flowcean.cli.initialize()
    rosbag_dir, rosbag_dirs = get_rosbag_paths(config)

    # Process each ROS2 bag directory if not already processed
    for rosbag_path in rosbag_dirs:
        out_path = rosbag_path.with_suffix(".processed.parquet")
        if out_path.exists():
            msg = "Processed data exists for a rosbag, skipping."
            logger.info(msg)
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

        # Apply preprocessing pipeline
        data = (
            rosbag
            | Collapse("/map", element=0)
            | Lambda(
                lambda df: df.with_columns(
                    pl.col("/map").struct.with_fields(
                        pl.field("data").list.eval(pl.element() != 0),
                    ),
                ),
            )
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
            data.observe().sink_parquet(out_path)
            logger.info("Processed data saved to parqet")
        except Exception:
            msg = "Error processing rosbag"
            logger.exception(msg)
            continue

    # Training: Load all processed Parquet files
    parquet_files = list(rosbag_dir.glob("*.processed.parquet"))
    if not parquet_files:
        msg = "No processed Parquet files found in specified directory."
        logger.error(msg)
        sys.exit(1)

    train_datasets, test_datasets = [], []
    for parquet_path in parquet_files:
        msg = "Loading Parquet file"
        logger.info(msg)
        try:
            data = pl.read_parquet(parquet_path)
            data = (
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
            # Split the data into train and test
            data = DataFrame(data)
            train, test = TrainTestSplit(ratio=0.8, shuffle=True).split(data)
            train_images = FeatureImagesData(
                train.data.collect(),
                image_size=config.architecture.image_size,
                width_meters=config.architecture.width_meters,
            )
            train_datasets.append(train_images)
            test_images = FeatureImagesData(
                test.data.collect(),
                image_size=config.architecture.image_size,
                width_meters=config.architecture.width_meters,
            )
            test_datasets.append(test_images)
        except Exception:
            msg = "Error loading parquet file"
            logger.exception(msg)

    # Combinedatasets
    train_data = pl.concat([train.data for train in train_datasets])
    test_data = pl.concat([test.data for test in test_datasets])

    out_path = f"models/{rosbag_dir.name}.pt"
    train_and_evaluate(train_data, test_data, config, out_path)


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
        dataset_factory=lambda data: FeatureImagesData(
            data,
            image_size=config.architecture.image_size,
            width_meters=config.architecture.width_meters,
        ),
        dataset_kwargs={
            "image_size": config.architecture.image_size,
            "width_meters": config.architecture.width_meters,
        },
    )
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
    report = evaluate_offline(
        model,
        DataFrame(test_data),
        inputs=test_data.drop(["is_delocalized"]).columns,
        outputs=["is_delocalized"],
        metrics=metrics,
    )
    print(report)
    # Save the model
    msg = "Saving model."
    logger.info(msg)
    torch.save(model.module.state_dict(), out_path)


def get_rosbag_paths(
    config: DictConfig | ListConfig,
) -> tuple[Path, list[Path]]:
    rosbag_dir = Path(config.rosbag.path)

    if not rosbag_dir.is_dir():
        msg = "Specified ROS2 bag directory does not exist."
        logger.error(msg)
        raise FileNotFoundError(msg)

    # Find ROS2 bag subdirectories
    rosbag_dirs = [
        d
        for d in rosbag_dir.iterdir()
        if d.is_dir() and (d / "metadata.yaml").exists()
    ]
    if not rosbag_dirs:
        msg = "No ROS2 bag directories found in specified directory."
        logger.error(msg)
        sys.exit(1)

    return rosbag_dir, rosbag_dirs


if __name__ == "__main__":
    main()
