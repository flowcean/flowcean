#!/usr/bin/env python
import logging
from collections.abc import Iterable
from os import PathLike
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
from flowcean.core import Transform
from flowcean.core.strategies import evaluate_offline
from flowcean.core.transform import Lambda
from flowcean.polars import DataFrame
from flowcean.polars.transforms.drop import Drop
from flowcean.ros import load_rosbag
from flowcean.sklearn.metrics.classification import (
    Accuracy,
    ClassificationReport,
    FBetaScore,
    PrecisionScore,
    Recall,
)

logger = logging.getLogger(__name__)


def define_transforms(
    position_threshold: float,
    heading_threshold: float,
) -> Transform:
    def convert_map_to_bool(df: pl.LazyFrame) -> pl.LazyFrame:
        return df.with_columns(
            pl.col("/map").struct.with_fields(
                pl.field("data").list.eval(pl.element() != 0),
            ),
        )

    return (
        Collapse("/map", element=0)
        | Lambda(convert_map_to_bool)
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
            position_threshold=position_threshold,
            heading_threshold=heading_threshold,
        )
    )


def load_and_process_data(
    rosbags: Iterable[str | PathLike],
    config: DictConfig | ListConfig,
) -> pl.LazyFrame:
    processed_data = []
    for path in rosbags:
        cache_path = Path(path).with_suffix(".processed.parquet")
        if cache_path.exists():
            logger.info(
                "Loading already processed rosbag from cache",
                extra={"path": cache_path},
            )
            data = pl.scan_parquet(cache_path)
            processed_data.append(data)
            continue

        logger.info("Processing rosbag", extra={"path": path})
        data = load_rosbag(
            path=path,
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
        transform = define_transforms(
            config.localization.position_threshold,
            config.localization.heading_threshold,
        )
        data = transform.apply(data)
        data.sink_parquet(cache_path)
        processed_data.append(data)
    return pl.concat(processed_data)


def create_image_dataset(
    data: pl.LazyFrame,
) -> pl.LazyFrame:
    return (
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


def train_and_evaluate(
    train_data: pl.LazyFrame,
    test_data: pl.LazyFrame,
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
        inputs=train_data.drop(["is_delocalized"]).collect(engine="streaming"),
        outputs=train_data.select(["is_delocalized"]).collect(
            engine="streaming",
        ),
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


def main() -> None:
    config = flowcean.cli.initialize()

    processed_train_data = load_and_process_data(
        rosbags=config.rosbag.training_paths,
        config=config,
    )
    processed_evaluation_data = load_and_process_data(
        rosbags=config.rosbag.evaluation_paths,
        config=config,
    )
    logger.info("Creating image datasets from processed data")
    train_image_dataset = create_image_dataset(
        processed_train_data,
    )
    test_image_dataset = create_image_dataset(
        processed_evaluation_data,
    )
    out_path = (
        f"models/{config.model_name}_{config.architecture.image_size}p_"
        f"{config.architecture.width_meters}m.pt"
    )
    train_and_evaluate(
        train_image_dataset,
        test_image_dataset,
        config,
        out_path,
    )


if __name__ == "__main__":
    main()
