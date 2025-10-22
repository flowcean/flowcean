from __future__ import annotations

import logging
from os import PathLike
from pathlib import Path

import polars as pl
from architectures.cnn import CNN
from custom_learners.image_based_lightning_learner import (
    ImageBasedLightningLearner,
    ImageBasedPyTorchModel,
)
from custom_transforms.collapse import Collapse
from custom_transforms.detect_delocalizations import DetectDelocalizations
from custom_transforms.localization_status import LocalizationStatus
from custom_transforms.slice_time_series import SliceTimeSeries
from custom_transforms.zero_order_hold_matching import ZeroOrderHold
from omegaconf import DictConfig, ListConfig

from flowcean.core import Report, Transform
from flowcean.core.strategies import evaluate_offline
from flowcean.core.transform import Lambda
from flowcean.polars import DataFrame
from flowcean.polars.transforms.drop import Drop
from flowcean.ros import load_rosbag
from flowcean.sklearn.metrics.classification import (
    Accuracy,
    ClassificationReport,
    ConfusionMatrix,
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


def load_and_process_rosbag(
    path: str | PathLike,
    config: DictConfig | ListConfig,
) -> pl.LazyFrame:
    cache_path = Path(path).with_suffix(".processed.parquet")
    if cache_path.exists():
        logger.info(
            "Loading already processed rosbag from cache: %s",
            cache_path,
        )
        return pl.scan_parquet(cache_path)

    logger.info("Processing rosbag: %s", path)
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
    transformed_data: pl.DataFrame = transform.apply(data).collect(
        engine="streaming",
    )

    logger.info("Caching processed data to Parquet file: %s", cache_path)
    transformed_data.write_parquet(cache_path)

    return pl.scan_parquet(cache_path)


def explode_and_collect_samples(data: pl.LazyFrame) -> pl.LazyFrame:
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


def collect_data(
    config: DictConfig | ListConfig,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    logger.info("Collecting training data")
    runs_train_lf = [
        load_and_process_rosbag(path=path, config=config)
        for path in config.rosbag.training_paths
    ]
    logger.info(
        "Lazily combining training data. This takes 8 min for 130 GB of data",
    )
    samples_train_lf = explode_and_collect_samples(
        pl.concat(runs_train_lf, how="vertical"),
    )
    samples_train = samples_train_lf.collect(engine="streaming")

    logger.info("Collecting evaluation data")
    runs_eval_lf = [
        load_and_process_rosbag(path=path, config=config)
        for path in config.rosbag.evaluation_paths
    ]
    logger.info("Combining evaluation data (lazy)")
    samples_eval_lf = explode_and_collect_samples(
        pl.concat(runs_eval_lf, how="vertical"),
    )
    samples_eval = samples_eval_lf.collect(engine="streaming")

    ##########################################
    train_counts = samples_train["is_delocalized"].value_counts()
    eval_counts = samples_eval["is_delocalized"].value_counts()

    print("Training set:")
    print(train_counts)

    print("\nEvaluation set:")
    print(eval_counts)
    ##########################################

    return (samples_train, samples_eval)


#### Previous train function
# def train(
#     train_data: pl.DataFrame,
#     config: DictConfig | ListConfig,
# ) -> ImageBasedPyTorchModel:
#     # check if disk cache dir is empty, if no, remove its contents
#     if config.learning.disk_cache_dir:
#         disk_cache_path = Path(config.learning.disk_cache_dir)
#         if disk_cache_path.exists() and any(disk_cache_path.iterdir()):
#             logger.info(
#                 "Clearing existing disk cache directory: %s",
#                 disk_cache_path,
#             )
#             for item in disk_cache_path.iterdir():
#                 if item.is_dir():
#                     for subitem in item.iterdir():
#                         subitem.unlink()
#                     item.rmdir()
#                 else:
#                     item.unlink()
#         disk_cache_path.mkdir(parents=True, exist_ok=True)
#     learner = ImageBasedLightningLearner(
#         module=CNN(
#             image_size=config.architecture.image_size,
#             in_channels=3,
#             learning_rate=config.learning.learning_rate,
#         ),
#         batch_size=config.learning.batch_size,
#         max_epochs=config.learning.epochs,
#         image_size=config.architecture.image_size,
#         width_meters=config.architecture.width_meters,
#         preload=config.learning.preload,
#         disk_cache_dir=config.learning.disk_cache_dir,
#     )
#     logger.info("Training model for %s epochs", config.learning.epochs)
#     return learner.learn(
#         inputs=train_data.drop(["is_delocalized"]),
#         outputs=train_data.select(["is_delocalized"]),
#     )

### New train function to handle imbalance
import torch


def train(
    train_data: pl.DataFrame,
    config: DictConfig | ListConfig,
) -> ImageBasedPyTorchModel:
    # Count samples by class
    class_counts = train_data["is_delocalized"].value_counts()
    counts_dict = dict(
        zip(
            class_counts["is_delocalized"], class_counts["count"], strict=False
        )
    )

    num_neg = counts_dict.get(False, 1)
    num_pos = counts_dict.get(True, 1)

    # Calculate pos_weight for BCEWithLogitsLoss
    # Formula: weight = num_neg / num_pos
    pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32)

    print(
        f"📊 Class imbalance: {num_neg} negative / {num_pos} positive → pos_weight = {pos_weight.item():.2f}"
    )

    # Initialize learner with weighted CNN
    learner = ImageBasedLightningLearner(
        module=CNN(
            image_size=config.architecture.image_size,
            in_channels=3,
            learning_rate=config.learning.learning_rate,
            pos_weight=pos_weight,  # 👈 Pass weight here
        ),
        batch_size=config.learning.batch_size,
        max_epochs=config.learning.epochs,
        image_size=config.architecture.image_size,
        width_meters=config.architecture.width_meters,
        preload=config.learning.preload,
        disk_cache_dir=config.learning.disk_cache_dir,
    )

    return learner.learn(
        inputs=train_data.drop(["is_delocalized"]),
        outputs=train_data.select(["is_delocalized"]),
    )


def evaluate(model: ImageBasedPyTorchModel, test_data: pl.DataFrame) -> Report:
    metrics = [
        Accuracy(),
        ClassificationReport(),
        FBetaScore(beta=0.5),  # 1.0
        PrecisionScore(),
        Recall(),
        ConfusionMatrix(normalize=True),
    ]
    logger.info("Evaluating model on test data")
    return evaluate_offline(
        model,
        DataFrame(test_data),
        inputs=["/map", "/scan", "/particle_cloud", "/amcl_pose"],
        outputs=["is_delocalized"],
        metrics=metrics,
    )
