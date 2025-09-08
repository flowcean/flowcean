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
from omegaconf import DictConfig, ListConfig

from flowcean.core import Lambda, Report, Transform, evaluate_offline
from flowcean.polars import DataFrame, Drop, ZeroOrderHold
from flowcean.ros import load_rosbag
from flowcean.sklearn import (
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


def load_and_process_rosbag(
    path: str | PathLike,
    config: DictConfig | ListConfig,
) -> pl.DataFrame:
    cache_path = Path(path).with_suffix(".processed.parquet")
    if cache_path.exists():
        logger.info(
            "Loading already processed rosbag from cache: %s",
            cache_path,
        )
        return pl.read_parquet(cache_path)

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

    return transformed_data


def explode_and_collect_samples(data: pl.DataFrame) -> pl.DataFrame:
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
    runs_train = [
        load_and_process_rosbag(
            path=path,
            config=config,
        )
        for path in config.rosbag.training_paths
    ]
    logger.info("Combining training data")
    samples_train = explode_and_collect_samples(
        pl.concat(runs_train, how="vertical"),
    )

    logger.info("Collecting evaluation data")
    runs_eval = [
        load_and_process_rosbag(
            path=path,
            config=config,
        )
        for path in config.rosbag.evaluation_paths
    ]
    logger.info("Combining evaluation data")
    samples_eval = explode_and_collect_samples(
        pl.concat(runs_eval, how="vertical"),
    )
    return (samples_train, samples_eval)


def train(
    train_data: pl.DataFrame,
    config: DictConfig | ListConfig,
) -> ImageBasedPyTorchModel:
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
    logger.info("Training model for %s epochs", config.learning.epochs)
    return learner.learn(
        inputs=train_data.drop(["is_delocalized"]),
        outputs=train_data.select(["is_delocalized"]),
    )


def evaluate(model: ImageBasedPyTorchModel, test_data: pl.DataFrame) -> Report:
    metrics = [
        Accuracy(),
        ClassificationReport(),
        FBetaScore(beta=1.0),
        PrecisionScore(),
        Recall(),
    ]
    logger.info("Evaluating model on test data")
    return evaluate_offline(
        model,
        DataFrame(test_data),
        inputs=["/map", "/scan", "/particle_cloud", "/amcl_pose"],
        outputs=["is_delocalized"],
        metrics=metrics,
    )
