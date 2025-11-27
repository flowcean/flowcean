from __future__ import annotations

import logging
from os import PathLike
from pathlib import Path

import polars as pl
import torch
from architectures.cnn_4_layers import CNN
from custom_learners.image_based_lightning_learner import (
    ImageBasedLightningLearner,
    ImageBasedPyTorchModel,
)
from custom_transforms.collapse import Collapse
from custom_transforms.detect_delocalizations import DetectDelocalizations
from custom_transforms.localization_status import LocalizationStatus
from omegaconf import DictConfig, ListConfig

from flowcean.core import Lambda, Report, Transform, evaluate_offline
from flowcean.polars import DataFrame, Drop, SliceTimeSeries, ZeroOrderHold
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
        "Lazily combining training data. This can take several minutes...",
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


def train(
    train_data: pl.DataFrame,
    config: DictConfig | ListConfig,
) -> ImageBasedPyTorchModel:
    # check if disk cache dir is empty, if no, remove its contents
    if config.learning.disk_cache_dir:
        disk_cache_path = Path(config.learning.disk_cache_dir)
        if disk_cache_path.exists() and any(disk_cache_path.iterdir()):
            logger.info(
                "Clearing existing disk cache directory: %s",
                disk_cache_path,
            )
            for item in disk_cache_path.iterdir():
                if item.is_dir():
                    for subitem in item.iterdir():
                        subitem.unlink()
                    item.rmdir()
                else:
                    item.unlink()
        disk_cache_path.mkdir(parents=True, exist_ok=True)
    true_counts = (
        train_data["is_delocalized"]
        .value_counts()
        .filter(pl.col("is_delocalized") == True)
        .select("count")
        .item()
    )
    false_counts = (
        train_data["is_delocalized"]
        .value_counts()
        .filter(pl.col("is_delocalized") == False)
        .select("count")
        .item()
    )
    ratio = false_counts / true_counts
    print(
        "Negative to Positive ratio:",
        ratio,
    )
    pos_weight = torch.tensor(
        [ratio],
        dtype=torch.float32,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    learner = ImageBasedLightningLearner(
        module=CNN(
            in_channels=3,
            learning_rate=config.learning.learning_rate,
            pos_weight=pos_weight,  # None,  # pos_weight,
        ),
        batch_size=config.learning.batch_size,
        max_epochs=config.learning.epochs,
        image_size=config.architecture.image_size,
        width_meters=config.architecture.width_meters,
        preload=config.learning.preload,
        disk_cache_dir=config.learning.disk_cache_dir,
    )
    logger.info("Training model for %s epochs", config.learning.epochs)
    return learner.learn(
        inputs=train_data.drop(["is_delocalized"]),
        outputs=train_data.select(["is_delocalized"]),
    )


def evaluate(
    model: ImageBasedPyTorchModel,
    test_data: pl.DataFrame,
    config: DictConfig | ListConfig,
) -> Report:
    from feature_images import DiskCaching, FeatureImagesData

    # Define base dataset
    base_dataset = FeatureImagesData(
        inputs=test_data.drop(["is_delocalized"]),
        outputs=test_data.select(["is_delocalized"]),
        image_size=config.architecture.image_size,
        width_meters=config.architecture.width_meters,
    )

    # Get cache directory from config
    eval_cache_dir = getattr(config.learning, "eval_cache_dir", None)
    clear_eval_cache = getattr(config.learning, "clear_eval_cache", False)

    dataset = base_dataset  # default

    if eval_cache_dir:
        eval_cache_path = Path(eval_cache_dir)
        eval_cache_path.mkdir(parents=True, exist_ok=True)

        # Optionally clear cache
        if clear_eval_cache and any(eval_cache_path.iterdir()):
            logger.info(
                "Clearing existing evaluation cache: %s",
                eval_cache_path,
            )
            for item in eval_cache_path.iterdir():
                if item.is_dir():
                    for subitem in item.iterdir():
                        subitem.unlink()
                    item.rmdir()
                else:
                    item.unlink()

        # Wrap in DiskCaching for re-use
        dataset = DiskCaching(base_dataset, eval_cache_path)

        # Warmup (precompute all .pt files)
        if config.learning.preload:
            if not any(eval_cache_path.iterdir()):
                logger.info("Precomputing evaluation cache...")
                dataset.warmup(show_progress=True)
            else:
                logger.info(
                    "Using existing cached tensors from %s",
                    eval_cache_path,
                )

    # Log dataset size
    logger.info("Evaluation dataset contains %d samples", len(dataset))

    # Define metrics
    metrics = [
        Accuracy(),
        ClassificationReport(),
        FBetaScore(beta=0.5),
        PrecisionScore(),
        Recall(),
    ]

    # Use cached dataset directly if available
    data_source = (
        dataset if isinstance(dataset, DiskCaching) else DataFrame(test_data)
    )

    logger.info("Evaluating model on test data")
    return evaluate_offline(
        model,
        DataFrame(test_data),
        inputs=["/map", "/scan", "/particle_cloud", "/amcl_pose"],
        outputs=["is_delocalized"],
        metrics=metrics,
    )
