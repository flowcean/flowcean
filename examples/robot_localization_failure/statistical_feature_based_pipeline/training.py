import logging
from os import PathLike
from pathlib import Path

import polars as pl
from custom_transforms.localization_status import LocalizationStatus
from custom_transforms.particle_cloud_statistics import ParticleCloudStatistics
from custom_transforms.scan_map_statistics import ScanMapStatistics
from omegaconf import DictConfig, ListConfig

from flowcean.core import Transform
from flowcean.polars import Drop
from flowcean.polars.transforms.explode_time_series import ExplodeTimeSeries
from flowcean.polars.transforms.resample_to_reference import (
    ResampleToReference,
)
from flowcean.ros import load_rosbag

logger = logging.getLogger(__name__)


def define_transforms(
    position_threshold: float,
    heading_threshold: float,
    occupancy_map: dict,
) -> Transform:
    return (
        ScanMapStatistics(
            occupancy_map=occupancy_map,
            scan_topic="/scan",
            sensor_pose_topic="/amcl_pose",
        )
        | ParticleCloudStatistics(
            particle_cloud_feature_name="/particle_cloud",
        )
        | Drop("/map", "/scan", "/particle_cloud")
        | ResampleToReference(reference="ray_inlier_percent")
        | LocalizationStatus(
            time_series="resampled",
            ground_truth="/momo/pose",
            estimation="/amcl_pose",
            position_threshold=position_threshold,
            heading_threshold=heading_threshold,
        )
        | ExplodeTimeSeries("resampled")
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
            "/particle_cloud": ["particles"],
        },
        message_paths=config.rosbag.message_paths,
    )
    transform = define_transforms(
        config.localization.position_threshold,
        config.localization.heading_threshold,
        occupancy_map=data.select("/map").collect()["/map"][0][0]["value"],
    )
    transformed_data: pl.DataFrame = transform.apply(data).collect(
        engine="streaming",
    )

    logger.info("Caching processed data to Parquet file: %s", cache_path)
    transformed_data.write_parquet(cache_path)
    transformed_data.write_csv(cache_path.with_suffix(".csv"))
    return transformed_data


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
    samples_train = pl.concat(runs_train, how="vertical")

    logger.info("Collecting evaluation data")
    runs_eval = [
        load_and_process_rosbag(
            path=path,
            config=config,
        )
        for path in config.rosbag.evaluation_paths
    ]
    logger.info("Combining evaluation data")
    samples_eval = pl.concat(runs_eval, how="vertical")
    return (samples_train, samples_eval)
