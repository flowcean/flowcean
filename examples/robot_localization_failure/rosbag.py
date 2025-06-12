import logging
from collections.abc import Iterable
from os import PathLike
from pathlib import Path

import polars as pl

from flowcean.ros.rosbag import RosbagLoader

logger = logging.getLogger(__name__)


def load_or_cache_ros_data(
    path: PathLike,
    *,
    message_definitions: Iterable[PathLike] | None = None,
    cache_path: PathLike | None = None,
    ignore_cache: bool = False,
) -> pl.LazyFrame:
    """Load data from a ROS bag file and cache it as a Parquet file.

    Args:
        path: Path to the ROS bag file.
        message_definitions: Paths to additional ROS message definitions.
        cache_path: Path to the cache file. If None, defaults to the same
            directory as the ROS bag file with a .parquet extension.
        ignore_cache: If True, ignore the cache and reload data.

    Returns:
        LazyFrame containing the ROS bag data.
    """
    path = Path(path)

    cache_path = cache_path or path.with_suffix(".parquet")
    cache_path = Path(cache_path)

    if cache_path.exists() and not ignore_cache:
        logger.info("Loading data from cache...")
        return pl.scan_parquet(cache_path)

    logger.info("Loading data from ROS bag...")
    environment = RosbagLoader(
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
        message_paths=message_definitions,
    )
    data = environment.observe()

    logger.info("Caching data...")
    data.sink_parquet(cache_path)
    return data
