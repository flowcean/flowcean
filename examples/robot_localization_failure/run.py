#!/usr/bin/env python
# /// script
# dependencies = [
#     "flowcean",
#     "matplotlib",
#     "opencv-python",
# ]
#
# [tool.uv.sources]
# flowcean = { path = "../../", editable = true }
# ///

from pathlib import Path

import polars as pl

from custom_transforms.kl_divergence import KLDivergence
from custom_transforms.localization_status import LocalizationStatus
from custom_transforms.particle_cloud_statistics import ParticleCloudStatistics

import flowcean.cli
from flowcean.polars.transforms.match_sampling_rate import MatchSamplingRate
from flowcean.polars.transforms.select import Select

from flowcean.ros.rosbag import RosbagLoader

USE_ROSBAG = False
WS = Path(__file__).resolve().parent
CACHE_FILE = WS / "cached_ros_data.parquet"
ROS_BAG_PATH = WS / "rec_20241021_152106"


def load_or_cache_ros_data(
    *,
    force_refresh: bool = False,
) -> pl.LazyFrame:
    """Load data from ROS bag or cache, with optional refresh.

    Args:
        force_refresh: If True, reload from ROS bag and overwrite cache.

    Returns:
        LazyFrame containing the ROS bag data.
    """
    # Check if cache exists and is valid
    cache_exists = CACHE_FILE.exists()

    if cache_exists and not force_refresh:
        # Load cached data
        print("Loading data from cache.")
        data = pl.read_parquet(CACHE_FILE).lazy()
        # Optional: Validate cache (e.g., check metadata or row count)
        if data.collect().height > 0:
            return data
        print("Cache invalid; reloading from ROS bag.")

    # Load from ROS bag
    print("Loading data from ROS bag.")
    environment = RosbagLoader(
        path=ROS_BAG_PATH,
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
            "/position_error": ["data"],
            "/heading_error": ["data"],
        },
        msgpaths=[
            str(WS / "ros_msgs/LaserScan.msg"),
            str(WS / "ros_msgs/nav2_msgs/msg/Particle.msg"),
            str(WS / "ros_msgs/nav2_msgs/msg/ParticleCloud.msg"),
        ],
    )
    data = environment.observe()

    # Cache the data
    print("Caching data to Parquet.")
    collected_data = data.collect()
    collected_data.write_parquet(CACHE_FILE, compression="snappy")
    print(f"Cache created/updated at {CACHE_FILE}")
    return data


def load_or_cache_ros_data(
    *,
    force_refresh: bool = False,
) -> pl.LazyFrame:
    """Load data from ROS bag or cache, with optional refresh.

    Args:
        force_refresh: If True, reload from ROS bag and overwrite cache.

    Returns:
        LazyFrame containing the ROS bag data.
    """
    # Check if cache exists and is valid
    cache_exists = CACHE_FILE.exists()

    if cache_exists and not force_refresh:
        # Load cached data
        print("Loading data from cache.")
        data = pl.read_parquet(CACHE_FILE).lazy()
        # Optional: Validate cache (e.g., check metadata or row count)
        if data.collect().height > 0:
            return data
        print("Cache invalid; reloading from ROS bag.")

    # Load from ROS bag
    print("Loading data from ROS bag.")
    environment = RosbagLoader(
        path=ROS_BAG_PATH,
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
            "/position_error": ["data"],
            "/heading_error": ["data"],
        },
        msgpaths=[
            str(WS / "ros_msgs/LaserScan.msg"),
            str(WS / "ros_msgs/nav2_msgs/msg/Particle.msg"),
            str(WS / "ros_msgs/nav2_msgs/msg/ParticleCloud.msg"),
        ],
    )
    data = environment.observe()

    # Cache the data
    print("Caching data to Parquet.")
    collected_data = data.collect()
    collected_data.write_parquet(CACHE_FILE, compression="snappy")
    print(f"Cache created/updated at {CACHE_FILE}")
    return data


def main()-> None:
    flowcean.cli.initialize_logging()
    data = load_or_cache_ros_data(force_refresh=USE_ROSBAG)

    transform = (
        Select(["/position_error", "/heading_error", "/particle_cloud"])
        | MatchSamplingRate(
            reference_feature_name="/heading_error",
            feature_interpolation_map={"/position_error": "linear"},
        )
        | LocalizationStatus(
            position_error_feature_name="/position_error",
            heading_error_feature_name="/heading_error",
            position_threshold=1.2,
            heading_threshold=1.2,
        )
        | ParticleCloudStatistics(
            particle_cloud_feature_name="/particle_cloud",
        )
        | Select(
            [
                "isDelocalized",
                "cog_max_distance",
                "cog_mean_dist",
                "cog_median",
                "cog_min_distance",
            ],
        )
        | MatchSamplingRate(
            reference_feature_name="isDelocalized",
            feature_interpolation_map={
                "cog_max_distance": "linear",
                "cog_mean_dist": "linear",
                "cog_median": "linear",
            },
        )
        | KLDivergence(
            target_column="isDelocalized",
            features={
                "cog_max_distance": 0.01,
                "cog_mean_dist": 0.01,
                "cog_median": 0.01,
            },
            top_n=2,
        )
     )

    transformed_data = transform(data)

    print(transformed_data.collect())


if __name__ == "__main__":
    main()
