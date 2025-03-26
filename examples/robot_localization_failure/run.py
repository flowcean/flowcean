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

import numpy as np
import polars as pl
from custom_transforms.localization_status import LocalizationStatus
from custom_transforms.particle_cloud_statistics import ParticleCloudStatistics
from scipy.special import kl_div

import flowcean.cli
from flowcean.polars.transforms.match_sampling_rate import MatchSamplingRate
from flowcean.polars.transforms.select import Select
from flowcean.ros.rosbag import RosbagLoader

USE_ROSBAG = False
WS = Path(__file__).resolve().parent
CACHE_FILE = WS / "cached_ros_data.parquet"
ROS_BAG_PATH = WS / "rec_20241021_152106"


def load_or_cache_ros_data(*, force_refresh: bool = False) -> pl.LazyFrame:
    """Load data from ROS bag or cache."""
    if CACHE_FILE.exists() and not force_refresh:
        print("Loading data from cache.")
        data = pl.read_parquet(CACHE_FILE).lazy()
        if data.collect().height > 0:
            return data
        print("Cache invalid; reloading from ROS bag.")

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
    print("Caching data to Parquet.")
    collected_data = data.collect()
    collected_data.write_parquet(CACHE_FILE, compression="snappy")
    print(f"Cache created/updated at {CACHE_FILE}")
    return data


def compute_kl_divergence(statuses, cog_values, bin_size=0.01):
    """Compute the KL divergence between the cog_max_distance distributions
    for localized (status 0) and delocalized (status 1) samples using a fixed bin size.

    Args:
        statuses: List of status values (0 or 1).
        cog_values: List of cog_max_distance values (floats).
        bin_size: The fixed size of each bin (for meters, 0.01 m for 1 cm).

    Returns:
        A tuple containing:
            - The KL divergence value.
            - The bins used.
            - Histogram counts for localized samples.
            - Histogram counts for delocalized samples.
    """
    # Separate values based on status
    cog_values_localized = [
        c for s, c in zip(statuses, cog_values, strict=False) if s == 0
    ]
    cog_values_delocalized = [
        c for s, c in zip(statuses, cog_values, strict=False) if s == 1
    ]

    # Determine overall range
    min_val = min(cog_values)
    max_val = max(cog_values)

    # Compute bin edges using the fixed bin size (e.g., 0.01 m for 1 cm)
    bins = np.arange(min_val, max_val + bin_size, bin_size)

    # Build histograms using the same bins for both groups
    counts_localized, _ = np.histogram(cog_values_localized, bins=bins)
    counts_delocalized, _ = np.histogram(cog_values_delocalized, bins=bins)

    # Normalize to get probability distributions
    p_localized = counts_localized.astype(float) / counts_localized.sum()
    q_delocalized = counts_delocalized.astype(float) / counts_delocalized.sum()

    # Add a small epsilon to avoid division-by-zero issues
    epsilon = 1e-10
    p_localized += epsilon
    q_delocalized += epsilon
    p_localized /= p_localized.sum()
    q_delocalized /= q_delocalized.sum()

    # Compute KL divergence element-wise and sum
    kl_elements = kl_div(p_localized, q_delocalized)
    kl_divergence = kl_elements.sum()

    return kl_divergence, bins, counts_localized, counts_delocalized


def main():
    flowcean.cli.initialize_logging()
    data = load_or_cache_ros_data(force_refresh=USE_ROSBAG)

    # Transformation pipeline
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
        | Select(["isDelocalized", "cog_max_distance"])
        | MatchSamplingRate(
            reference_feature_name="isDelocalized",
            feature_interpolation_map={"cog_max_distance": "linear"},
        )
    )

    transformed_data = transform(data)
    row = transformed_data.collect()[0]

    # Extract the lists from the single row
    is_delocalized_list = row["isDelocalized"]
    cog_max_distance_list = row["cog_max_distance"]

    # Inspect structure (optional)
    print("cog_max_distance_list[0][0]:", cog_max_distance_list[0][0])
    print("cog_max_distance_list[0][1]:", cog_max_distance_list[0][1])

    # Extract statuses and cog_max_distance values
    # For isDelocalized, each element is a dict: {'time': ..., 'value': {'data': <0 or 1>}}
    statuses_extracted = [
        item["value"]["data"] for item in is_delocalized_list[0]
    ]
    # For cog_max_distance, each element is a dict: {'time': ..., 'value': <float>}
    cog_values_extracted = [item["value"] for item in cog_max_distance_list[0]]

    print("Extracted statuses (first 5):", statuses_extracted[:5])
    print(
        "Extracted cog_max_distance values (first 5):",
        cog_values_extracted[:5],
    )
    print("Number of localized samples (0):", statuses_extracted.count(0))
    print("Number of delocalized samples (1):", statuses_extracted.count(1))

    # Compute KL divergence using fixed bin size (1 cm for meters)
    kl_divergence, bins, counts_loc, counts_deloc = compute_kl_divergence(
        statuses_extracted,
        cog_values_extracted,
        bin_size=0.01,
    )


if __name__ == "__main__":
    main()
