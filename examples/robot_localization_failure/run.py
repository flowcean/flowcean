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

# Constants
USE_ROSBAG = False
WS = Path(__file__).resolve().parent
CACHE_FILE = WS / "cached_ros_data.parquet"
ROS_BAG_PATH = WS / "rec_20241021_152106"


def load_or_cache_ros_data(*, force_refresh: bool = False) -> pl.LazyFrame:
    """Load data from ROS bag or cache, with optional refresh.

    Args:
        force_refresh: If True, reload from ROS bag and overwrite cache.

    Returns:
        A LazyFrame containing the ROS bag data.
    """
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


def extract_numeric_value(x):
    """Extract the numeric value from a dict-like structure.

    Args:
        x: A dict or a numeric value.

    Returns:
        A float representing the numeric value.
    """
    if isinstance(x, dict):
        for v in x.values():
            if isinstance(v, float):
                return v
        for v in x.values():
            if isinstance(v, (int, float)):
                return float(v)
        raise ValueError("No numeric value found in dict.")
    return x


def extract_status(x):
    """Extract the status (0 or 1) from a dict-like or iterable structure.

    Args:
        x: A dict, set, or list.

    Returns:
        An integer (0 or 1).
    """
    if isinstance(x, (set, list)):
        return list(x)[0]
    if isinstance(x, dict):
        for v in x.values():
            if v in (0, 1):
                return v
        for v in x.values():
            if isinstance(v, int):
                return v
        raise ValueError("No valid status found in dict.")
    return x


def compute_kl_divergence(statuses, cog_values, num_bins=25):
    """Compute the KL divergence between the distribution of cog_max_distance
    for localized (status 0) and delocalized (status 1) samples.

    Args:
        statuses: A list of status values (0 for localized, 1 for delocalized).
        cog_values: A list of cog_max_distance values.
        num_bins: Number of bins to use for the histogram.

    Returns:
        A tuple containing:
            - The KL divergence value.
            - The bins used.
            - Histogram counts for localized samples.
            - Histogram counts for delocalized samples.
    """
    # Separate cog_max_distance values by status
    cog_values_localized = [
        c for s, c in zip(statuses, cog_values, strict=False) if s == 0
    ]
    cog_values_delocalized = [
        c for s, c in zip(statuses, cog_values, strict=False) if s == 1
    ]

    print(
        "Localized cog_max_distance: min =",
        min(cog_values_localized),
        "max =",
        max(cog_values_localized),
    )
    print(
        "Delocalized cog_max_distance: min =",
        min(cog_values_delocalized),
        "max =",
        max(cog_values_delocalized),
    )

    # Determine overall range for binning
    all_values = cog_values
    min_val = min(all_values)
    max_val = max(all_values)
    bins = np.linspace(min_val, max_val, num_bins + 1)

    # Build histograms for each group using the same bins
    counts_localized, _ = np.histogram(cog_values_localized, bins=bins)
    counts_delocalized, _ = np.histogram(cog_values_delocalized, bins=bins)

    print("Histogram counts for localized:", counts_localized)
    print("Histogram counts for delocalized:", counts_delocalized)

    # Normalize counts to get probability distributions
    p_localized = counts_localized.astype(float) / counts_localized.sum()
    q_delocalized = counts_delocalized.astype(float) / counts_delocalized.sum()

    # Add epsilon to avoid zeros
    epsilon = 1e-10
    p_localized += epsilon
    q_delocalized += epsilon

    # Renormalize after adding epsilon
    p_localized /= p_localized.sum()
    q_delocalized /= q_delocalized.sum()

    # Compute KL divergence element-wise and sum
    kl_elements = kl_div(p_localized, q_delocalized)
    kl_divergence = kl_elements.sum()

    return kl_divergence, bins, counts_localized, counts_delocalized


def main():
    flowcean.cli.initialize_logging()
    data = load_or_cache_ros_data(force_refresh=USE_ROSBAG)

    # Define transformation pipeline
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

    # Optional: inspect structure
    print("cog_max_distance_list[0][0]:", cog_max_distance_list[0][0])
    print("cog_max_distance_list[0][1]:", cog_max_distance_list[0][1])

    # Extract statuses and cog_max_distance values
    statuses_extracted = [
        item["value"]["data"] for item in is_delocalized_list[0]
    ]
    cog_values_extracted = [item["value"] for item in cog_max_distance_list[0]]

    print("Extracted statuses (first 5):", statuses_extracted[:5])
    print(
        "Extracted cog_max_distance values (first 5):",
        cog_values_extracted[:5],
    )
    print("Number of localized samples (0):", statuses_extracted.count(0))
    print("Number of delocalized samples (1):", statuses_extracted.count(1))

    # Compute and print KL divergence
    kl_divergence, bins, counts_loc, counts_deloc = compute_kl_divergence(
        statuses_extracted,
        cog_values_extracted,
        num_bins=25,
    )
    print("KL divergence (Localized || Delocalized):", kl_divergence)


if __name__ == "__main__":
    main()
