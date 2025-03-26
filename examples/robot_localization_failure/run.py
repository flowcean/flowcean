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
from scipy.special import kl_div

import flowcean.cli
from flowcean.polars.transforms.match_sampling_rate import MatchSamplingRate
from flowcean.polars.transforms.select import Select
from flowcean.ros.rosbag import RosbagLoader

from custom_transforms.localization_status import LocalizationStatus
from custom_transforms.particle_cloud_statistics import ParticleCloudStatistics

USE_ROSBAG = False
WS = Path(__file__).resolve().parent
CACHE_FILE = WS / "cached_ros_data.parquet"
ROS_BAG_PATH = WS / "rec_20241021_152106"


def load_or_cache_ros_data(*, force_refresh: bool = False) -> pl.LazyFrame:
    """Load data from a ROS bag or cache."""
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


def compute_kl_divergence(statuses, feature_values, bin_size=0.01) -> float:
    """
    Compute the KL divergence for a given feature's values using a fixed bin size.
    """
    values_localized = [v for s, v in zip(statuses, feature_values, strict=False) if s == 0]
    values_delocalized = [v for s, v in zip(statuses, feature_values, strict=False) if s == 1]

    if not values_localized or not values_delocalized:
        return float('nan')

    min_val = min(feature_values)
    max_val = max(feature_values)
    bins = np.arange(min_val, max_val + bin_size, bin_size)
    if len(bins) < 2:
        return 0.0

    counts_localized, _ = np.histogram(values_localized, bins=bins)
    counts_delocalized, _ = np.histogram(values_delocalized, bins=bins)

    p_localized = counts_localized.astype(float) / counts_localized.sum()
    q_delocalized = counts_delocalized.astype(float) / counts_delocalized.sum()

    epsilon = 1e-10
    p_localized += epsilon
    q_delocalized += epsilon
    p_localized /= p_localized.sum()
    q_delocalized /= q_delocalized.sum()

    kl_elements = kl_div(p_localized, q_delocalized)
    return kl_elements.sum()


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
        | ParticleCloudStatistics(particle_cloud_feature_name="/particle_cloud")
        | Select(["isDelocalized", "cog_max_distance", "cog_mean_dist"])
        | MatchSamplingRate(
            reference_feature_name="isDelocalized",
            feature_interpolation_map={
                "cog_max_distance": "linear",
                "cog_mean_dist": "linear",
            },
        )
    )

    df_collected = transform(data).collect()
    row = df_collected[0]

    # Extract statuses from isDelocalized (each element is {'time': ..., 'value': {'data': 0 or 1}})
    is_delocalized_list = row["isDelocalized"]
    statuses_extracted = [item["value"]["data"] for item in is_delocalized_list[0]]

    # List of feature names for which to compute KL divergence.
    feature_names = ["cog_max_distance", "cog_mean_dist"]
    kl_dict = {}

    for feature in feature_names:
        feature_list = row[feature]
        feature_values = [item["value"] for item in feature_list[0]]
        kl_val = compute_kl_divergence(statuses_extracted, feature_values, bin_size=0.01)
        kl_dict[feature] = kl_val

    print("KL Divergence Dictionary:", kl_dict)

    # Create a literal column from the dictionary.
    # Specify the dtype explicitly so that the field names are preserved.
    kl_div_column_value = pl.lit(
        kl_dict,
        dtype=pl.Struct({
            "cog_max_distance": pl.Float64,
            "cog_mean_dist": pl.Float64,
        })
    )
    df_final = df_collected.with_columns(kl_div_column_value.alias("kl_divergence_features"))

    print(df_final)

    print("Field names in 'kl_divergence_features':", df_final.schema)


if __name__ == "__main__":
    main()
