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
from custom_transforms.localization_status import LocalizationStatus
from custom_transforms.map_image import MapImage
from custom_transforms.particle_cloud_statistics import ParticleCloudStatistics
from custom_transforms.scan_image import ScanImage
from custom_transforms.scan_map import ScanMap

import flowcean.cli
from flowcean.core.strategies.offline import evaluate_offline, learn_offline
from flowcean.polars.environments.dataframe import DataFrame
from flowcean.polars.environments.train_test_split import TrainTestSplit
from flowcean.polars.transforms.match_sampling_rate import MatchSamplingRate
from flowcean.polars.transforms.time_window import TimeWindow
from flowcean.ros.rosbag import RosbagLoader

USE_ROSBAG = False
WS = Path(__file__).resolve().parent
CACHE_FILE = WS / "cached_ros_data.parquet"
ROS_BAG_PATH = WS / "rec_20241021_152106"
IMAGE_PIXEL_SIZE = 200
CROP_REGION_SIZE = 20.0


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


def main() -> None:
    flowcean.cli.initialize_logging()

    # Load data with caching (set force_refresh=True to always reload)
    data = load_or_cache_ros_data(force_refresh=USE_ROSBAG)

    transform = (
        TimeWindow(  # full data set exceeds memory
            time_start=1729516868012553090,
            time_end=1729516968012553090,
        )
        # timestamps need to be aligned before applying LocalizationStatus
        | MatchSamplingRate(
            reference_feature_name="/heading_error",
            feature_interpolation_map={"/position_error": "linear"},
        )
        | LocalizationStatus(
            position_error_feature_name="/position_error",
            heading_error_feature_name="/heading_error",
        )
        | ParticleCloudStatistics()
        | ScanMap()  # calculates the scan_points which are used for ScanImage
        | MatchSamplingRate(
            reference_feature_name="/particle_cloud",
            feature_interpolation_map={
                "/map": "nearest",
                "scan_points_sensor": "nearest",
            },
        )
        | ScanImage(
            crop_region_size=CROP_REGION_SIZE,
            image_pixel_size=IMAGE_PIXEL_SIZE,
            save_images=True,
        )
        | MapImage(
            crop_region_size=CROP_REGION_SIZE,
            image_pixel_size=IMAGE_PIXEL_SIZE,
            save_images=True,
        )
        # | Drop(
        #     features=[
        #         "/particle_cloud",
        #         "/map",
        #         "/scan",
        #         "/delocalizations",
        #         "/momo/pose",
        #         "/amcl_pose",
        #         "/position_error",
        #         "/heading_error",
        #         "scan_points",
        #     ],
        # )
        # | MatchSamplingRate(  # for all features
        #     reference_feature_name="point_distance",
        # )
        # | Explode()  # explode all columns
    )

    transformed_data = transform(data)

    print(f"transformed data: {transformed_data.collect()}")
    collected_transformed_data = transformed_data.collect()
    return
    # loop over all columns and unnest them
    for column in collected_transformed_data.columns:
        collected_transformed_data = collected_transformed_data.unnest(
            column,
        ).rename({"time": column + "_time", "value": column + "_value"})
        # drop time columns
        collected_transformed_data = collected_transformed_data.drop(
            column + "_time",
        )
    # convert dict to value for isDelocalized_value
    collected_transformed_data = collected_transformed_data.unnest(
        "isDelocalized_value",
    ).rename(
        {"data": "isDelocalized_value"},
    )
    print(f"collected transformed data: {collected_transformed_data}")
    data_environment = DataFrame(data=collected_transformed_data)
    train, test = TrainTestSplit(ratio=0.8, shuffle=True).split(
        data_environment,
    )
    # inputs are all features except "isDelocalized_value"
    inputs = collected_transformed_data.columns
    print(f"inputs: {inputs}")
    inputs.remove("isDelocalized_value")
    outputs = ["isDelocalized_value"]
    learners = [
        AdaBoost(),
    ]
    for learner in learners:
        t_start = datetime.now(tz=timezone.utc)
        model = learn_offline(
            train,
            learner,
            inputs,
            outputs,
        )
        delta_t = datetime.now(tz=timezone.utc) - t_start
        print(f"Learning took {np.round(delta_t.microseconds / 1000, 1)} ms")

        report = evaluate_offline(
            model,
            test,
            inputs,
            outputs,
            [Accuracy()],
        )
        print(report)


if __name__ == "__main__":
    main()
