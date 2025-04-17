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

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
from architectures.cnn import CNN
from custom_transforms.images_to_tensor import ImagesToTensor
from custom_transforms.localization_status import LocalizationStatus
from custom_transforms.shift_timestamps import ShiftTimestamps
from custom_transforms.slice_time_series import SliceTimeSeries

import flowcean.cli
from flowcean.core.strategies.offline import evaluate_offline, learn_offline
from flowcean.polars.environments.dataframe import DataFrame
from flowcean.polars.environments.train_test_split import TrainTestSplit
from flowcean.polars.transforms.drop import Drop
from flowcean.ros.rosbag import RosbagLoader
from flowcean.sklearn.adaboost_classifier import AdaBoost
from flowcean.sklearn.metrics.classification import Accuracy
from flowcean.torch.lightning_learner import LightningLearner

SAVE_IMAGES = True
USE_ROSBAG = False
WS = Path(__file__).resolve().parent
CACHE_FILE = WS / "cached_ros_data.parquet"
ROSBAG_NAME = "rec_20241021_152106"
ROS_BAG_PATH = WS / ROSBAG_NAME
IMAGE_PIXEL_SIZE = 100
CROP_REGION_SIZE = 10.0


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
        data = pl.scan_parquet(CACHE_FILE)
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

    dataframes = []
    transformed_cache_exists = (WS / "transformed_data.parquet").exists()
    if transformed_cache_exists:
        print("Transformed data already exists. Loading from cache.")
        data = pl.scan_parquet(WS / "transformed_data.parquet")
    else:
        # Load data with caching (set force_refresh=True to always reload)
        data = load_or_cache_ros_data(force_refresh=USE_ROSBAG)
        print("Data loaded.")

        # extract map because it is only published once
        collected_data = data.collect()
        occupancy_map = collected_data[0, "/map"][1]["value"]

        transform = (
            Drop(features=["/map"])
            | LocalizationStatus(
                ground_truth_pose="/momo/pose",
                estimated_pose="/amcl_pose",
                position_threshold=0.4,
                heading_threshold=0.4,
            )
            | ShiftTimestamps(
                shift=1.0,
                feature="isDelocalized",
            )
            | SliceTimeSeries(
                counter_column="/delocalizations",
                deadzone=500_000_000,  # 0.5 seconds
            )
            | Drop(["position_error", "heading_error", "/delocalizations"])
            # | ZeroOrderHoldMatching(
            #     topics=[
            #         "/scan",
            #         "/particle_cloud",
            #         "/momo/pose",
            #         "/amcl_pose",
            #         "isDelocalized",
            #     ],
            # )
            # | Drop(features=["/delocalizations"])
            # | MapImage(
            #     occupancy_map=occupancy_map,
            #     crop_region_size=CROP_REGION_SIZE,
            #     image_pixel_size=IMAGE_PIXEL_SIZE,
            #     save_images=SAVE_IMAGES,
            # )
            # | ParticleCloudStatistics()
            # | ScanMapStatistics(occupancy_map=occupancy_map)
            # | ScanImage(
            #     crop_region_size=CROP_REGION_SIZE,
            #     image_pixel_size=IMAGE_PIXEL_SIZE,
            #     save_images=SAVE_IMAGES,
            # )
            # | ParticleCloudImage(
            #     crop_region_size=CROP_REGION_SIZE,
            #     image_pixel_size=IMAGE_PIXEL_SIZE,
            #     save_images=SAVE_IMAGES,
            # )
            # | Drop(
            #     features=[
            #         "/particle_cloud",
            #         "/scan",
            #         "/momo/pose",
            #         "/amcl_pose",
            #         "position_error",
            #         "heading_error",
            #         "scan_points",
            #         "scan_points_sensor",
            #     ],
            # )
        )

        transformed_data = transform(data)
        print("Data after transformations:")
        print(transformed_data.collect())
        # Use streaming to save memory
        output_file = WS / "transformed_data.parquet"
        transformed_data.sink_parquet(
            output_file,
            compression="snappy",
            row_group_size=1,
        )
        print(f"Transformed data streamed to {output_file}")

    return
    transformed_dataframes = []
    for row_df in dataframes:
        # Apply the transform to each slice
        transformed_slice = zoh_transform.apply(row_df.lazy())
        transformed_dataframes.append(transformed_slice)
    # Concatenate all transformed slices
    transformed_data = pl.concat(transformed_dataframes).lazy()
    print("Data after ZeroOrderHoldMatching:")
    print(transformed_data.collect())
    tensor_transform = ImagesToTensor(
        image_columns=[
            "/scan_image_value",
            "/map_image_value",
            "/particle_cloud_image_value",
        ],
        height=IMAGE_PIXEL_SIZE,
        width=IMAGE_PIXEL_SIZE,
    )
    transformed_data = tensor_transform.apply(transformed_data)
    print("Data after ImagesToTensor:")
    print(transformed_data.collect())

    transformed_data = (
        transformed_data.with_columns(
            pl.col("isDelocalized_value")
            .struct[0]
            .alias("isDelocalized_value_scalar"),
        )
        .drop("isDelocalized_value")
        .rename({"isDelocalized_value_scalar": "isDelocalized_value"})
    )
    collected_transformed_data = transformed_data.collect()
    data_environment = DataFrame(data=collected_transformed_data)
    train, test = TrainTestSplit(ratio=0.8, shuffle=True).split(
        data_environment,
    )
    cnn_inputs = [
        "image_tensor",
    ]
    outputs = ["isDelocalized_value"]
    # adaboost inputs are all features except outputs and tensors
    adaboost_inputs = collected_transformed_data.columns
    adaboost_inputs.remove("isDelocalized_value")
    adaboost_inputs.remove("/scan_image_value")
    adaboost_inputs.remove("/map_image_value")
    adaboost_inputs.remove("image_tensor")
    adaboost_inputs.remove("time")
    adaboost_inputs.remove("scan_points_sensor_value")
    adaboost_inputs.remove("slice_id")
    print(f"adaboost inputs: {adaboost_inputs}")
    print(f"cnn inputs: {cnn_inputs}")
    print(f"outputs: {outputs}")
    cnn_learners = [
        LightningLearner(
            module=CNN(
                image_size=IMAGE_PIXEL_SIZE,
                learning_rate=1e-3,
                in_channels=2,
            ),
            batch_size=4,
            max_epochs=5,
        ),
    ]
    adaboost_learners = [
        AdaBoost(),
    ]
    for learner in cnn_learners:
        t_start = datetime.now(tz=timezone.utc)
        model = learn_offline(
            environment=train,
            learner=learner,
            inputs=cnn_inputs,
            outputs=outputs,
        )
        delta_t = datetime.now(tz=timezone.utc) - t_start
        print(f"Learning took {np.round(delta_t.microseconds / 1000, 1)} ms")

        report = evaluate_offline(
            model,
            test,
            cnn_inputs,
            outputs,
            [Accuracy()],
        )
        print(report)

    for learner in adaboost_learners:
        t_start = datetime.now(tz=timezone.utc)
        model = learn_offline(
            train,
            learner,
            adaboost_inputs,
            outputs,
        )
        delta_t = datetime.now(tz=timezone.utc) - t_start
        print(f"Learning took {np.round(delta_t.microseconds / 1000, 1)} ms")

        report = evaluate_offline(
            model,
            test,
            adaboost_inputs,
            outputs,
            [Accuracy()],
        )
        print(report)


if __name__ == "__main__":
    main()
