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

from flowcean.sklearn.ada_boost_classifier import AdaptiveBoostingClassifier
import numpy as np
import polars as pl
from custom_transforms.localization_status import LocalizationStatus
from custom_transforms.particle_cloud_image import ParticleCloudImage

import flowcean.cli
from flowcean.core.strategies.offline import evaluate_offline, learn_offline
from flowcean.polars.environments.dataframe import DataFrame
from flowcean.polars.environments.train_test_split import TrainTestSplit
from flowcean.polars.transforms.explode import Explode
from flowcean.polars.transforms.match_sampling_rate import MatchSamplingRate
from flowcean.polars.transforms.select import Select
from flowcean.ros.rosbag import RosbagLoader
from flowcean.sklearn.metrics.classification import Accuracy
from flowcean.torch import ConvolutionalNeuralNetwork, LightningLearner

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


def main() -> None:
    flowcean.cli.initialize_logging()

    # Load data with caching (set force_refresh=True to always reload)
    data = load_or_cache_ros_data(force_refresh=USE_ROSBAG)

    pixel_size = 100
    transform = (
        ParticleCloudImage(
            particle_cloud_feature_name="/particle_cloud",
            save_images=False,
            cutting_area=15.0,
            image_pixel_size=pixel_size,
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
        | Select(
            [
                "/particle_cloud_image",
                "/position_error",
                "/heading_error",
                "isDelocalized",
            ],
        )
        | MatchSamplingRate(
            reference_feature_name="/particle_cloud_image",
            feature_interpolation_map={
                "/position_error": "linear",
                "/heading_error": "linear",
                "isDelocalized": "nearest",
            },
        )
        | Explode(
            features=[
                "/particle_cloud_image",
                "/position_error",
                "/heading_error",
                "isDelocalized",
            ],
        )
    )
    transformed_data = transform(data)

    print(f"transformed data: {transformed_data.collect()}")
    # unnest data
    collected_transformed_data = transformed_data.collect()
    # loop over all columns and unnest them
    for column in collected_transformed_data.columns:
        collected_transformed_data = collected_transformed_data.unnest(
            column,
        ).rename({"time": column + "_time", "value": column + "_value"})
    # convert dict to value for isDelocalized_value
    collected_transformed_data = collected_transformed_data.unnest(
        "isDelocalized_value",
    ).rename(
        {"data": "isDelocalized_value"},
    )
    print(collected_transformed_data)
    print(collected_transformed_data.shape)
    data_environment = DataFrame(data=collected_transformed_data)
    train, test = TrainTestSplit(ratio=0.8, shuffle=True).split(
        data_environment,
    )
    # Define inputs and outputs with task types
    inputs = ["/particle_cloud_image_value"]
    outputs = ["isDelocalized_value"]
    learners = [
        # CNN (simple)
        LightningLearner(
            module=ConvolutionalNeuralNetwork(
                learning_rate=1e-3,
                conv_configs=[
                    (1, 32, 3),
                ],  # 1 conv layer: 1 input, 32 output channels, 3x3 kernel
                fully_connected_layer_sizes=[
                    128,
                    1,
                ],  # 2 fully connected layers: 128 units, then 1 output
                output_size=1,
                input_shape=(1, pixel_size, pixel_size),
            ),
            max_epochs=4,
            batch_size=4,
            num_workers=15,
            accelerator="auto",
        ),
        # CNN (mid-complex)
        LightningLearner(
            module=ConvolutionalNeuralNetwork(
                learning_rate=1e-3,
                conv_configs=[
                    (1, 32, 3),  # (input channel, output channel, kernel size)
                    (32, 64, 3),
                ],
                fully_connected_layer_sizes=[
                    256,
                    128,
                    64,
                    32,  # 4 fully connected layers
                ],
                output_size=1,  # Single output for binary classification
                input_shape=(1, pixel_size, pixel_size),  # (1, 100, 100)
            ),
            max_epochs=4,
            batch_size=4,
            num_workers=15,
            accelerator="auto",
        ),
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
