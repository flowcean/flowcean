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
from custom_transforms.localization_status import LocalizationStatus
from custom_transforms.particle_cloud_image import ParticleCloudImage

import flowcean.cli
from flowcean.core.strategies.offline import evaluate_offline, learn_offline
from flowcean.polars.environments.dataframe import DataFrame
from flowcean.polars.environments.train_test_split import TrainTestSplit
from flowcean.polars.transforms.explode import Explode
from flowcean.polars.transforms.match_sampling_rate import MatchSamplingRate
from flowcean.polars.transforms.select import Select
from flowcean.polars.transforms.time_window import TimeWindow
from flowcean.ros.rosbag import RosbagLoader
from flowcean.sklearn import MeanAbsoluteError, MeanSquaredError
from flowcean.torch import ConvolutionalNeuralNetwork, LightningLearner

USE_CACHED_ROS_DATA = False
UPDATE_CACHE = False
WS = Path(__file__).resolve().parent
CACHE_FILE = WS / "cached_ros_data.parquet"
ROS_BAG_PATH = WS / "rec_20241021_152106"


def load_or_cache_ros_data(force_refresh: bool = False) -> pl.LazyFrame:
    """Load data from ROS bag or cache, with optional refresh.

    Args:
        force_refresh: If True, reload from ROS bag and overwrite cache.

    Returns:
        LazyFrame containing the ROS bag data.
    """
    # Check if cache exists and is valid
    cache_exists = CACHE_FILE.exists()
    if cache_exists and not force_refresh:
        try:
            # Load cached data
            print("Loading data from cache.")
            data = pl.read_parquet(CACHE_FILE).lazy()
            # Optional: Validate cache (e.g., check metadata or row count)
            if data.collect().height > 0:
                return data
            print("Cache invalid; reloading from ROS bag.")
        except Exception as e:
            print(f"Cache read failed ({e}); reloading from ROS bag.")

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
    data = load_or_cache_ros_data(force_refresh=False)

    pixel_size = 300
    transform = (
        TimeWindow(  # get the first two minutes of data
            time_start=1729516868012553090,
            time_end=1729516988012553090,
        )
        | ParticleCloudImage(
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

    data_environment = DataFrame(data=collected_transformed_data)
    train, test = TrainTestSplit(ratio=0.8, shuffle=True).split(
        data_environment,
    )
    inputs = ["/particle_cloud_image_value"]
    outputs = ["isDelocalized_value"]
    learners = [
        # CNN (simple)
        LightningLearner(
            module=ConvolutionalNeuralNetwork(
                learning_rate=1e-3,
                conv_configs=[
                    (1, 32, 3),
                ],  # 1 conv layer: 1 input channel, 32 output channels, 3x3 kernel
                fully_connected_layer_sizes=[
                    128,
                    1,
                ],  # 2 fully connected layers: 128 units, then 1 output
                output_size=1,  # Binary classification (isDelocalized)
            ),
            max_epochs=10,
        ),
        # CNN (mid-complex)
        LightningLearner(
            module=ConvolutionalNeuralNetwork(
                learning_rate=1e-3,
                conv_configs=[(1, 32, 3)]
                + [(32, 32, 3)]
                * 20,  # 21 conv layers: 1->32, then 20 at 32->32
                fully_connected_layer_sizes=[128] * 10
                + [1],  # 11 fully connected layers
                output_size=1,
            ),
            max_epochs=10,
        ),
        # CNN (complex)
        LightningLearner(
            module=ConvolutionalNeuralNetwork(
                learning_rate=1e-3,
                conv_configs=[(1, 32, 3)] * 21,
                fully_connected_layer_sizes=[128] * 11,
                output_size=1,
            ),
            max_epochs=10,
        ),
        # # LSTM (simple)
        # LightningLearner(
        #     module=LongShortTermMemoryNetwork(
        #         learning_rate=1e-3,
        #         input_size=300 * 300,
        #         output_size=1,
        #         hidden_sizes=[128],
        #         fully_connected_layer_sizes=[64, 32],
        #     ),
        #     max_epochs=10,
        # ),
        # # LSTM (mid-complex)
        # LightningLearner(
        #     module=LongShortTermMemoryNetwork(
        #         learning_rate=1e-3,
        #         input_size=300 * 300,
        #         output_size=1,
        #         hidden_sizes=[128, 64],
        #         fully_connected_layer_sizes=[64, 32, 16],
        #     ),
        #     max_epochs=10,
        # ),
        # # LSTM (complex)
        # LightningLearner(
        #     module=LongShortTermMemoryNetwork(
        #         learning_rate=1e-3,
        #         input_size=300 * 300,
        #         output_size=1,
        #         hidden_sizes=[128, 64, 32],
        #         fully_connected_layer_sizes=[64, 32, 16, 8],
        #     ),
        #     max_epochs=10,
        # ),
        # # LRCN (simple)
        # LightningLearner(
        #     module=LongTermRecurrentConvolutionalNetwork(
        #         learning_rate=1e-3,
        #         conv_configs=[(1, 32, 3), (32, 64, 3)],
        #         lstm_hidden_sizes=[128, 64],
        #         fully_connected_layer_sizes=[64, 32],
        #         output_size=1,
        #     ),
        #     max_epochs=10,
        # ),
        # # LRCN (mid-complex)
        # LightningLearner(
        #     module=LongTermRecurrentConvolutionalNetwork(
        #         learning_rate=1e-3,
        #         conv_configs=[(1, 32, 3), (32, 64, 3), (64, 128, 3)],
        #         lstm_hidden_sizes=[128, 64],
        #         fully_connected_layer_sizes=[128, 64, 32, 16, 8],
        #         output_size=1,
        #     ),
        #     max_epochs=10,
        # ),
        # # LRCN (complex)
        # LightningLearner(
        #     module=LongTermRecurrentConvolutionalNetwork(
        #         learning_rate=1e-3,
        #         conv_configs=[(1, 32, 3)] * 5,
        #         lstm_hidden_sizes=[128, 64],
        #         fully_connected_layer_sizes=[128] * 7,
        #         output_size=1,
        #     ),
        #     max_epochs=10,
        # ),
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
            [MeanAbsoluteError(), MeanSquaredError()],
        )
        print(report)


if __name__ == "__main__":
    main()
