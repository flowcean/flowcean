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
from flowcean.core.environment.offline import OfflineEnvironment
from flowcean.core.strategies.offline import evaluate_offline, learn_offline
from flowcean.polars.environments.train_test_split import TrainTestSplit
from flowcean.polars.transforms.explode import Explode
from flowcean.polars.transforms.match_sampling_rate import MatchSamplingRate
from flowcean.polars.transforms.select import Select
from flowcean.polars.transforms.time_window import TimeWindow
from flowcean.ros.rosbag import RosbagLoader
from flowcean.sklearn import MeanAbsoluteError, MeanSquaredError
from flowcean.torch import (
    ConvolutionalNeuralNetwork,
    LightningLearner,
    LongShortTermMemoryNetwork,
    LongTermRecurrentConvolutionalNetwork,
)

USE_CACHED_ROS_DATA = False
UPDATE_CACHE = False
WS = Path(__file__).resolve().parent


def main() -> None:
    flowcean.cli.initialize_logging()

    if USE_CACHED_ROS_DATA:
        if Path(WS / "cached_ros_data.json").exists():
            print("Loading data from cache.")
            data = pl.read_json(WS / "cached_ros_data.json").lazy()
        else:
            msg = "Cached data not found."
            raise FileNotFoundError(msg)
    else:
        print("Loading data from rosbag.")
        environment = RosbagLoader(
            path=WS / "rec_20241021_152106",
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
    print(f"loaded data: {data.collect()}")
    pixel_size = 300
    transform = (
        TimeWindow(  # get the first two minutes of data
            features=[
                "/amcl_pose",
                "/momo/pose",
                "/scan",
                "/map",
                "/delocalizations",
                "/particle_cloud",
                "/position_error",
                "/heading_error",
            ],
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
    image_data = transform(data)
    print(f"image data: {image_data.collect()}")

    if UPDATE_CACHE:
        collected_data = data.collect()
        if Path(WS / "cached_ros_data.json").exists():
            collected_data.write_json()
            print("Cache updated")
        else:
            collected_data.write_json(file=WS / "cached_ros_data.json")
            print("Cache created")
    data_environment = OfflineEnvironment(data=image_data.collect())
    train, test = TrainTestSplit(ratio=0.8, shuffle=True).split(
        data_environment,
    )
    inputs = ["/particle_cloud_image"]
    outputs = ["/position_error", "/heading_error"]
    learners = [
        # CNN (simple)
        LightningLearner(
            module=ConvolutionalNeuralNetwork(
                learning_rate=1e-3,
                conv_configs=[(1, 32, 3)],
                fully_connected_layer_sizes=[128, 64],
                output_size=2,
            ),
            max_epochs=10,
        ),
        # CNN (mid-complex)
        LightningLearner(
            module=ConvolutionalNeuralNetwork(
                learning_rate=1e-3,
                conv_configs=[(1, 32, 3), (32, 64, 3)],
                fully_connected_layer_sizes=[128, 64, 32, 16],
                output_size=2,
            ),
            max_epochs=10,
        ),
        # CNN (complex)
        LightningLearner(
            module=ConvolutionalNeuralNetwork(
                learning_rate=1e-3,
                conv_configs=[(1, 32, 3)] * 21,
                fully_connected_layer_sizes=[128] * 11,
                output_size=2,
            ),
            max_epochs=10,
        ),
        # LSTM (simple)
        LightningLearner(
            module=LongShortTermMemoryNetwork(
                learning_rate=1e-3,
                input_size=300 * 300,
                output_size=2,
                hidden_sizes=[128],
                fully_connected_layer_sizes=[64, 32],
            ),
            max_epochs=10,
        ),
        # LSTM (mid-complex)
        LightningLearner(
            module=LongShortTermMemoryNetwork(
                learning_rate=1e-3,
                input_size=300 * 300,
                output_size=2,
                hidden_sizes=[128, 64],
                fully_connected_layer_sizes=[64, 32, 16],
            ),
            max_epochs=10,
        ),
        # LSTM (complex)
        LightningLearner(
            module=LongShortTermMemoryNetwork(
                learning_rate=1e-3,
                input_size=300 * 300,
                output_size=2,
                hidden_sizes=[128, 64, 32],
                fully_connected_layer_sizes=[64, 32, 16, 8],
            ),
            max_epochs=10,
        ),
        # LRCN (simple)
        LightningLearner(
            module=LongTermRecurrentConvolutionalNetwork(
                learning_rate=1e-3,
                conv_configs=[(1, 32, 3), (32, 64, 3)],
                lstm_hidden_sizes=[128, 64],
                fully_connected_layer_sizes=[64, 32],
                output_size=2,
            ),
            max_epochs=10,
        ),
        # LRCN (mid-complex)
        LightningLearner(
            module=LongTermRecurrentConvolutionalNetwork(
                learning_rate=1e-3,
                conv_configs=[(1, 32, 3), (32, 64, 3), (64, 128, 3)],
                lstm_hidden_sizes=[128, 64],
                fully_connected_layer_sizes=[128, 64, 32, 16, 8],
                output_size=2,
            ),
            max_epochs=10,
        ),
        # LRCN (complex)
        LightningLearner(
            module=LongTermRecurrentConvolutionalNetwork(
                learning_rate=1e-3,
                conv_configs=[(1, 32, 3)] * 5,
                lstm_hidden_sizes=[128, 64],
                fully_connected_layer_sizes=[128] * 7,
                output_size=2,
            ),
            max_epochs=10,
        ),
    ]
    return
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
