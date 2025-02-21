#!/usr/bin/env python
# /// script
# dependencies = [
#     "flowcean",
#     "matplotlib",
#     "opencv-python",
#     "rclpy",
#     "numpy",
#     "polars",
#     "pyyaml",
#     "builtin_interfaces",
#     "std_msgs"
# ]
#
# [tool.uv.sources]
# flowcean = { path = "../../", editable = true }
# ///


import importlib
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import rclpy
import yaml
from builtin_interfaces.msg import Time
from custom_transforms.particle_cloud_image import ParticleCloudImage
from custom_transforms.particle_cloud_statistics import ParticleCloudStatistics
from custom_transforms.scan_map import ScanMap
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from std_msgs.msg import Float32MultiArray


class DataPreprocessor(Node):
    def __init__(self) -> None:
        super().__init__("preprocessing_node")
        self._init_config()
        self._init_subscribers()
        self._init_publisher()
        self._init_timer()

    def _init_config(self) -> None:
        self.transform = (
            ScanMap(plotting=False)
            | ParticleCloudImage(save_images=False)
            | ParticleCloudStatistics()
        )
        self.topic_config: dict[str, Any] = {}
        config_file = (
            Path.home()
            / "ros2_ws/src/flowcean/flowcean_ros/config/topic_info.yaml"
        )
        with Path.open(config_file) as f:
            loaded_config = yaml.safe_load(f)
            self.topic_config = (
                loaded_config if isinstance(loaded_config, dict) else {}
            )
        self.quality_of_service_mapping = {
            "SENSOR_DATA": QoSPresetProfiles.SENSOR_DATA.value,
            "PARAMETER_EVENTS": QoSPresetProfiles.PARAMETER_EVENTS.value,
            "SYSTEM_DEFAULT": QoSPresetProfiles.SYSTEM_DEFAULT.value,
        }
        self.subscribers = {}
        self.data_buffer: dict[str, list[dict]] = defaultdict(list)
        self.max_buffer_size = 50
        self.map_data = None

    def _init_subscribers(self) -> None:
        self.subscribers = {}
        if isinstance(self.topic_config, dict):
            for topic, config in self.topic_config.items():
                msg_cls = self._get_msg_class(config["msg_type"])
                self.subscribers[topic] = self.create_subscription(
                    msg_cls,
                    topic,
                    lambda msg, topic=topic: self.callback(msg, topic),
                    self.quality_of_service_mapping[config["qos_profile"]],
                )
        else:
            self.get_logger().error("No topics found in config")

    def _get_msg_class(self, msg_type: str) -> Any:
        module, cls = msg_type.rsplit(".", 1)
        return getattr(importlib.import_module(module), cls)

    def _init_publisher(self) -> None:
        self.preprocessed_publisher = self.create_publisher(
            Float32MultiArray,
            "/preprocessed_data",
            10,
        )

    def _init_timer(self) -> None:
        self.timer_period = 0.5
        self.timer = self.create_timer(
            self.timer_period,
            self.apply_transforms,
        )

    def _convert_ros_time(self, timestamp: Time) -> int:
        return timestamp.sec * 1_000_000_000 + timestamp.nanosec

    def _create_entry(self, msg_dict: dict, config: dict) -> dict:
        return {
            "time": self._convert_ros_time(self.get_clock().now().to_msg()),
            "value": self._extract_fields(msg_dict, config["fields"]),
        }

    def _extract_fields(self, msg_dict: dict, fields: list[str]) -> dict:
        """Extract nested fields using dotted notation."""
        result = {}
        for field in fields:
            parts = field.split(".")
            value = msg_dict
            for part in parts:
                value = value.get(part, {})
            # make sure this does not result in a pl.Object later
            result[field] = (
                list(value) if isinstance(value, Iterable) else value
            )
        return result

    def _handle_one_time_data(self, msg: Any, topic: str) -> None:
        msg_dict = self.ros_msg_to_dict(msg)
        value = {}
        for field in self.topic_config[topic]["fields"]:
            parts = field.split(".")
            current = msg_dict
            for part in parts:
                current = current.get(part, {})
            value[field] = (
                list(current) if isinstance(current, Iterable) else current
            )
        self.map_data = {
            "time": self.get_clock().now().to_msg(),
            "value": value,
        }

    def callback(self, msg: Any, topic: str) -> None:
        try:
            if self.topic_config[topic].get("one_time_data"):
                self._handle_one_time_data(msg, topic)
                return

            msg_dict = self.ros_msg_to_dict(msg)
            entry = self._create_entry(msg_dict, self.topic_config[topic])
            self._update_buffer(topic, entry)
        except (AttributeError, ValueError) as e:
            self.get_logger().error(f"Error processing {topic}: {e!s}")

    def _update_buffer(self, topic: str, entry: dict) -> None:
        self.data_buffer[topic].append(entry)
        if len(self.data_buffer[topic]) > self.max_buffer_size:
            self.data_buffer[topic].pop(0)

    def ros_msg_to_dict(self, msg: Any) -> dict:
        """Recursively convert ROS message to dict."""
        result = {}
        for field in msg.__slots__:
            key = field.lstrip("_")
            value = getattr(msg, field)
            if hasattr(value, "__slots__"):  # Nested message
                result[key] = self.ros_msg_to_dict(value)
            elif isinstance(value, np.ndarray):
                result[key] = value.tolist()
            elif isinstance(value, list):
                # Recursively convert list items if they are ROS messages
                result[key] = [
                    self.ros_msg_to_dict(item)
                    if hasattr(item, "__slots__")
                    else item.tolist()
                    if hasattr(item, "tolist")
                    else item
                    for item in value
                ]
            else:
                result[key] = value.item() if hasattr(value, "item") else value
        return result

    def _prepare_dataset(self) -> pl.DataFrame:
        """Prepare dataset with time series columns for each topic."""
        frames = []

        for topic, entries in self.data_buffer.items():
            if not entries:
                continue
            struct_list = [
                {"time": entry["time"], "value": entry["value"]}
                for entry in entries
            ]
            df = pl.DataFrame({topic: [struct_list]}, strict=False)
            frames.append(df)

        # Add map data -> should be generalized later for all one-time data
        if self.map_data:
            df_map = pl.DataFrame(
                {
                    "/map": [
                        [
                            {
                                "time": self._convert_ros_time(
                                    self.map_data["time"],
                                ),
                                "value": self.map_data["value"],
                            },
                        ],
                    ],
                },
                strict=False,
            )
            frames.append(df_map)

        return (
            pl.concat(frames, how="horizontal") if frames else pl.DataFrame()
        )

    def apply_transforms(self) -> None:
        dataset = self._prepare_dataset()
        if dataset.is_empty():
            self.get_logger().warn("No dataframe to process")
            return
        print(dataset)
        # Check if all topics are present
        if not all(topic in dataset.columns for topic in self.topic_config):
            self.get_logger().warn("Not all topics present in dataframe")
            return

        transformed_dataset = self.transform(dataset.lazy())
        print(f"transformed data: {transformed_dataset.collect()}")


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = DataPreprocessor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().debug("Shutting down")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
