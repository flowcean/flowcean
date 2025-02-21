import importlib
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import numpy as np
import polars as pl
import rclpy
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
            | ParticleCloudImage(
                save_images=False,
            )
            | ParticleCloudStatistics()
        )
        self.topic_config = {
            "/amcl_pose": {
                "msg_type": "geometry_msgs.msg.PoseWithCovarianceStamped",
                "fields": [
                    "pose.pose.position.x",
                    "pose.pose.position.y",
                    "pose.pose.orientation.x",
                    "pose.pose.orientation.y",
                    "pose.pose.orientation.z",
                    "pose.pose.orientation.w",
                ],
                "qos_profile": QoSPresetProfiles.SENSOR_DATA.value,
            },
            "/momo/pose": {
                "msg_type": "geometry_msgs.msg.PoseStamped",
                "fields": ["pose.position.x", "pose.position.y"],
                "qos_profile": QoSPresetProfiles.SENSOR_DATA.value,
            },
            "/scan": {
                "msg_type": "sensor_msgs.msg.LaserScan",
                "fields": [
                    "ranges",
                    "angle_min",
                    "angle_max",
                    "angle_increment",
                    "range_min",
                    "range_max",
                ],
                "qos_profile": QoSPresetProfiles.SENSOR_DATA.value,
            },
            "/map": {
                "msg_type": "nav_msgs.msg.OccupancyGrid",
                "fields": [
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
                "qos_profile": QoSPresetProfiles.SYSTEM_DEFAULT.value,
                "one_time_data": True,
            },
            "/particle_cloud": {
                "msg_type": "nav2_msgs.msg.ParticleCloud",
                "fields": ["particles"],
                "qos_profile": QoSPresetProfiles.SENSOR_DATA.value,
            },
            "/delocalizations": {
                "msg_type": "std_msgs.msg.Int16",
                "fields": ["data"],
                "qos_profile": QoSPresetProfiles.SENSOR_DATA.value,
            },
            "/position_error": {
                "msg_type": "std_msgs.msg.Float32",
                "fields": ["data"],
                "qos_profile": QoSPresetProfiles.SENSOR_DATA.value,
            },
            "/heading_error": {
                "msg_type": "std_msgs.msg.Float32",
                "fields": ["data"],
                "qos_profile": QoSPresetProfiles.SENSOR_DATA.value,
            },
        }

        self.subscribers = {}
        self.data_buffer: dict[str, list[dict]] = defaultdict(list)
        self.max_buffer_size = 50
        self.map_data = None

    def _init_subscribers(self) -> None:
        self.subscribers = {}
        for topic, config in self.topic_config.items():
            msg_cls = self._get_msg_class(config["msg_type"])
            self.subscribers[topic] = self.create_subscription(
                msg_cls,
                topic,
                lambda msg, topic=topic: self.callback(msg, topic),
                config["qos_profile"],
            )

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

    def _convert_ros_time(self, timestamp) -> int:
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
        """Recursively convert ROS message to dict with proper list handling."""
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
        """Prepare dataset with each topic as a column of time-value structs."""
        frames = []

        # Add regular topics
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
        print(dataset.schema)
        print(dataset)
        # Check if all topics are present
        if not all(topic in dataset.columns for topic in self.topic_config):
            self.get_logger().warn("Not all topics present in dataframe")
            return
        # print(f"dataset['m/ap']: {dataset['/map']}")
        # print(f"dataset['/map'].to_list(): {dataset['/map'].to_list()}")
        # print(f"dataset['/map'].to_list()[0]: {dataset['/map'].to_list()[0]}")
        # print(
        #     f"dataset['/map'].to_list()[0][1]: {dataset['/map'].to_list()[0][0]}",
        # )
        # print(
        #     f"dataset['/map'].to_list()[0][1]['value']: {dataset['/map'].to_list()[0][0]['value']}",
        # )

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
