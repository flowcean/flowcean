import importlib
from collections import defaultdict
from typing import Any, cast

import numpy as np
import polars as pl
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from rosbags.interfaces.typing import (
    BaseDesc,
    Basename,
    FieldDesc,
    NameDesc,
    Nodetype,
)
from std_msgs.msg import Float32MultiArray

Fielddefs = list[tuple[str, FieldDesc]]


class DataPreprocessor(Node):
    def __init__(self) -> None:
        super().__init__("preprocessing_node")
        self._init_config()
        self._init_buffers()
        self._init_subscribers()
        self._init_publisher()
        self._init_timer()

    def _init_config(self) -> None:
        self.topic_config = {
            "/amcl_pose": {
                "msg_type": "geometry_msgs.msg.PoseStamped",
                "fields": ["pose.pose.position.x"],
                "qos_profile": QoSPresetProfiles.SENSOR_DATA.value,
            },
            "/momo/pose": {
                "msg_type": "geometry_msgs.msg.PoseStamped",
                "fields": ["pose.pose.position.x"],
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
            # "/particle_cloud": {
            #     "msg_type": "nav2_msgs.msg.ParticleCloud",
            #     "fields": ["particles"],
            #     "qos_profile": QoSPresetProfiles.SENSOR_DATA.value,
            # },
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
        self.msg_definitions = {}

    def _init_buffers(self) -> None:
        self.data_buffer: dict[str, list[dict]] = defaultdict(list)

    def _init_subscribers(self) -> None:
        self.subscribers = {}
        for topic, config in self.topic_config.items():
            msg_cls = self._get_msg_class(config["msg_type"])
            self._store_msg_definition(topic, msg_cls)
            self.subscribers[topic] = self.create_subscription(
                msg_cls,
                topic,
                lambda msg, topic=topic: self.callback(msg, topic),
                config["qos_profile"],
            )

    def _get_msg_class(self, msg_type: str) -> Any:
        module, cls = msg_type.rsplit(".", 1)
        return getattr(importlib.import_module(module), cls)

    def _store_msg_definition(self, topic: str, msg_cls: Any) -> None:
        fields = [
            (name, self.get_field_descriptor(field))
            for name, field in msg_cls.get_fields_and_field_types().items()
        ]
        self.msg_definitions[topic] = ([], fields)

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
            self.publish_preprocessed_data,
        )

    def get_field_descriptor(self, field_type: str) -> FieldDesc:
        """Convert ROS2 type string to FieldDesc."""
        if "[" in field_type:
            base_type, size = field_type.rstrip("]").split("[")
            return (
                Nodetype.ARRAY,
                (
                    (Nodetype.BASE, (cast(Basename, base_type), 0)),
                    int(size),
                ),
            )
        if field_type in {"time", "duration"}:
            return (Nodetype.NAME, field_type)
        if "/" in field_type:
            return (Nodetype.NAME, field_type)
        return (Nodetype.BASE, (cast(Basename, field_type), 0))

    def get_polars_type(self, ros_type: str) -> pl.DataType:
        """Map ROS types to Polars types with proper type annotations."""
        type_map = {
            "float32": pl.Float32,
            "float64": pl.Float64,
            "int8": pl.Int8,
            "uint8": pl.UInt8,
            "int16": pl.Int16,
            "uint16": pl.UInt16,
            "int32": pl.Int32,
            "uint32": pl.UInt32,
            "int64": pl.Int64,
            "uint64": pl.UInt64,
            "bool": pl.Boolean,
            "string": pl.String,
            "time": pl.Datetime(time_unit="ns", time_zone="UTC"),
            "duration": pl.Int64,
        }
        return type_map.get(ros_type, pl.Object)

    def ros_msg_to_dict(self, msg: Any) -> dict:
        """Recursively convert ROS message to dict with proper typing."""
        result = {}
        for field in msg.__slots__:
            value = getattr(msg, field)
            if hasattr(value, "__slots__"):
                result[field] = self.ros_msg_to_dict(value)
            elif isinstance(value, np.ndarray):
                result[field] = value.tolist()
            elif isinstance(value, list):
                result[field] = [
                    self.ros_msg_to_dict(item)
                    if hasattr(item, "__slots__")
                    else item.tolist()
                    if hasattr(item, "tolist")
                    else item
                    for item in value
                ]
            else:
                result[field] = (
                    value.item() if hasattr(value, "item") else value
                )
        return result

    def desc_to_pltype(self, desc: FieldDesc) -> pl.DataType | type[pl.Object]:
        """Convert FieldDesc to Polars type."""
        if desc[0] == Nodetype.BASE:
            return self.get_polars_type(cast(BaseDesc, desc)[1][0])
        if desc[0] == Nodetype.NAME:
            typename = cast(NameDesc, desc)[1]
            return self.get_polars_type(typename.split("/")[-1])
        if desc[0] in (Nodetype.ARRAY, Nodetype.SEQUENCE):
            subtype = self.desc_to_pltype(
                cast(tuple[FieldDesc, int], desc[1])[0],
            )
            return pl.List(subtype)
        return pl.Object

    def _convert_ros_time(self, timestamp) -> int:
        return timestamp.sec * 1_000_000_000 + timestamp.nanosec

    def _create_entry(self, msg_dict: dict, config: dict) -> dict:
        return {
            "time": self._convert_ros_time(self.get_clock().now().to_msg()),
            "value": self._extract_fields(msg_dict, config["fields"]),
        }

    def _extract_fields(self, msg_dict: dict, fields: list[str]) -> dict:
        result = {}
        for field in fields:
            parts = field.split(".")
            value = msg_dict
            for part in parts:
                self.get_logger().debug(f"part: {part}")
                value = value.get("_" + part, {})
            self._set_nested_value(result, parts, value)
        self.get_logger().debug(f"Extracted fields: {result}")
        return result

    def _set_nested_value(
        self,
        data: dict,
        parts: list[str],
        value: Any,
    ) -> None:
        current = data
        self.get_logger().debug(f"data in _set_nested_value(): {data}")
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = value
        self.get_logger().debug(f"value in _set_nested_value(): {value}")

    def _handle_one_time_data(self, msg: Any, topic: str) -> None:
        """Process one-time data like map information."""
        msg_dict = self.ros_msg_to_dict(msg)
        value = {}
        for field in self.topic_config[topic]["fields"]:
            parts = field.split(".")
            current = msg_dict
            for part in parts:
                current = current.get(part, {})
            value[field] = current

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
            self.get_logger().debug(f"Keys in msg_dict: {msg_dict.keys()}")
            entry = self._create_entry(msg_dict, self.topic_config[topic])
            self._update_buffer(topic, entry)

        except (AttributeError, ValueError) as e:
            self.get_logger().error(f"Error processing {topic}: {e!s}")

    def _update_buffer(self, topic: str, entry: dict) -> None:
        self.data_buffer[topic].append(entry)
        if len(self.data_buffer[topic]) > self.max_buffer_size:
            self.data_buffer[topic].pop(0)

    def _prepare_dataset(self) -> pl.DataFrame:
        """Prepare dataset with each topic as a column containing time-value structs."""
        frames = []
        self.get_logger().debug(f"Data buffer: {self.data_buffer}")
        for topic, entries in self.data_buffer.items():
            if not entries:
                continue
            self.get_logger().debug(f"entries: {entries}")
            # Create list of structs for this topic
            struct_list = [
                {"time": entry["time"], "value": entry["value"]}
                for entry in entries
            ]

            # Create DataFrame with single column containing the struct list
            df = pl.DataFrame({topic: [struct_list]})
            frames.append(df)

        return (
            pl.concat(frames, how="horizontal") if frames else pl.DataFrame()
        )

    def publish_preprocessed_data(self) -> None:
        dataset = self._prepare_dataset()
        if dataset.is_empty():
            self.get_logger().warn("No data to publish")
            return

        # self.get_logger().debug(f"Dataset schema:\n{dataset.schema}")
        self.get_logger().info(f"Sample data:\n{dataset}")
        self.get_logger().debug(
            f"/position_error:\n{dataset['/position_error']}",
        )
        self.get_logger().debug(
            f"/heading_error:\n{dataset['/heading_error']}",
        )
        self.get_logger().debug(
            f"/particle_cloud:\n{dataset['/particle_cloud']}",
        )
        self.get_logger().debug(f"/scan:\n{dataset['/scan']}")


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
