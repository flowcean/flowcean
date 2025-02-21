from collections.abc import Generator
from typing import Any

import numpy as np
import pytest
import rclpy
from rclpy.clock import Clock
from rclpy.time import Time

from flowcean_ros.flowcean_ros.data_preprocessor import DataPreprocessor


# Mock classes for testing.
class DummyTime:
    def __init__(self, sec: int = 1, nanosec: int = 500) -> None:
        self.sec = sec
        self.nanosec = nanosec
        self.nanosec = nanosec


class DummyClock(Clock):
    def __init__(self) -> None:
        super().__init__()
        super().__init__()

    def now(self) -> Time:
        return Time(seconds=1, nanoseconds=500, clock_type=self.clock_type)


class DummyMsg:
    __slots__ = ["a", "b"]

    def __init__(self, a: Any, b: Any) -> None:
        self.a = a
        self.b = b
        self.b = b


class DummyOrigin:
    __slots__ = ["orientation", "position"]

    def __init__(self) -> None:
        self.position = DummyPosition()
        self.orientation = DummyOrientation()


class DummyPosition:
    __slots__ = ["x", "y", "z"]

    def __init__(self) -> None:
        self.x = 1.0
        self.y = 2.0
        self.z = 3.0


class DummyOrientation:
    __slots__ = ["w", "x", "y", "z"]

    def __init__(self) -> None:
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.w = 1.0


class DummyInfo:
    __slots__ = ["height", "origin", "resolution", "width"]

    def __init__(self) -> None:
        self.resolution = 0.5
        self.width = 10
        self.height = 10
        self.origin = DummyOrigin()


class DummyMapMsg:
    __slots__ = ["data", "info"]

    def __init__(self) -> None:
        self.data = [0, 1, 2]
        self.info = DummyInfo()


# Fixture to initialize a DataPreprocessor node for testing.
@pytest.fixture
def node() -> Generator[DataPreprocessor, None, None]:
    rclpy.init(args=[])
    dp_node = DataPreprocessor()
    dp_node.get_clock = lambda: DummyClock()
    yield dp_node
    dp_node.destroy_node()
    rclpy.shutdown()


def test_ros_msg_to_dict(node: DataPreprocessor) -> None:
    # Create a dummy message with simple fields
    msg = DummyMsg(a=10, b=[1, 2, 3])
    result = node.ros_msg_to_dict(msg)
    assert result["a"] == 10
    assert result["b"] == [1, 2, 3]


def test_update_buffer(node: DataPreprocessor) -> None:
    topic = "/test_topic"
    # Ensure the buffer for the topic is empty.
    node.data_buffer[topic] = []
    # Add more entries than max_buffer_size.
    for i in range(node.max_buffer_size + 10):
        entry = {"time": i, "value": {"dummy": i}}
        node._update_buffer(topic, entry)  # noqa: SLF001
    # The buffer should not exceed max_buffer_size.
    assert len(node.data_buffer[topic]) == node.max_buffer_size
    # The first entry in the buffer should be the one after the overflow.
    assert node.data_buffer[topic][0]["time"] == 10


def test_prepare_dataset(node: DataPreprocessor) -> None:
    topic = "/test_topic"
    node.data_buffer[topic] = []
    # Populate the buffer with sample entries.
    for i in range(5):
        entry = {"time": 1_000_000_000 + i, "value": {"val": i}}
        node.data_buffer[topic].append(entry)
    dataset = node._prepare_dataset()  # noqa: SLF001
    # Verify that the dataset is not empty and contains the test topic.
    assert not dataset.is_empty()
    assert topic in dataset.columns


def test_handle_one_time_data(node: DataPreprocessor) -> None:
    # Set the one_time_data flag for the /map topic.
    node.topic_config["/map"]["one_time_data"] = True
    dummy_msg = DummyMapMsg()
    node._handle_one_time_data(dummy_msg, "/map")  # noqa: SLF001
    assert node.map_data is not None
    # Verify that the map_data has a time and value entry.
    assert "time" in node.map_data
    assert "value" in node.map_data


def test_laserscan_processing(node: DataPreprocessor) -> None:
    class LaserScanMsg:
        __slots__ = [
            "_angle_increment",
            "_angle_max",
            "_angle_min",
            "_range_max",
            "_range_min",
            "_ranges",
        ]

        def __init__(self) -> None:
            self._ranges = np.array([1.0, 2.0], dtype=np.float32)
            self._angle_min = -1.0
            self._angle_max = 1.0
            self._angle_increment = 0.1
            self._range_min = 0.0
            self._range_max = 10.0

    # Process sample message
    msg = LaserScanMsg()
    node.callback(msg, "/scan")

    # Verify buffer content
    buffer = node.data_buffer["/scan"]
    assert len(buffer) == 1
    assert buffer[0]["value"]["ranges"] == [1.0, 2.0]
    assert buffer[0]["value"]["angle_min"] == -1.0
