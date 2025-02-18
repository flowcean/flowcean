import numpy as np
import pytest
import rclpy
from flowcean_ros.flowcean_ros.data_preprocessor import DataPreprocessor
from rclpy.clock import Clock
from rclpy.time import Time


# Dummy classes to simulate ROS time
class DummyTime:
    def __init__(self, sec=1, nanosec=500):
        self.sec = sec
        self.nanosec = nanosec


class DummyClock(Clock):
    def __init__(self):
        super().__init__()

    def now(self) -> Time:
        return Time(seconds=1, nanoseconds=500, clock_type=self.clock_type)


# Dummy message for testing ros_msg_to_dict and related methods
class DummyMsg:
    __slots__ = ["a", "b"]

    def __init__(self, a, b):
        self.a = a
        self.b = b


# Fixture to initialize a DataPreprocessor node for testing.
# We override get_clock to use a DummyClock.
@pytest.fixture
def node():
    rclpy.init(args=[])
    dp_node = DataPreprocessor()
    dp_node.get_clock = lambda: DummyClock()
    yield dp_node
    dp_node.destroy_node()
    rclpy.shutdown()


def test_ros_msg_to_dict(node):
    # Create a dummy message with simple fields
    msg = DummyMsg(a=10, b=[1, 2, 3])
    result = node.ros_msg_to_dict(msg)
    assert result["a"] == 10
    assert result["b"] == [1, 2, 3]


def test_set_nested_value(node):
    data = {}
    parts = ["level1", "level2", "value"]
    node._set_nested_value(data, parts, 42)
    assert data == {"level1": {"level2": {"value": 42}}}


def test_extract_fields(node):
    # Simulate a nested dictionary similar to what ros_msg_to_dict would produce.
    msg_dict = {
        "pose": {
            "pose": {
                "position": {
                    "x": 3.14,
                    "y": 0.0,
                },
            },
        },
    }
    fields = ["pose.pose.position.x"]
    extracted = node._extract_fields(msg_dict, fields)
    assert (
        extracted.get("pose", {}).get("pose", {}).get("position", {}).get("x")
        == 3.14
    )


def test_update_buffer(node):
    topic = "/test_topic"
    # Ensure the buffer for the topic is empty.
    node.data_buffer[topic] = []
    # Add more entries than max_buffer_size.
    for i in range(node.max_buffer_size + 10):
        entry = {"time": i, "value": {"dummy": i}}
        node._update_buffer(topic, entry)
    # The buffer should not exceed max_buffer_size.
    assert len(node.data_buffer[topic]) == node.max_buffer_size
    # The first entry in the buffer should be the one after the overflow.
    assert node.data_buffer[topic][0]["time"] == 10


def test_prepare_dataset(node):
    topic = "/test_topic"
    node.data_buffer[topic] = []
    # Populate the buffer with sample entries.
    for i in range(5):
        entry = {"time": 1_000_000_000 + i, "value": {"val": i}}
        node.data_buffer[topic].append(entry)
    dataset = node._prepare_dataset()
    # Verify that the dataset is not empty and contains the test topic.
    assert not dataset.is_empty()
    assert topic in dataset.columns


def test_handle_one_time_data(node):
    # Dummy message classes to simulate a map message.
    class DummyOrigin:
        __slots__ = ["orientation", "position"]

        def __init__(self):
            self.position = DummyPosition()
            self.orientation = DummyOrientation()

    class DummyPosition:
        __slots__ = ["x", "y", "z"]

        def __init__(self):
            self.x = 1.0
            self.y = 2.0
            self.z = 3.0

    class DummyOrientation:
        __slots__ = ["w", "x", "y", "z"]

        def __init__(self):
            self.x = 0.0
            self.y = 0.0
            self.z = 0.0
            self.w = 1.0

    class DummyInfo:
        __slots__ = ["height", "origin", "resolution", "width"]

        def __init__(self):
            self.resolution = 0.5
            self.width = 10
            self.height = 10
            self.origin = DummyOrigin()

    class DummyMapMsg:
        __slots__ = ["data", "info"]

        def __init__(self):
            self.data = [0, 1, 2]
            self.info = DummyInfo()

    # Set the one_time_data flag for the /map topic.
    node.topic_config["/map"]["one_time_data"] = True
    dummy_msg = DummyMapMsg()
    node._handle_one_time_data(dummy_msg, "/map")
    assert node.map_data is not None
    # Verify that the map_data has a time and value entry.
    assert "time" in node.map_data
    assert "value" in node.map_data


def test_laserscan_processing(node):
    class LaserScanMsg:
        __slots__ = [
            "_angle_increment",
            "_angle_max",
            "_angle_min",
            "_range_max",
            "_range_min",
            "_ranges",
        ]

        def __init__(self):
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


def test_slashed_topic_name(node):
    class PoseMsg:
        __slots__ = ["_pose"]

        def __init__(self):
            self._pose = type(
                "",
                (),
                {"position": type("", (), {"x": 3.14})},
            )()

    # Process sample message
    msg = PoseMsg()
    node.callback(msg, "/pose")

    # Verify dataset structure
    dataset = node._prepare_dataset()
    assert "/pose" in dataset.columns
    assert dataset["/pose"][0][0]["value"]["pose"]["position"]["x"] == 3.14


def test_particle_cloud_processing(node):
    class Particle:
        __slots__ = ["_pose"]

        def __init__(self):
            self._pose = type("", (), {"x": 1.0, "y": 2.0})()

    class ParticleCloudMsg:
        __slots__ = ["_particles"]

        def __init__(self):
            self._particles = [Particle(), Particle()]

    # Process sample message
    msg = ParticleCloudMsg()
    node.callback(msg, "/particle_cloud")

    # Verify buffer content
    buffer = node.data_buffer["/particle_cloud"]
    assert len(buffer) == 1
    assert len(buffer[0]["value"]["particles"]) == 2
    assert buffer[0]["value"]["particles"][0]["pose"]["x"] == 1.0
