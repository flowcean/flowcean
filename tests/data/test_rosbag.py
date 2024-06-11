import unittest
from pathlib import Path

from flowcean.environments.rosbag import RosbagEnvironment


class TestRosbagEnvironment(unittest.TestCase):
    def test_rosbag_loader_multiple_topics(self) -> None:
        path = Path("tests/data/test_rosbag")
        environment = RosbagEnvironment(
            path=path,
            topics={
                "/j100_0000/amcl_pose": [
                    "pose.pose.position.x",
                    "pose.pose.position.y",
                ],
                "/j100_0000/platform/odom": [
                    "pose.pose.position.x",
                    "pose.pose.position.y",
                ],
            },
            custom_msgs_path=Path(
                "src/flowcean/environments/ros_msgs/nav2_msgs/msg"
            ),
        )
        environment.load()
        data = environment.get_data()

        assert data.columns == [
            "j100_0000/amcl_pose.time",
            "j100_0000/amcl_pose.pose.pose.position.x",
            "j100_0000/amcl_pose.pose.pose.position.y",
            "j100_0000/platform/odom.time",
            "j100_0000/platform/odom.pose.pose.position.x",
            "j100_0000/platform/odom.pose.pose.position.y",
        ]


if __name__ == "__main__":
    unittest.main()
