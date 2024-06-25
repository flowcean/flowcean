import logging
from pathlib import Path

import flowcean.cli
from flowcean.environments.rosbag import RosbagEnvironment
from flowcean.transforms import MatchSamplingRate, Select

logger = logging.getLogger(__name__)


def main() -> None:
    flowcean.cli.initialize_logging()

    path = Path("examples/ros_offline/example_rosbag")
    environment = RosbagEnvironment(
        path=path,
        topics={
            "/j100_0000/amcl_pose": [
                "pose.pose.position.x",
                "pose.pose.position.y",
            ],
            "/ground_truth": [
                "pose.pose.position.x",
                "pose.pose.position.y",
            ],
        },
        custom_msgs_path=Path("optional/ros/nav2_msgs/msg"),
    )
    environment.load()
    data = environment.get_data()
    print(data.head())


if __name__ == "__main__":
    main()
