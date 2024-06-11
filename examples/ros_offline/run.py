import logging
from pathlib import Path

import flowcean.cli
from flowcean.environments.rosbag import RosbagEnvironment
from flowcean.transforms import MatchSamplingRate, Select

logger = logging.getLogger(__name__)


def main() -> None:
    flowcean.cli.initialize_logging()

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
    transforms = MatchSamplingRate(
        reference_timestamps="j100_0000/amcl_pose.time",
        feature_columns_with_timestamps={
            "j100_0000/platform/odom.pose.pose.position.x": "j100_0000/platform/odom.time"  # noqa: E501
        },
    ) | Select(
        features=[
            "j100_0000/amcl_pose.pose.pose.position.x",
            "j100_0000/platform/odom.pose.pose.position.x",
        ]
    )
    transformed_data = transforms.transform(data)
    print(data.head())
    print(transformed_data.head())


if __name__ == "__main__":
    main()
