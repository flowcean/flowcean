import logging

import polars.selectors as cs

import flowcean.cli
from flowcean.environments.rosbag import RosbagLoader
from flowcean.transforms import MatchSamplingRate

logger = logging.getLogger(__name__)


def main() -> None:
    flowcean.cli.initialize_logging()

    environment = RosbagLoader(
        path="example_rosbag",
        topics={
            "/j100_0000/amcl_pose": [
                "pose.pose.position.x",
                "pose.pose.position.y",
            ],
            "/j100_0000/odometry": [
                "pose.pose.position.x",
                "pose.pose.position.y",
            ],
        },
    )
    environment.load()
    data = environment.get_data()
    print(data)
    transform = MatchSamplingRate(
        reference_feature_name="/j100_0000/amcl_pose",
        feature_interpolation_map={
            "/j100_0000/odometry": "linear",
        },
    )
    transformed_data = transform.transform(
        data.select("/j100_0000/amcl_pose", "/j100_0000/odometry")
    )

    print(transformed_data)


if __name__ == "__main__":
    main()
