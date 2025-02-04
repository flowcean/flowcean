#!/usr/bin/env python
# /// script
# dependencies = [
#     "flowcean",
# ]
#
# [tool.uv.sources]
# flowcean = { path = "../../", editable = true }
# ///

import logging

import flowcean.cli
from flowcean.environments.rosbag import RosbagLoader
from flowcean.transforms import MatchSamplingRate

logger = logging.getLogger(__name__)


def main() -> None:
    flowcean.cli.initialize_logging()

    environment = RosbagLoader(
        path="rec_20241021_152106",
        topics={
            "/momo/pose": [
                "pose.position.x",
                "pose.position.y",
            ],
            "/odometry/filtered": [
                "pose.pose.position.x",
                "pose.pose.position.y",
            ],
        },
    )
    data = environment.observe()
    print(data)
    transform = MatchSamplingRate(
        reference_feature_name="/momo/pose",
        feature_interpolation_map={
            "/odometry/filtered": "linear",
        },
    )
    transformed_data = transform(
        data.select("/momo/pose", "/odometry/filtered"),
    )

    print(transformed_data)


if __name__ == "__main__":
    main()
