#!/usr/bin/env python
# /// script
# dependencies = [
#     "flowcean",
# ]
# ///

import logging

import polars.selectors as cs

import flowcean.cli
from flowcean.environments.rosbag import RosbagLoader

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
    data = environment.observe()
    print(data)
    data = (
        data.select("/j100_0000/amcl_pose")
        .explode(cs.all())
        .unnest(cs.all())
        .unnest("value")
    )
    print(data)


if __name__ == "__main__":
    main()
