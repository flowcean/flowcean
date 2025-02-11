#!/usr/bin/env python
# /// script
# dependencies = [
#     "flowcean",
# ]
#
# [tool.uv.sources]
# flowcean = { path = "../../", editable = true }
# ///

from pathlib import Path

import polars as pl

import flowcean.cli
from flowcean.ros import RosbagLoader

USE_CACHED_ROS_DATA = False
UPDATE_CACHE = False
WS = Path(__file__).resolve().parent


def main() -> None:
    flowcean.cli.initialize_logging()

    if USE_CACHED_ROS_DATA and Path("cached_ros_data.json").exists():
        data = pl.read_json("cached_ros_data.json").lazy()
    else:
        environment = RosbagLoader(
            path="rec_20241021_152106",
            topics={
                "/amcl_pose": [
                    "pose.pose.position.x",
                    "pose.pose.position.y",
                ],
                "/momo/pose": [
                    "pose.position.x",
                    "pose.position.y",
                ],
                "/scan": [
                    "ranges",
                ],
                "/particle_cloud": ["particles"],
                "/position_error": ["data"],
                "/heading_error": ["data"],
            },
            msgpaths=[
                str(WS / "ros_msgs/LaserScan.msg"),
                str(WS / "ros_msgs/nav2_msgs/msg/Particle.msg"),
                str(WS / "ros_msgs/nav2_msgs/msg/ParticleCloud.msg"),
            ],
        )
        data = environment.observe()

        if UPDATE_CACHE:
            if Path("cached_ros_data.json").exists():
                user_input = input("Overwrite cache? (y/n): ")
                if user_input == "y":
                    data.collect().write_json()
                    print("Cache updated")
            else:
                data.collect().write_json(file="cached_ros_data.json")
                print("Cache created")
    print(f"original data: {data.collect()}")


if __name__ == "__main__":
    main()
