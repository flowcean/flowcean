#!/usr/bin/env python
# /// script
# dependencies = [
#     "flowcean",
# ]
# ///

import logging
from pathlib import Path

import polars as pl

import flowcean.cli
from flowcean.environments.rosbag import RosbagLoader

logger = logging.getLogger(__name__)

USE_CACHED_ROS_DATA = True
UPDATE_CACHE = False


def main() -> None:
    flowcean.cli.initialize_logging()

    if USE_CACHED_ROS_DATA and Path("cached_ros_data.json").exists():
        data = pl.read_json("cached_ros_data.json")
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
            },
            msgpaths=[
                "/opt/ros/humble/share/sensor_msgs/msg/LaserScan.msg",
                "/opt/ros/humble/share/nav2_msgs/msg/ParticleCloud.msg",
                "/opt/ros/humble/share/nav2_msgs/msg/Particle.msg",
            ],
        )
        data = environment.observe()

    if UPDATE_CACHE:
        if Path("cached_ros_data.json").exists():
            user_input = input("Overwrite cache? (y/n): ")
            if user_input == "y":
                data.write_json()
                print("Cache updated")
        else:
            data.write_json(file="cached_ros_data.json")
            print("Cache created")
    print(f"original data: {data}")


if __name__ == "__main__":
    main()
