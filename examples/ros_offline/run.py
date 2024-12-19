#!/usr/bin/env python
# /// script
# dependencies = [
#     "flowcean",
# ]
# ///

import logging
from pathlib import Path

import polars as pl
from point_distance import point_distance_transform
from scan_points import scan_points_transform

import flowcean.cli
from flowcean.environments.rosbag import RosbagLoader

logger = logging.getLogger(__name__)

USE_CACHED_ROS_DATA = False
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
                    "pose.pose.orientation.x",
                    "pose.pose.orientation.y",
                    "pose.pose.orientation.z",
                    "pose.pose.orientation.w",
                ],
                "/momo/pose": [
                    "pose.position.x",
                    "pose.position.y",
                ],
                "/scan": [
                    "ranges",
                    "angle_min",
                    "angle_max",
                    "angle_increment",
                    "range_min",
                    "range_max",
                ],
                "/map": [
                    "data",
                    "info.resolution",
                    "info.width",
                    "info.height",
                    "info.origin.position.x",
                    "info.origin.position.y",
                    "info.origin.position.z",
                    "info.origin.orientation.x",
                    "info.origin.orientation.y",
                    "info.origin.orientation.z",
                    "info.origin.orientation.w",
                ],
                "/delocalizations": ["data"],
            },
            msgpaths=[
                "/opt/ros/humble/share/sensor_msgs/msg/LaserScan.msg",
                "/opt/ros/humble/share/nav2_msgs/msg/ParticleCloud.msg",
                "/opt/ros/humble/share/nav2_msgs/msg/Particle.msg",
            ],
        )
        data = environment.observe()
        data = scan_points_transform(
            data=data, scan_topic="/scan", sensor_pose_topic="/amcl_pose"
        )
        data = point_distance_transform(data)
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
