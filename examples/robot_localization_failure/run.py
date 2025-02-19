#!/usr/bin/env python
# /// script
# dependencies = [
#     "flowcean",
#     "matplotlib",
#     "opencv-python",
# ]
#
# [tool.uv.sources]
# flowcean = { path = "../../", editable = true }
# ///

from pathlib import Path

import polars as pl
from custom_transforms.scan_map import ScanMap

import flowcean.cli
from flowcean.polars.transforms.particle_cloud_image import ParticleCloudImage
from flowcean.ros.rosbag import RosbagLoader

USE_CACHED_ROS_DATA = True
UPDATE_CACHE = False
WS = Path(__file__).resolve().parent


def main() -> None:
    flowcean.cli.initialize_logging()

    if USE_CACHED_ROS_DATA:
        if Path(WS / "cached_ros_data.json").exists():
            print("Loading data from cache.")
            data = pl.read_json(WS / "cached_ros_data.json").lazy()
        else:
            msg = "Cached data not found."
            raise FileNotFoundError(msg)
    else:
        print("Loading data from rosbag.")
        environment = RosbagLoader(
            path=WS / "rec_20241021_152106",
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
    print(f"loaded data: {data.collect()}")
    transform = ScanMap(plotting=True)
    transformed_data = transform(data)
    print(f"transformed data: {transformed_data.collect()}")

    if UPDATE_CACHE:
        collected_data = data.collect()
        if Path(WS / "cached_ros_data.json").exists():
            collected_data.write_json()
            print("Cache updated")
        else:
            collected_data.write_json(file=WS / "cached_ros_data.json")
            print("Cache created")

    transform = ParticleCloudImage(
        particle_cloud_feature_name="/particle_cloud",
        save_images=True,
        cutting_area=15.0,
        image_pixel_size=300,
    )

    transformed_data = transform(data)
    print(f"transformed data: {transformed_data.collect()}")

    if UPDATE_CACHE:
        if Path("cached_ros_data.json").exists():
            user_input = input("Overwrite cache? (y/n): ")
            if user_input == "y":
                data.collect().write_json()
                print("Cache updated")
        else:
            data.collect().write_json(file="cached_ros_data.json")
            print("Cache created")


if __name__ == "__main__":
    main()
