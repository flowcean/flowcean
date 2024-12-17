#!/usr/bin/env python
# /// script
# dependencies = [
#     "flowcean",
# ]
# ///

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from utils import hough_line_transform, plot_detected_lines

import flowcean.cli
from flowcean.environments.rosbag import RosbagLoader

logger = logging.getLogger(__name__)

USE_CACHED_ROS_DATA = False
UPDATE_CACHE = False


def plot_occupancy_grid(
    occupancy_grid: np.ndarray, width: int, height: int, resolution: int
) -> None:
    plt.figure(figsize=(10, 10))  # Adjust the figure size
    plt.imshow(
        occupancy_grid,
        cmap="gray",
        origin="lower",
        extent=(0, width * resolution, 0, height * resolution),
    )
    plt.colorbar(label="Occupancy Value")
    plt.xlabel("X [meters]")
    plt.ylabel("Y [meters]")
    plt.title("Occupancy Grid Map")
    plt.show()


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
            },
            msgpaths=[
                "/opt/ros/humble/share/sensor_msgs/msg/LaserScan.msg",
                "/opt/ros/humble/share/nav2_msgs/msg/ParticleCloud.msg",
                "/opt/ros/humble/share/nav2_msgs/msg/Particle.msg",
            ],
        )
        data = environment.observe()
        map_array = data["/map"].to_list()[0][1]["value"]["data"]
        width = data["/map"].to_list()[0][1]["value"]["info.width"]
        height = data["/map"].to_list()[0][1]["value"]["info.height"]
        resolution = data["/map"].to_list()[0][1]["value"]["info.resolution"]
        occupancy_grid = np.array(map_array).reshape((height, width))
        # plot_occupancy_grid(occupancy_grid, width, height, resolution)
        detected_lines, edges = hough_line_transform(
            occupancy_grid,
            threshold1=50,
            threshold2=150,
            hough_threshold=75,
            min_line_length=10,
            max_line_gap=10,
        )
        plot_detected_lines(
            occupancy_grid=occupancy_grid, edges=edges, lines=detected_lines
        )

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
