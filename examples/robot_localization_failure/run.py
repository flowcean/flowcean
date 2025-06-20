#!/usr/bin/env python

import logging
from pathlib import Path

from custom_transforms.collapse import Collapse
from custom_transforms.detect_delocalizations import DetectDelocalizations
from custom_transforms.localization_status import LocalizationStatus
from custom_transforms.slice_time_series import SliceTimeSeries
from custom_transforms.zero_order_hold_matching import ZeroOrderHold

import flowcean.cli
from flowcean.polars.transforms.drop import Drop
from flowcean.ros.rosbag import RosbagLoader

logger = logging.getLogger(__name__)

WS = Path(__file__).resolve().parent
ROSBAG = WS / "recordings/rec_20250618_113817"
ROS_MESSAGE_TYPES = [
    WS / "ros_msgs/LaserScan.msg",
    WS / "ros_msgs/nav2_msgs/msg/Particle.msg",
    WS / "ros_msgs/nav2_msgs/msg/ParticleCloud.msg",
]


flowcean.cli.initialize_logging(log_level=logging.DEBUG)

rosbag = RosbagLoader(
    path=ROSBAG,
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
            "pose.orientation.x",
            "pose.orientation.y",
            "pose.orientation.z",
            "pose.orientation.w",
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
    },
    message_paths=ROS_MESSAGE_TYPES,
)
logger.info("Loaded data from ROS bag")

rc = rosbag.data.collect()

data = (
    rosbag
    # collapse map time series to a single value
    | Collapse("/map", element=0)
    # align all time series features using zero-order hold
    | ZeroOrderHold(
        features=[
            "/scan",
            "/particle_cloud",
            "/momo/pose",
            "/amcl_pose",
        ],
        name="measurements",
    )
    | Drop("/scan", "/particle_cloud", "/momo/pose", "/amcl_pose")
    # detect experiment slice points based on delocalization events
    | DetectDelocalizations("/delocalizations", name="slice_points")
    | Drop("/delocalizations")
    | SliceTimeSeries(
        time_series="measurements",
        slice_points="slice_points",
    )
    | Drop("slice_points")
    # detect localization status based on position and heading errors
    | LocalizationStatus(
        time_series="measurements",
        ground_truth="/momo/pose",
        estimation="/amcl_pose",
        position_threshold=0.4,
        heading_threshold=0.4,
    )
)

logger.info("Writing processed data to Parquet file...")
out_path = ROSBAG.with_suffix(".processed.parquet")
data.observe().sink_parquet(out_path)
logger.info("Data processing complete. Processed data saved to %s", out_path)
