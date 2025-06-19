#!/usr/bin/env python

import logging
from pathlib import Path

from custom_transforms.collapse import Collapse
from custom_transforms.detect_delocalizations import DetectDelocalizations
from custom_transforms.localization_status import LocalizationStatus
from custom_transforms.slice_time_series import SliceTimeSeries
from custom_transforms.zero_order_hold_matching import ZeroOrderHold
from rosbag import load_or_cache_ros_data

import flowcean.cli
from flowcean.polars.transforms.drop import Drop

logger = logging.getLogger(__name__)

WS = Path(__file__).resolve().parent
ROSBAG_NAME = "rec_20241021_152106"
ROSBAG_PATH = WS / ROSBAG_NAME
ROS_MESSAGE_TYPES = [
    WS / "ros_msgs/LaserScan.msg",
    WS / "ros_msgs/nav2_msgs/msg/Particle.msg",
    WS / "ros_msgs/nav2_msgs/msg/ParticleCloud.msg",
]

SAVE_IMAGES = True
IMAGE_PIXEL_SIZE = 100
CROP_REGION_SIZE = 5.0


flowcean.cli.initialize_logging(log_level=logging.DEBUG)

data = load_or_cache_ros_data(
    ROSBAG_PATH,
    message_definitions=ROS_MESSAGE_TYPES,
    ignore_cache=True,
)
logger.info("Loaded data from ROS bag")


transform = (
    # collapse map time series to a single value
    Collapse("/map", element=1)
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

# transformed_data = transform(data)
# collected = transformed_data.collect(engine="streaming")
# print(collected)
