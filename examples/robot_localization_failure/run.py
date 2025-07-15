#!/usr/bin/env python
import logging
import sys
from pathlib import Path

import polars as pl
from custom_transforms.collapse import Collapse
from custom_transforms.detect_delocalizations import DetectDelocalizations
from custom_transforms.localization_status import LocalizationStatus
from custom_transforms.slice_time_series import SliceTimeSeries
from custom_transforms.zero_order_hold_matching import ZeroOrderHold

import flowcean.cli
from flowcean.core.transform import Lambda
from flowcean.polars.transforms.drop import Drop
from flowcean.ros.rosbag import RosbagLoader

logger = logging.getLogger(__name__)

config = flowcean.cli.initialize()
rosbag_dir = Path(config.rosbag.path)

if not rosbag_dir.is_dir():
    msg = f"Specified ROS2 bag directory {rosbag_dir} does not exist."
    logger.error(msg)
    sys.exit(1)

rosbag_dirs = [
    d
    for d in rosbag_dir.iterdir()
    if d.is_dir() and (d / "metadata.yaml").exists()
]
if not rosbag_dirs:
    msg = f"No ROS2 bag directories found in {rosbag_dir}. Check the path."
    logger.error(msg)
    sys.exit(1)

msg = f"Found {len(rosbag_dirs)} ROS2 bag directories in {rosbag_dir}"
logger.info(msg)

for rosbag_path in rosbag_dirs:
    msg = f"Processing ROS2 bag: {rosbag_path}"
    logger.info(msg)

    rosbag = RosbagLoader(
        path=rosbag_path,
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
        message_paths=config.rosbag.message_paths,
    )

    # Apply the preprocessing pipeline
    data = (
        rosbag
        # collapse map time series to a single value
        | Collapse("/map", element=0)
        | Lambda(
            lambda data: data.with_columns(
                pl.col("/map").struct.with_fields(
                    pl.field("data").list.eval(pl.element() != 0),
                ),
            ),
        )
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
            position_threshold=config.localization.position_threshold,
            heading_threshold=config.localization.heading_threshold,
        )
    )

    out_path = rosbag_path.with_suffix(".processed.parquet")
    msg = f"Writing processed data to {out_path}"
    logger.info(msg)
    try:
        data.observe().sink_parquet(out_path)
    except Exception as e:
        msg = f"Error writing processed data to {out_path}: {e}"
        logger.exception(msg)
        continue
    msg = f"Processing complete for {rosbag_path}"
    logger.info(msg)

logger.info("All ROS2 bag directories processed")
