#!/usr/bin/env python
import logging
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

rosbag = RosbagLoader(
    path=config.rosbag.path,
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
logger.info("Loaded data from ROS bag")

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

logger.info("Writing processed data to Parquet file...")
out_path = Path(config.rosbag.path).with_suffix(".processed.parquet")
data.observe().sink_parquet(out_path)
logger.info("Data processing complete. Processed data saved to %s", out_path)
