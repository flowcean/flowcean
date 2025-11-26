from __future__ import annotations

import polars as pl
from custom_transforms.localization_status import LocalizationStatus
from custom_transforms.particle_cloud_statistics import ParticleCloudStatistics
from custom_transforms.scan_map_statistics import ScanMapStatistics

from flowcean.polars.transforms.drop import Drop
from flowcean.polars.transforms.explode_time_series import ExplodeTimeSeries
from flowcean.polars.transforms.resample_to_reference import (
    ResampleToReference,
)
from flowcean.ros import load_rosbag


def process_single_bag(
    bag_path: str,
    topics: dict,
    message_paths: list,
    position_threshold: float,
    heading_threshold: float,
) -> pl.DataFrame:
    print(f"\n=== Processing bag: {bag_path} ===")

    raw_lf = load_rosbag(bag_path, topics, message_paths=message_paths)

    occupancy_map = raw_lf.select("/map").collect()["/map"][0][0]["value"]

    transform = (
        ScanMapStatistics(
            occupancy_map=occupancy_map,
            scan_topic="/scan",
            sensor_pose_topic="/amcl_pose",
        )
        | ParticleCloudStatistics(
            particle_cloud_feature_name="/particle_cloud",
        )
        | Drop("/map", "/scan", "/particle_cloud")
        | ResampleToReference(reference="ray_inlier_percent")
        | LocalizationStatus(
            time_series="resampled",
            ground_truth="/momo/pose",
            estimation="/amcl_pose",
            position_threshold=position_threshold,
            heading_threshold=heading_threshold,
        )
        | ExplodeTimeSeries("resampled")
    )

    full_df = transform(raw_lf).collect().drop_nulls()
    print(full_df.columns)
    print(full_df.schema)
    return full_df
