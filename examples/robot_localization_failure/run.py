#!/usr/bin/env python
import logging
import os
import sys
from pathlib import Path

import lightning
import polars as pl
import torch
from architectures.cnn import CNN
from custom_transforms.collapse import Collapse
from custom_transforms.detect_delocalizations import DetectDelocalizations
from custom_transforms.localization_status import LocalizationStatus
from custom_transforms.slice_time_series import SliceTimeSeries
from custom_transforms.zero_order_hold_matching import ZeroOrderHold
from feature_images import FeatureImagesData
from omegaconf import DictConfig, ListConfig
from torch.utils.data import ConcatDataset, DataLoader

import flowcean.cli
from flowcean.core.transform import Lambda
from flowcean.polars.transforms.drop import Drop
from flowcean.ros.rosbag import RosbagLoader

logger = logging.getLogger(__name__)


def main() -> None:
    # Initialize configuration and logging
    config = flowcean.cli.initialize()
    rosbag_dir, rosbag_dirs = get_rosbag_paths(config)

    # Process each ROS2 bag directory if not already processed
    for rosbag_path in rosbag_dirs:
        out_path = rosbag_path.with_suffix(".processed.parquet")
        if out_path.exists():
            msg = f"Processed data exists for {rosbag_path}, skipping."
            logger.info(msg)
            continue
        msg = f"Processing {rosbag_path} to {out_path}"
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

        # Apply preprocessing pipeline
        data = (
            rosbag
            | Collapse("/map", element=0)
            | Lambda(
                lambda df: df.with_columns(
                    pl.col("/map").struct.with_fields(
                        pl.field("data").list.eval(pl.element() != 0),
                    ),
                ),
            )
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
            | DetectDelocalizations("/delocalizations", name="slice_points")
            | Drop("/delocalizations")
            | SliceTimeSeries(
                time_series="measurements",
                slice_points="slice_points",
            )
            | Drop("slice_points")
            | LocalizationStatus(
                time_series="measurements",
                ground_truth="/momo/pose",
                estimation="/amcl_pose",
                position_threshold=config.localization.position_threshold,
                heading_threshold=config.localization.heading_threshold,
            )
        )

        # Save processed data
        try:
            data.observe().sink_parquet(out_path)
            msg = f"Processed data saved to {out_path}"
            logger.info(msg)
        except Exception as e:
            msg = f"Error processing {rosbag_path}: {e}"
            logger.exception(msg)
            continue

    # Training: Load all processed Parquet files
    parquet_files = list(rosbag_dir.glob("*.processed.parquet"))
    if not parquet_files:
        msg = f"No processed Parquet files found in {rosbag_dir}."
        logger.error(msg)
        sys.exit(1)
    msg = f"Found {len(parquet_files)} processed Parquet files in {rosbag_dir}"
    logger.info(msg)

    datasets = []
    for parquet_path in parquet_files:
        msg = f"Loading Parquet file: {parquet_path}"
        logger.info(msg)
        try:
            data = pl.read_parquet(parquet_path)
            data = (
                data.explode("measurements")
                .unnest("measurements")
                .unnest("value")
            )
            dataset = FeatureImagesData(
                data,
                image_size=config.architecture.image_size,
                width_meters=config.architecture.width_meters,
            )
            datasets.append(dataset)
        except Exception as e:
            msg = f"Error loading {parquet_path}: {e}"
            logger.exception(msg)
            continue

    if not datasets:
        logger.error("No datasets loaded, cannot proceed with training.")
        sys.exit(1)

    # Combine datasets and train
    combined_dataset = ConcatDataset(datasets)

    robot_data = DataLoader(
        combined_dataset,
        batch_size=config.learning.batch_size,
        num_workers=os.cpu_count() or 0,
    )

    module = CNN(
        image_size=config.architecture.image_size,
        in_channels=3,
        learning_rate=config.learning.learning_rate,
    )
    trainer = lightning.Trainer(max_epochs=config.learning.epochs)
    trainer.fit(module, robot_data)

    # Save the model
    out_path = f"models/{rosbag_dir.name}.pt"
    msg = f"Saving model to {out_path}"
    logger.info(msg)
    torch.save(module.state_dict(), out_path)


def get_rosbag_paths(
    config: DictConfig | ListConfig,
) -> tuple[Path, list[Path]]:
    rosbag_dir = Path(config.rosbag.path)

    if not rosbag_dir.is_dir():
        msg = f"Specified ROS2 bag directory {rosbag_dir} does not exist."
        logger.error(msg)
        sys.exit(1)

    # Find ROS2 bag subdirectories
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

    return rosbag_dir, rosbag_dirs


if __name__ == "__main__":
    main()
