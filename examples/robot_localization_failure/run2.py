import logging
import os
import sys
from pathlib import Path

import lightning
import numpy as np
import polars as pl
import torch
from architectures.cnn import CNN
from custom_transforms.map_image import crop_map_image
from custom_transforms.particle_cloud_image import particles_to_image
from custom_transforms.scan_image import scan_to_image
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation
from torch import Tensor
from torch.utils.data import ConcatDataset, DataLoader, Dataset

import flowcean.cli
from flowcean.core.transform import Lambda
from flowcean.polars import DataFrame

_ = pl

logger = logging.getLogger(__name__)

config = flowcean.cli.initialize()
rosbag_dir = Path(config.rosbag.path)

parquet_files = list(rosbag_dir.glob("*.processed.parquet"))
if not parquet_files:
    msg = f"No processed Parquet files found in {rosbag_dir}. Check the path."
    logger.error(msg)
    sys.exit(1)
msg = f"Found {len(parquet_files)} processed Parquet files in {rosbag_dir}"
logger.info(msg)


class FeatureImagesData(Dataset):
    def __init__(
        self,
        data: pl.DataFrame,
        *,
        image_size: int,
        width_meters: float,
    ) -> None:
        self.data = data
        self.image_size = image_size
        self.width_meters = width_meters

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        row = self.data.row(index, named=True)
        map_image = compute_map_image(
            row,
            image_width=self.image_size,
            image_height=self.image_size,
            width_meters=self.width_meters,
        )
        scan_image = compute_scan_image(
            row,
            image_width=self.image_size,
            image_height=self.image_size,
            width_meters=self.width_meters,
        )
        particle_image = compute_particle_image(
            row,
            image_width=self.image_size,
            image_height=self.image_size,
            width_meters=self.width_meters,
        )
        inputs = np.stack(
            [
                map_image,
                scan_image,
                particle_image,
            ],
            axis=0,
        )
        inputs = Tensor(inputs)
        outputs = Tensor([row["is_delocalized"]])
        return inputs, outputs

    def __len__(self) -> int:
        return len(self.data)


def extract_position_and_orientation(x: dict) -> tuple[NDArray, NDArray]:
    position = np.array(
        [
            x["/amcl_pose/pose.pose.position.x"],
            x["/amcl_pose/pose.pose.position.y"],
        ],
    )
    orientation = Rotation.from_quat(
        (
            x["/amcl_pose/pose.pose.orientation.x"],
            x["/amcl_pose/pose.pose.orientation.y"],
            x["/amcl_pose/pose.pose.orientation.z"],
            x["/amcl_pose/pose.pose.orientation.w"],
        ),
    ).as_matrix()[:2, :2]
    return (position, orientation)


def compute_map_image(
    x: dict,
    image_width: int,
    image_height: int,
    width_meters: float,
) -> NDArray:
    map_image = (
        np.array(x["/map"]["data"])
        .reshape(
            (-1, x["/map"]["info.width"]),
        )
        .astype(np.uint8)
    )
    position, orientation = extract_position_and_orientation(x)
    map_resolution = x["/map"]["info.resolution"]
    map_origin = np.array(
        [
            x["/map"]["info.origin.position.x"],
            x["/map"]["info.origin.position.y"],
        ],
    )

    return crop_map_image(
        map_image,
        robot_position=position,
        robot_orientation=orientation,
        map_resolution=map_resolution,
        map_origin=map_origin,
        width=image_width,
        height=image_height,
        width_meters=width_meters,
    )


def compute_scan_image(
    x: dict,
    image_width: int,
    image_height: int,
    width_meters: float,
) -> NDArray:
    distances = np.array(x["/scan/ranges"])
    return scan_to_image(
        distances,
        angle_min=x["/scan/angle_min"],
        angle_increment=x["/scan/angle_increment"],
        width=image_width,
        height=image_height,
        width_meters=width_meters,
        hit_value=1,
        background_value=0,
    )


def compute_particle_image(
    x: dict,
    image_width: int,
    image_height: int,
    width_meters: float,
) -> NDArray:
    particles = np.array(
        [
            [
                particle["pose"]["position"]["x"],
                particle["pose"]["position"]["y"],
                particle["weight"],
            ]
            for particle in x["/particle_cloud/particles"]
        ],
    )
    position, orientation = extract_position_and_orientation(x)
    return particles_to_image(
        particles,
        width=image_width,
        height=image_height,
        width_meters=width_meters,
        robot_position=position,
        robot_orientation=orientation,
    )


datasets = []
for parquet_path in parquet_files:
    msg = f"Loading Parquet file: {parquet_path}"
    logger.info(msg)
    try:
        environment = DataFrame.from_parquet(parquet_path)
    except Exception as e:
        msg = f"Error loading Parquet file {parquet_path}: {e}"
        logger.exception(msg)
        continue
    environment.with_transform(
        Lambda(
            lambda data: data.explode("measurements")
            .unnest("measurements")
            .unnest("value"),
        ),
    )
    data = environment.observe().collect()
    dataset = FeatureImagesData(
        data,
        image_size=config.architecture.image_size,
        width_meters=config.architecture.width_meters,
    )
    datasets.append(dataset)

combined_dataset = ConcatDataset(datasets)
msg = f"Combined dataset size: {len(combined_dataset)} samples"
logger.info(msg)

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
out_path = "models/" + rosbag_dir.name + ".pt"
msg = f"Saving model to {out_path}"
logger.info(msg)
torch.save(module.state_dict(), out_path)
