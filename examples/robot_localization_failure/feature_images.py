import numpy as np
import polars as pl
from custom_transforms.map_image import crop_map_image
from custom_transforms.particle_cloud_image import particles_to_image
from custom_transforms.scan_image import scan_to_image
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation
from torch import Tensor
from torch.utils.data import Dataset


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
