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


class FeatureImagesPredictionData(Dataset):
    def __init__(
        self,
        data: pl.DataFrame,
        image_size: int,
        width_meters: float,
    ) -> None:
        self.data = data
        self.image_size = image_size
        self.width_meters = width_meters

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tensor:
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
        inputs = np.stack([map_image, scan_image, particle_image], axis=0)
        return Tensor(inputs)


def extract_position_and_orientation(x: dict) -> tuple[NDArray, NDArray]:
    amcl_pose = x["/amcl_pose"]["pose"]
    position = np.array(
        [
            amcl_pose["position.x"],
            amcl_pose["position.y"],
        ],
    )
    orientation = Rotation.from_quat(
        (
            amcl_pose["orientation.x"],
            amcl_pose["orientation.y"],
            amcl_pose["orientation.z"],
            amcl_pose["orientation.w"],
        ),
    ).as_matrix()[:2, :2]
    return (position, orientation)


def compute_map_image(
    x: dict,
    image_width: int,
    image_height: int,
    width_meters: float,
) -> NDArray:
    map_data = x["/map"]
    map_image = (
        np.array(map_data["data"])
        .reshape(
            (-1, map_data["info.width"]),
        )
        .astype(np.uint8)
    )
    position, orientation = extract_position_and_orientation(x)
    map_resolution = map_data["info.resolution"]
    map_origin = np.array(
        [
            map_data["info.origin.position.x"],
            map_data["info.origin.position.y"],
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
    scan = x["/scan"]
    distances = np.array(scan["ranges"])
    return scan_to_image(
        distances,
        angle_min=scan["angle_min"],
        angle_increment=scan["angle_increment"],
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
    particle_cloud = x["/particle_cloud"]
    particles = np.array(
        [
            [
                particle["pose"]["position"]["x"],
                particle["pose"]["position"]["y"],
                particle["weight"],
            ]
            for particle in particle_cloud
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
