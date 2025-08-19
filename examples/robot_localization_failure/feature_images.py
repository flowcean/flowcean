from __future__ import annotations

from collections.abc import Sized
from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np
import polars as pl
from custom_transforms.map_image import crop_map_image
from custom_transforms.particle_cloud_image import particles_to_image
from custom_transforms.scan_image import scan_to_image
from scipy.spatial.transform import Rotation
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm
from typing_extensions import override

if TYPE_CHECKING:
    from numpy.typing import NDArray

T = TypeVar("T")


class InMemoryCaching(Dataset[T], Generic[T]):
    """Wraps any sized Dataset and caches items in memory per-index."""

    def __init__(
        self,
        dataset: Dataset[T],
    ) -> None:
        if not isinstance(dataset, Sized):
            msg = "requires a Sized dataset (__len__ implemented)"
            raise TypeError(msg)
        self._dataset = dataset
        self._cache: list[None | T] = [None] * len(dataset)

    def __len__(self) -> int:
        return len(self._dataset)

    @override
    def __getitem__(self, index: int) -> T:
        cached = self._cache[index]
        if cached is not None:
            return cached

        item = self._dataset[index]
        self._cache[index] = item
        return item

    def warmup(self, *, show_progress: bool = False) -> None:
        """Preload all items into the cache."""
        indices = range(len(self))
        if show_progress:
            indices = tqdm(indices, desc="Preloading dataset")

        for i in indices:
            self[i]


class FeatureImagesData(Dataset, Sized):
    def __init__(
        self,
        inputs: pl.DataFrame,
        outputs: pl.DataFrame | None = None,
        *,
        image_size: int,
        width_meters: float,
    ) -> None:
        self.inputs = inputs
        self.outputs = outputs
        self.image_size = image_size
        self.width_meters = width_meters

    def __len__(self) -> int:
        return len(self.inputs)

    def _position_and_orientation(
        self,
        index: int,
    ) -> tuple[NDArray, NDArray]:
        row = self.inputs.slice(index, 1)

        # position: [x, y]
        position = row.select(
            pl.col("/amcl_pose")
            .struct.field("pose")
            .struct.field("position.x"),
            pl.col("/amcl_pose")
            .struct.field("pose")
            .struct.field("position.y"),
        ).to_numpy()[0]

        # quaternion: [x, y, z, w]
        quat_arr = row.select(
            pl.col("/amcl_pose")
            .struct.field("pose")
            .struct.field("orientation.x"),
            pl.col("/amcl_pose")
            .struct.field("pose")
            .struct.field("orientation.y"),
            pl.col("/amcl_pose")
            .struct.field("pose")
            .struct.field("orientation.z"),
            pl.col("/amcl_pose")
            .struct.field("pose")
            .struct.field("orientation.w"),
        ).to_numpy()[0]
        # Rotation.from_quat expects [x, y, z, w]
        orientation = Rotation.from_quat(quat_arr).as_matrix()[:2, :2]

        return (position, orientation)

    def _map_image(
        self,
        index: int,
        *,
        position: NDArray,
        orientation: NDArray,
        image_width: int,
        image_height: int,
        width_meters: float,
    ) -> NDArray:
        row = self.inputs.slice(index, 1)
        info = row.select(
            pl.col("/map").struct.field(
                "info.width",
                "info.resolution",
                "info.origin.position.x",
                "info.origin.position.y",
            ),
        ).row(0, named=True)
        map_data = (
            row.select(pl.col("/map").struct.field("data"))
            .to_series()
            .to_numpy()[0]
        )
        map_image = map_data.reshape((-1, info["info.width"])).astype(np.uint8)
        map_origin = np.array(
            [
                info["info.origin.position.x"],
                info["info.origin.position.y"],
            ],
        )
        return crop_map_image(
            map_image,
            robot_position=position,
            robot_orientation=orientation,
            map_resolution=info["info.resolution"],
            map_origin=map_origin,
            width=image_width,
            height=image_height,
            width_meters=width_meters,
        )

    def _scan_image(
        self,
        index: int,
        *,
        image_width: int,
        image_height: int,
        width_meters: float,
    ) -> NDArray:
        row = self.inputs.slice(index, 1)
        scan = (
            row.select(pl.col("/scan").struct.field("ranges"))
            .to_series()
            .to_numpy()[0]
        )
        angle_min = row.select(
            pl.col("/scan").struct.field("angle_min"),
        ).to_numpy()[0]
        angle_increment = row.select(
            pl.col("/scan").struct.field("angle_increment"),
        ).to_numpy()[0]
        return scan_to_image(
            scan,
            angle_min=angle_min,
            angle_increment=angle_increment,
            width=image_width,
            height=image_height,
            width_meters=width_meters,
            hit_value=1,
            background_value=0,
        )

    def _particle_data(
        self,
        index: int,
        *,
        position: NDArray,
        orientation: NDArray,
        image_width: int,
        image_height: int,
        width_meters: float,
    ) -> NDArray:
        row = self.inputs.slice(index, 1)
        particles = (
            row.select(pl.col("/particle_cloud").explode().struct.unnest())
            .select(
                pl.col("pose").struct.field("position").struct.field("x"),
                pl.col("pose").struct.field("position").struct.field("y"),
                pl.col("weight"),
            )
            .to_numpy()
        )
        return particles_to_image(
            particles,
            width=image_width,
            height=image_height,
            width_meters=width_meters,
            robot_position=position,
            robot_orientation=orientation,
        )

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor] | Tensor:
        position, orientation = self._position_and_orientation(index)
        map_image = self._map_image(
            index,
            position=position,
            orientation=orientation,
            image_width=self.image_size,
            image_height=self.image_size,
            width_meters=self.width_meters,
        )
        scan_image = self._scan_image(
            index,
            image_width=self.image_size,
            image_height=self.image_size,
            width_meters=self.width_meters,
        )
        particle_image = self._particle_data(
            index,
            position=position,
            orientation=orientation,
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

        if self.outputs is None:
            return inputs

        output_row = self.outputs.row(index, named=True)
        outputs = Tensor([output_row["is_delocalized"]])
        return inputs, outputs
