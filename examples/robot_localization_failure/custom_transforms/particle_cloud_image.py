import logging

import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation
from typing_extensions import override

from flowcean.core.transform import Transform

logger = logging.getLogger(__name__)


class ScaleArgumentError(ValueError):
    """Error raised when an invalid number of scale arguments are provided."""

    def __init__(self) -> None:
        message = (
            "Specify exactly one of"
            "meter_per_pixel, width_meters, or height_meters"
        )
        super().__init__(message)


def particles_to_image(
    particles: np.ndarray,
    width: int,
    height: int,
    *,
    isometry: tuple[np.ndarray, np.ndarray],
    meter_per_pixel: float | None = None,
    width_meters: float | None = None,
    height_meters: float | None = None,
) -> np.ndarray:
    specified_scales = sum(
        argument is not None
        for argument in (meter_per_pixel, width_meters, height_meters)
    )
    if specified_scales != 1:
        raise ScaleArgumentError
    if meter_per_pixel is None:
        if width_meters is not None:
            meter_per_pixel = width_meters / width
        elif height_meters is not None:
            meter_per_pixel = height_meters / height
        else:
            raise ScaleArgumentError

    rotation, translation = isometry
    rotation = np.asarray(rotation, float).reshape(2, 2)
    translation = np.asarray(translation, float).reshape(2)

    image = np.zeros((height, width), dtype=float)

    pts = particles[:, :2].T
    x_trans, y_trans = rotation @ pts + translation[:, None]
    ws = particles[:, 2]

    cols = np.round((y_trans / meter_per_pixel) + (width / 2)).astype(int)
    rows = np.round(-(x_trans / meter_per_pixel) + (height / 2)).astype(int)

    mask = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
    np.add.at(image, (rows[mask], cols[mask]), ws[mask])

    return image


class ParticleCloudImage(Transform):
    def __init__(
        self,
        time_series: str,
        *,
        width: int,
        height: int,
        meter_per_pixel: float | None = None,
        width_meters: float | None = None,
        height_meters: float | None = None,
    ) -> None:
        super().__init__()
        self.time_series = time_series
        self.width = width
        self.height = height
        self.meter_per_pixel = meter_per_pixel
        self.width_meters = width_meters
        self.height_meters = height_meters

    @override
    def apply(self, data: pl.LazyFrame) -> pl.LazyFrame:
        def _compute_image(x: dict) -> NDArray:
            particles = np.array(
                [
                    [particle["x"], particle["y"], particle["weight"]]
                    for particle in x["particles"]
                ],
            )
            position = np.array([x["position"]["x"], x["position"]["y"]])
            rotation = (
                Rotation.from_quat(
                    (
                        x["orientation"]["x"],
                        x["orientation"]["y"],
                        x["orientation"]["z"],
                        x["orientation"]["w"],
                    ),
                )
                .inv()
                .as_matrix()[:2, :2]
            )
            map_to_robot = (rotation, rotation @ -position)
            return particles_to_image(
                particles,
                width=self.width,
                height=self.height,
                meter_per_pixel=self.meter_per_pixel,
                width_meters=self.width_meters,
                height_meters=self.height_meters,
                isometry=map_to_robot,
            )

        values = pl.col(self.time_series).explode().struct.field("value")
        particles = (
            values.struct.field("/particle_cloud/particles")
            .list.eval(
                pl.struct(
                    pl.element()
                    .struct.field("pose")
                    .struct.field("position")
                    .struct.field("x"),
                    pl.element()
                    .struct.field("pose")
                    .struct.field("position")
                    .struct.field("y"),
                    pl.element().struct.field("weight"),
                ),
            )
            .alias("particles")
        )
        position = pl.struct(
            values.struct.field("/amcl_pose/pose.pose.position.x").alias("x"),
            values.struct.field("/amcl_pose/pose.pose.position.y").alias("y"),
        ).alias("position")
        orientation = pl.struct(
            values.struct.field("/amcl_pose/pose.pose.orientation.x").alias(
                "x",
            ),
            values.struct.field("/amcl_pose/pose.pose.orientation.y").alias(
                "y",
            ),
            values.struct.field("/amcl_pose/pose.pose.orientation.z").alias(
                "z",
            ),
            values.struct.field("/amcl_pose/pose.pose.orientation.w").alias(
                "w",
            ),
        ).alias("orientation")
        return data.with_columns(
            pl.struct(
                particles,
                position,
                orientation,
            )
            .map_elements(_compute_image, return_dtype=pl.Object)
            .implode(),
        )
