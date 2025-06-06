import logging

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class ScaleArgumentError(ValueError):
    """Error raised when an invalid number of scale arguments are provided."""

    def __init__(self) -> None:
        message = (
            "Specify exactly one of"
            "meters_per_pixel, width_meters, or height_meters"
        )
        super().__init__(message)


def particles_to_image(
    particles: NDArray[np.floating],
    width: int,
    height: int,
    *,
    robot_position: NDArray[np.floating] | None = None,
    robot_orientation: NDArray[np.floating] | None = None,
    meters_per_pixel: float | None = None,
    width_meters: float | None = None,
    height_meters: float | None = None,
) -> NDArray[np.floating]:
    if robot_position is None:
        robot_position = np.zeros(2, dtype=np.float64)
    if robot_orientation is None:
        robot_orientation = np.eye(2, dtype=np.float64)

    specified_scales = sum(
        argument is not None
        for argument in (meters_per_pixel, width_meters, height_meters)
    )
    if specified_scales != 1:
        raise ScaleArgumentError

    if meters_per_pixel is None:
        if width_meters is not None:
            meters_per_pixel = width_meters / width
        elif height_meters is not None:
            meters_per_pixel = height_meters / height
        else:
            raise ScaleArgumentError

    image = np.zeros((height, width), dtype=float)

    positions = particles[:, :2].T
    x_trans, y_trans = (
        robot_orientation.T @ positions
        + robot_orientation.T @ -robot_position[:, None]
    )

    x_idx = np.round((x_trans / meters_per_pixel) + (height / 2)).astype(int)
    y_idx = np.round((y_trans / meters_per_pixel) + (width / 2)).astype(int)
    ws = particles[:, 2]

    in_bounds = (
        (x_idx >= 0) & (x_idx < width) & (y_idx >= 0) & (y_idx < height)
    )
    np.add.at(image, (x_idx[in_bounds], y_idx[in_bounds]), ws[in_bounds])

    return image
