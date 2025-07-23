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
    """Converts a set of weighted particles into a 2D image representation.

    Each particle is a 3-element vector representing (x, y, weight). The
    function projects particle positions into an image frame using the robot's
    position and orientation, and accumulates their weights into corresponding
    image pixels.

    Exactly one of `meters_per_pixel`, `width_meters`, or `height_meters` must
    be specified to define the spatial scale of the image.

    Args:
        particles: An (N, 3) array of particles, with (x, y, weight).
        width: Width of the output image in pixels.
        height: Height of the output image in pixels.
        robot_position: Vector representing the robot's (x, y) position.
        robot_orientation: Rotation matrix representing robot orientation.
        meters_per_pixel: Spatial resolution of the image in meters per pixel.
        width_meters: Width of the image in meters.
        height_meters: height of the image in meters.

    Returns:
        A (height, width) array representing the image, where each pixel
        contains the accumulated weight of particles mapped to that location.
    """
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
