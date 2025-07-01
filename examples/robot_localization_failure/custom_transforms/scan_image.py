import numpy as np
from numpy.typing import NDArray


class ScaleArgumentError(ValueError):
    """Error raised when an invalid number of scale arguments are provided."""

    def __init__(self) -> None:
        message = (
            "Specify exactly one of"
            "meters_per_pixel, width_meters, or height_meters"
        )
        super().__init__(message)


def scan_to_image(
    distances: NDArray[np.floating],
    angle_min: float,
    angle_increment: float,
    *,
    width: int,
    height: int,
    meters_per_pixel: float | None = None,
    width_meters: float | None = None,
    height_meters: float | None = None,
    hit_value: int = 255,
    background_value: int = 0,
) -> NDArray[np.uint8]:
    """Converts a 1D range scan into a 2D image representation.

    The function projects range measurements (distances) into Cartesian space
    using the given angle parameters, then maps these points into an image
    frame. Hit locations are assigned a specified value in the image.

    Exactly one of `meters_per_pixel`, `width_meters`, or `height_meters` must
    be specified to define the spatial scale of the image.

    Args:
        distances: A 1D array of range measurements.
        angle_min: Starting angle of the scan in radians.
        angle_increment: Angular increment between consecutive measurements.
        width: Width of the output image in pixels.
        height: Height of the output image in pixels.
        meters_per_pixel: Spatial resolution of the image in meters per pixel.
        width_meters: Width of the image in meters.
        height_meters: Height of the image in meters.
        hit_value: Pixel value assigned to valid scan hits.
        background_value: Pixel value for all other locations.

    Returns:
        A (width, height) array representing the image, with hit locations
        marked by `hit_value` and all other pixels set to `background_value`.
    """
    num_angles = distances.shape[0]
    angles = angle_min + np.arange(num_angles) * angle_increment

    specified_scales = sum(
        argument is not None
        for argument in (
            meters_per_pixel,
            width_meters,
            height_meters,
        )
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

    is_hit = (distances > 0) & np.isfinite(distances)
    xs = distances[is_hit] * np.sin(angles[is_hit])
    ys = distances[is_hit] * np.cos(angles[is_hit])

    x_px = (xs / meters_per_pixel) + (width / 2)
    y_px = (ys / meters_per_pixel) + (height / 2)

    x_idx = np.round(x_px).astype(int)
    y_idx = np.round(y_px).astype(int)

    image = np.full((width, height), background_value, dtype=np.uint8)

    in_bounds = (
        (x_idx >= 0) & (x_idx < width) & (y_idx >= 0) & (y_idx < height)
    )

    image[x_idx[in_bounds], y_idx[in_bounds]] = hit_value
    return image
