import cv2
import numpy as np
from numpy.typing import NDArray

from custom_transforms.affine import Affine


class ScaleArgumentError(ValueError):
    """Error raised when an invalid number of scale arguments are provided."""

    def __init__(self) -> None:
        message = (
            "Specify exactly one of"
            "meters_per_pixel, width_meters, or height_meters"
        )
        super().__init__(message)


def crop_map_image(
    map_image: NDArray[np.uint8],
    map_resolution: float,
    map_origin: NDArray[np.floating],
    width: int,
    height: int,
    *,
    robot_position: NDArray[np.floating] | None = None,
    robot_orientation: NDArray[np.floating] | None = None,
    meters_per_pixel: float | None = None,
    width_meters: float | None = None,
    height_meters: float | None = None,
    interpolation_flags: int = cv2.INTER_AREA,
) -> NDArray:
    """Crops and resamples a section of a map image centered on the robot.

    The function extracts a region from a global map image based on the robot's
    position and orientation, using affine transformations to align and scale
    the output. The cropped image is resampled to the specified output size.

    Exactly one of `meters_per_pixel`, `width_meters`, or `height_meters` must
    be specified to define the spatial scale of the output image.

    Args:
        map_image: The input map as a 2D array.
        map_resolution: Resolution of the map in meters per pixel.
        map_origin: Origin of the map in world coordinates.
        width: Width of the output image in pixels.
        height: Height of the output image in pixels.
        robot_position: Robot's position in world coordinates.
        robot_orientation: Rotation matrix representing robot orientation.
        meters_per_pixel: Spatial resolution of the output in meters per pixel.
        width_meters: Width of the output image in meters.
        height_meters: Height of the output image in meters.
        interpolation_flags: OpenCV interpolation flags for resampling.

    Returns:
        A (height, width) array representing the cropped and resampled region
        of the map centered on the robot.
    """
    if robot_position is None:
        robot_position = np.zeros(2, dtype=np.float64)
    if robot_orientation is None:
        robot_orientation = np.eye(2, dtype=np.float64)

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

    map_pixel_to_map = Affine.from_parts(scale=map_resolution)
    map_to_world = Affine.from_parts(translation=map_origin)
    robot_to_world = Affine.from_parts(
        translation=robot_position,
        rotation=robot_orientation,
    )
    world_to_robot = robot_to_world.inverse()
    robot_to_crop = Affine.from_parts(scale=1 / meters_per_pixel)
    crop_to_crop_centered = Affine.from_parts(
        translation=np.array([width / 2, height / 2]),
    )

    transform = (
        crop_to_crop_centered
        @ robot_to_crop
        @ world_to_robot
        @ map_to_world
        @ map_pixel_to_map
    )

    return cv2.warpAffine(
        map_image,
        transform.to_homogeneous()[:2, :],
        (width, height),
        flags=interpolation_flags,
    )
