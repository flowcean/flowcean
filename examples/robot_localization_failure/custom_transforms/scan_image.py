import numpy as np
from numpy.typing import NDArray


def lidar_to_image(
    distances: np.ndarray,
    angle_min: float,
    angle_increment: float,
    width_px: int,
    height_px: int,
    width_m: float,
    height_m: float,
    hit_value: int = 255,
    background_value: int = 0,
) -> NDArray:
    num_angles = distances.shape[0]
    angles = angle_min + np.arange(num_angles) * angle_increment

    xs = distances * np.cos(angles)
    ys = distances * np.sin(angles)

    x_px = ((xs + width_m / 2) / width_m) * width_px
    y_px = ((height_m / 2 - ys) / height_m) * height_px

    x_idx = np.round(x_px).astype(int)
    y_idx = np.round(y_px).astype(int)

    image = np.full((height_px, width_px), background_value, dtype=np.uint8)

    in_bounds = (
        (x_idx >= 0)
        & (x_idx < width_px)
        & (y_idx >= 0)
        & (y_idx < height_px)
        & (distances > 0)
        & (np.isfinite(distances))
    )

    image[y_idx[in_bounds], x_idx[in_bounds]] = hit_value
    return image
