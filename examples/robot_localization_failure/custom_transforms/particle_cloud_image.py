import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from PIL import Image

from flowcean.core.transform import Transform

logger = logging.getLogger(__name__)


class ParticleCloudImage(Transform):
    """Generates grayscale images from particle cloud data.

    Processes 2D particle cloud data to create grayscale images highlighting
    the particle distribution. Optionally saves images to disk and returns them
    embedded in a Polars DataFrame for further analysis. The respective column
    is named `/particle_cloud_image`.
    """

    def __init__(
        self,
        particle_cloud_feature_name: str = "/particle_cloud",
        cutting_area: float = 15.0,
        image_pixel_size: int = 300,
        *,
        save_images: bool = False,
    ) -> None:
        """Initialize the ParticleCloudImage transform.

        Args:
            particle_cloud_feature_name: Name of the particle cloud feature.
            cutting_area: Side length of square region for cropping (meters).
            image_pixel_size: Output image resolution (both width and height).
            save_images: Whether to save images to disk.
        """
        self.particle_cloud_feature_name = particle_cloud_feature_name
        self.cutting_area = cutting_area
        self.image_pixel_size = image_pixel_size
        self.save_images = save_images

    def apply(self, data: pl.LazyFrame) -> pl.LazyFrame:
        logger.debug("Processing particle cloud data to generate images.")

        half_region = self.cutting_area / 2.0
        region_str = str(self.cutting_area).replace(".", "_")
        output_dir = f"particle_images_{region_str}m_{self.image_pixel_size}p"
        if self.save_images:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        particle_cloud = data.collect()[0, self.particle_cloud_feature_name]
        image_records = []

        for message_number, message_data in enumerate(particle_cloud):
            time_stamp = message_data["time"]
            logger.debug(
                "Processing message %d at time %s",
                message_number,
                time_stamp,
            )

            particles_dict: dict[str, Any] = message_data["value"]
            list_of_particles = particles_dict["particles"]

            if not list_of_particles:
                logger.debug(
                    "Message %d has no particles. Skipping.",
                    message_number,
                )
                continue

            positions = np.array(
                [
                    [p["pose"]["position"]["x"], p["pose"]["position"]["y"]]
                    for p in list_of_particles
                ],
            )
            quats = np.array(
                [
                    [
                        p["pose"]["orientation"]["z"],
                        p["pose"]["orientation"]["w"],
                    ]
                    for p in list_of_particles
                ],
            )

            mean_pos = positions.mean(axis=0)
            mean_x, mean_y = mean_pos
            yaws = 2 * np.arctan2(quats[:, 0], quats[:, 1])
            mean_yaw = math.atan2(np.sin(yaws).mean(), np.cos(yaws).mean())

            # Rotate positions to align with mean yaw
            cos_val, sin_val = math.cos(-mean_yaw), math.sin(-mean_yaw)
            rotation_matrix = np.array(
                [[cos_val, -sin_val], [sin_val, cos_val]],
            )
            rotated_positions = (
                positions - mean_pos
            ) @ rotation_matrix.T + mean_pos

            # Filter positions within the cutting area
            mask = (
                np.abs(rotated_positions[:, 0] - mean_x) <= half_region
            ) & (np.abs(rotated_positions[:, 1] - mean_y) <= half_region)
            filtered_positions = rotated_positions[mask]

            if filtered_positions.size == 0:
                gray_image = np.full(
                    (self.image_pixel_size, self.image_pixel_size),
                    255,
                    dtype=np.uint8,
                )
            else:
                x_min, x_max = mean_x - half_region, mean_x + half_region
                y_min, y_max = mean_y - half_region, mean_y + half_region

                x_edges = np.linspace(x_min, x_max, self.image_pixel_size + 1)
                y_edges = np.linspace(y_min, y_max, self.image_pixel_size + 1)

                histogram, _, _ = np.histogram2d(
                    filtered_positions[:, 0],
                    filtered_positions[:, 1],
                    bins=[x_edges, y_edges],
                )

                # Adjust orientation to match image coordinates
                histogram = histogram.T[::-1, :]
                gray_image = np.where(histogram > 0, 0, 255).astype(np.uint8)

            if self.save_images:
                filename = (
                    Path(output_dir)
                    / f"particle_region_{message_number:04d}.png"
                )
                Image.fromarray(gray_image).save(filename)

            image_records.append(
                {"time": time_stamp, "value": gray_image.tolist()},
            )

        df_images = pl.DataFrame({"/particle_cloud_image": [image_records]})
        logger.debug("Processed images schema: %s", df_images.schema)
        return data.collect().hstack(df_images).lazy()
