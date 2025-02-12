import logging
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from flowcean.core.transform import Transform

logger = logging.getLogger(__name__)


class ParticleCloudImage(Transform):
    """Generates grayscale images from particle cloud data.

    Processes 2D particle cloud data to create grayscale images highlighting
    the particle distribution. Optionally saves images to disk and returns them
    embedded in a Polars DataFrame for further analysis.
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
            cutting_area: Square region side length for cropping (in meters).
            image_pixel_size: Output image resolution (both width and height).
            save_images: Whether to save images to disk.
        """
        self.particle_cloud_feature_name = particle_cloud_feature_name
        self.cutting_area = cutting_area
        self.image_pixel_size = image_pixel_size
        self.save_images = save_images

    def apply(self, data: pl.LazyFrame) -> pl.LazyFrame:
        logger.debug("Matching sampling rate of time series.")

        half_region = self.cutting_area / 2.0
        region_str = str(self.cutting_area).replace(".", "_")
        output_dir = f"particle_images_{region_str}m_{self.image_pixel_size}p"
        if self.save_images:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Collect the particle cloud data from the LazyFrame.
        particle_cloud = data.collect()[0, self.particle_cloud_feature_name]
        image_records = []

        for message_number, message_data in enumerate(
            particle_cloud,
        ):
            time_stamp = message_data["time"]
            print(f"Processing message {message_number} at time {time_stamp}")

            particles_dict = message_data["value"]
            list_of_particles = particles_dict["particles"]

            if not list_of_particles:
                print(f"Message {message_number} has no particles. Skipping.")
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
            # Compute yaw angles (rotation about z-axis)
            yaws = 2 * np.arctan2(quats[:, 0], quats[:, 1])
            # Compute mean position and circular mean of yaws
            mean_pos = positions.mean(axis=0)
            mean_x, mean_y = mean_pos
            mean_yaw = math.atan2(np.mean(np.sin(yaws)), np.mean(np.cos(yaws)))

            # Rotation by -mean_yaw
            rot_angle = -mean_yaw
            cos_val = math.cos(rot_angle)
            sin_val = math.sin(rot_angle)
            R = np.array([[cos_val, -sin_val], [sin_val, cos_val]])
            # Rotate all positions: subtract mean, rotate, then add mean back.
            rotated_positions = (positions - mean_pos) @ R.T + mean_pos

            # --- Filtering using a Boolean Mask ---
            mask = (
                np.abs(rotated_positions[:, 0] - mean_x) <= half_region
            ) & (np.abs(rotated_positions[:, 1] - mean_y) <= half_region)
            filtered_positions = rotated_positions[mask]

            # --- Plotting and Image Extraction ---
            fig, ax = plt.subplots(figsize=(1, 1), dpi=self.image_pixel_size)
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
            ax.axis("off")
            ax.set_xlim(mean_x - half_region, mean_x + half_region)
            ax.set_ylim(mean_y - half_region, mean_y + half_region)
            if filtered_positions.size > 0:
                # Plot all filtered points in one call.
                ax.plot(
                    filtered_positions[:, 0],
                    filtered_positions[:, 1],
                    "ko",
                    markersize=2,
                )

            if self.save_images:
                filename = (
                    Path(output_dir)
                    / f"particle_region_{message_number:04d}.png"
                )
                fig.savefig(filename, dpi=self.image_pixel_size)

            fig.canvas.draw()
            # Convert the figure to a NumPy array using the RGBA buffer.
            img_array = np.asarray(
                fig.canvas.buffer_rgba(),
            )  # shape: (image_pixel_size, image_pixel_size, 4)
            img_array = img_array[..., :3]  # Remove the alpha channel

            # Since image is strictly black-and-white, simply take one channel
            gray_image = img_array[..., 0].astype(np.uint8)
            plt.close(fig)

            image_records.append(
                {"time": time_stamp, "image": gray_image.tolist()},
            )

        plt.close("all")

        df_images = pl.DataFrame({"/particle_cloud_image": [image_records]})
        logger.debug(df_images)
        logger.debug(df_images.schema)
        return data.collect().hstack(df_images).lazy()

    def rotate_point(
        self,
        x: float,
        y: float,
        angle: float,
        origin: tuple[float, float],
    ) -> tuple[float, float]:
        """Rotate a point about given origin by specified angle (radians)."""
        ox, oy = origin
        dx = x - ox
        dy = y - oy
        rotated_x = dx * math.cos(angle) - dy * math.sin(angle)
        rotated_y = dx * math.sin(angle) + dy * math.cos(angle)
        return rotated_x + ox, rotated_y + oy
