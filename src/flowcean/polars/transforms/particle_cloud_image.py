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

    Processes 2D particle cloud data to create grayscale images
    highlighting the particle distribution. Optionally saves images to disk
    and returns them embedded in a Polars DataFrame for further analysis.
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

        # Collect the particle cloud data from the LazyFrame
        particle_cloud = data.collect()[0, self.particle_cloud_feature_name]
        image_records = []

        fig, ax = plt.subplots(figsize=(1, 1), dpi=self.image_pixel_size)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        ax.axis("off")

        # Process a slice of messages (adjust indices as needed)
        for message_number, message_data in enumerate(
            particle_cloud[0:9],
            start=0,
        ):
            time_stamp = message_data["time"]
            print(f"Processing message {message_number} at time {time_stamp}")

            particles_dict = message_data["value"]
            list_of_particles = particles_dict["particles"]

            if not list_of_particles:
                print(f"Message {message_number} has no particles. Skipping.")
                continue

            # Compute Mean Position and Mean Orientation
            sum_x, sum_y = 0.0, 0.0
            sum_sin, sum_cos = 0.0, 0.0
            num_particles = len(list_of_particles)
            particles_data = []  # To store each particle's x, y, and yaw

            for particle in list_of_particles:
                pos = particle["pose"]["position"]
                x = pos["x"]
                y = pos["y"]
                quat = particle["pose"]["orientation"]
                # Compute yaw (rotation about z-axis)
                yaw = 2 * math.atan2(quat["z"], quat["w"])
                particles_data.append({"x": x, "y": y, "yaw": yaw})
                sum_x += x
                sum_y += y
                sum_sin += math.sin(yaw)
                sum_cos += math.cos(yaw)

            mean_x = sum_x / num_particles
            mean_y = sum_y / num_particles
            mean_yaw = math.atan2(sum_sin, sum_cos)

            # Rotate Particle Data by -mean_yaw About the Mean Position
            rotated_particles = []
            for p in particles_data:
                x, y, yaw = p["x"], p["y"], p["yaw"]
                new_x, new_y = self.rotate_point(
                    x,
                    y,
                    -mean_yaw,
                    (mean_x, mean_y),
                )
                new_yaw = yaw - mean_yaw
                rotated_particles.append(
                    {"x": new_x, "y": new_y, "yaw": new_yaw},
                )

            # Filter Out Particles Outside the Square Region
            filtered_particles = [
                p
                for p in rotated_particles
                if abs(p["x"] - mean_x) <= half_region
                and abs(p["y"] - mean_y) <= half_region
            ]

            # Create the Image
            ax.cla()  # Clear axes for new plot.
            ax.axis("off")
            ax.set_xlim(mean_x - half_region, mean_x + half_region)
            ax.set_ylim(mean_y - half_region, mean_y + half_region)

            # Plot all filtered particles as black dots
            for p in filtered_particles:
                ax.plot(p["x"], p["y"], "ko", markersize=2)

            if self.save_images:
                filename = (
                    Path(output_dir)
                    / f"particle_region_{message_number:04d}.png"
                )
                fig.savefig(filename, dpi=self.image_pixel_size)

            # Convert the figure to a NumPy array using the RGBA buffer
            fig.canvas.draw()
            img_array = np.asarray(fig.canvas.buffer_rgba())
            img_array = img_array[..., :3]  # Remove alpha channel

            # Convert to grayscale using luminance conversion
            gray_image = np.dot(img_array, [0.2989, 0.5870, 0.1140]).astype(
                np.uint8,
            )
            image_records.append(
                {"time": time_stamp, "image": gray_image.tolist()},
            )

        plt.close(fig)

        df_images = pl.DataFrame({"/particle_cloud_image": [image_records]})

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
