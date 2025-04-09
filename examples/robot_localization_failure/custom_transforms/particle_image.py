import bisect
import logging
import math
from pathlib import Path

import numpy as np
import polars as pl
from PIL import Image
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from flowcean.core.transform import Transform

logger = logging.getLogger(__name__)


class ParticleImage(Transform):
    """Generates grayscale images from particle cloud data in the map frame.

    The particles from the `/particle_cloud` topic are transformed using the
    latest sensor pose from `/amcl_pose` so that the robot is centered at the
    origin, with its forward (x-axis) pointing upward. A 2D histogram of
    the transformed particle positions is computed over a square region
    (of side length crop_region_size in meters) and normalized to a grayscale
    image where pixel intensity reflects particle density.
    The image is resized to (image_pixel_size x image_pixel_size) pixels.
    Images are optionally saved to disk and embedded in the output
    Polars DataFrame under the feature `/particle_image`.
    """

    def __init__(
        self,
        particle_topic: str = "/particle_cloud",
        amcl_pose_topic: str = "/amcl_pose",
        crop_region_size: float = 15.0,
        image_pixel_size: int = 200,
        *,
        save_images: bool = False,
    ) -> None:
        """Initialize the ParticleImage transform.

        Args:
            particle_topic: Topic name for the particle cloud data.
            amcl_pose_topic: Topic name for pose data (e.g., `/amcl_pose`).
            crop_region_size: Side length (in m) of the square region.
            image_pixel_size: Output image resolution.
            save_images: Whether to save images to disk.
        """
        self.particle_topic = particle_topic
        self.amcl_pose_topic = amcl_pose_topic
        self.crop_region_size = crop_region_size
        self.image_pixel_size = image_pixel_size
        self.save_images = save_images

    def _transform_particles_to_sensor_frame(
        self,
        particle_data: dict,
        robot_pose: tuple[float, float, float],
    ) -> np.ndarray:
        """Transform particle cloud to sensor frame and generate grayscale img.

        Args:
            particle_data:  A dictionary corresponding to
                            one `/particle_cloud` message.
                            Expected to have a "time" key and a "value" key
                            containing a "particles" list.
            robot_pose:     A tuple containing the robot's pose
                            (x, y, theta) in the map frame.

        Returns:
            np.ndarray:     A generated grayscale image (single channel)
                            as a NumPy array.
        """
        x_r, y_r, theta_r = robot_pose

        # Compute rotation offset so that robot x-axis points upward.
        # Desired offset = π/2 - theta_r.
        offset = math.pi / 2 - theta_r

        particles = particle_data["value"]["particles"]
        particle_xs = []
        particle_ys = []
        for particle in particles:
            pos = particle["pose"]["position"]
            particle_xs.append(pos["x"])
            particle_ys.append(pos["y"])

        # Translate and rotate particle positions:
        # Translate by subtracting robot pose, then rotate by the offset.
        rotated_particle_xs = []
        rotated_particle_ys = []
        for x, y in zip(particle_xs, particle_ys, strict=False):
            dx = x - x_r
            dy = y - y_r
            new_x = dx * math.cos(offset) - dy * math.sin(offset)
            new_y = dx * math.sin(offset) + dy * math.cos(offset)
            rotated_particle_xs.append(new_x)
            rotated_particle_ys.append(new_y)

        half_region = self.crop_region_size / 2.0
        x_min, x_max = -half_region, half_region
        y_min, y_max = -half_region, half_region

        # Create bin edges for a 2D histogram.
        x_edges = np.linspace(x_min, x_max, self.image_pixel_size + 1)
        y_edges = np.linspace(y_min, y_max, self.image_pixel_size + 1)

        # Create a 2D histogram of the transformed particles.
        x_array = np.array(rotated_particle_xs)
        y_array = np.array(rotated_particle_ys)
        histogram, _, _ = np.histogram2d(
            x_array,
            y_array,
            bins=(np.asarray(x_edges), np.asarray(y_edges)),
        )

        # Rotate histogram 90° clockwise so robot's forward axis point upward.
        rotated_histogram = np.rot90(histogram, k=1)

        # Normalize the histogram:
        # Bins with maximum count will map to dark (0)
        # while empty bins become white (255).
        if rotated_histogram.max() > 0:
            normalized = rotated_histogram / rotated_histogram.max()
            gray_image = (255 - (normalized * 255)).astype(np.uint8)
        else:
            gray_image = np.full(
                (self.image_pixel_size, self.image_pixel_size),
                255,
                dtype=np.uint8,
            )

        return gray_image[:, ::-1]

    def apply(self, data: pl.LazyFrame) -> pl.LazyFrame:
        """Apply the transformation to generate particle images.

        Args:
            data:   A Polars LazyFrame containing the topics for
                    particle cloud and sensor pose.

        Returns:
            pl.LazyFrame:   An updated DataFrame with a new
                            column `/particle_image` embedding
                            the generated images.
        """
        logger.debug("Processing particle cloud data to generate images.")

        region_str = str(self.crop_region_size).replace(".", "_")
        output_dir = f"particle_images_{region_str}m_{self.image_pixel_size}p"
        if self.save_images:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        collected_data = data.collect()
        particle_data = collected_data[0, self.particle_topic]
        sensor_pose_data = collected_data[0, self.amcl_pose_topic]

        # Build a list of pose entries from sensor pose data.
        pose_entries = []
        for entry in sensor_pose_data:
            pose_data = entry["value"]
            x = pose_data["pose.pose.position.x"]
            y = pose_data["pose.pose.position.y"]
            orientation_x = pose_data["pose.pose.orientation.x"]
            orientation_y = pose_data["pose.pose.orientation.y"]
            orientation_z = pose_data["pose.pose.orientation.z"]
            orientation_w = pose_data["pose.pose.orientation.w"]
            # Extract yaw (theta) in radians.
            theta = Rotation.from_quat(
                [orientation_x, orientation_y, orientation_z, orientation_w],
            ).as_euler("xyz", degrees=False)[2]
            pose_entries.append((entry["time"], x, y, theta))
        # Sort pose entries by time.
        pose_entries.sort(key=lambda e: e[0])
        pose_times = [entry[0] for entry in pose_entries]

        image_records = []
        for message_number, message_data in enumerate(
            tqdm(particle_data, desc="Generating particle images"),
        ):
            timestamp = message_data["time"]
            logger.debug(
                "Processing particle message %d at time %s",
                message_number,
                timestamp,
            )

            # Find the latest sensor pose before particle message timestamp.
            idx = bisect.bisect_right(pose_times, timestamp) - 1
            if idx < 0:
                logger.warning(
                    "No sensor pose found before timestamp %s, skipping.",
                    timestamp,
                )
                continue
            _, x_robot, y_robot, theta_robot = pose_entries[idx]

            gray_image = self._transform_particles_to_sensor_frame(
                message_data,
                (x_robot, y_robot, theta_robot),
            )

            if self.save_images:
                filename = (
                    Path(output_dir)
                    / f"particle_image_{message_number:04d}.png"
                )
                Image.fromarray(gray_image, mode="L").save(filename)

            image_records.append(
                {"time": timestamp, "value": gray_image.tolist()},
            )

        df_images = pl.DataFrame({"/particle_image": [image_records]})
        logger.debug("Processed particle images schema: %s", df_images.schema)
        return collected_data.hstack(df_images).lazy()
