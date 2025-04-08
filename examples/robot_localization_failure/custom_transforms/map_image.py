import bisect
import logging
from pathlib import Path

import cv2
import numpy as np
import polars as pl
from PIL import Image
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from flowcean.core.transform import Transform

logger = logging.getLogger(__name__)


class MapImage(Transform):
    """Generates grayscale images from an occupancy map in the robot's frame.

    Transforms the occupancy map from `/map` into the robot's sensor coordinate
    frame using the latest pose from `/amcl_pose`, with sensor x (forward)
    pointing upwards and sensor y (left) pointing rightward to match RViz.
    The map is cropped around the robot and resized to the specified pixel
    size. Images are optionally saved to disk and embedded in a Polars
    DataFrame under `/map_image`.
    """

    def __init__(
        self,
        map_topic: str = "/map",
        sensor_pose_topic: str = "/amcl_pose",
        crop_region_size: float = 15.0,
        image_pixel_size: int = 300,
        *,
        save_images: bool = False,
    ) -> None:
        """Initialize the MapImage transform.

        Args:
            map_topic: Topic name for the occupancy map data.
            sensor_pose_topic: Topic name for the sensor pose data.
            crop_region_size: Side length of square region for cropping (m).
            image_pixel_size: Output image resolution (both width and height).
            save_images: Whether to save images to disk.
        """
        self.map_topic = map_topic
        self.sensor_pose_topic = sensor_pose_topic
        self.crop_region_size = crop_region_size
        self.image_pixel_size = image_pixel_size
        self.save_images = save_images

    def _transform_map_to_sensor_frame(
        self,
        map_data: dict,
        robot_pose: tuple[float, float, float],
    ) -> np.ndarray:
        """Transform the occupancy map to the robot's sensor frame.

        Args:
            map_data: Dictionary containing map data and metadata.
            robot_pose: Tuple of (x, y, theta) representing robot position and
                orientation.

        Returns:
            np.ndarray: Transformed grayscale image.
        """
        # Extract map info
        map_array = np.array(map_data["data"]).reshape(
            map_data["info.height"],
            map_data["info.width"],
        )
        resolution = map_data["info.resolution"]
        origin_x = map_data["info.origin.position.x"]
        origin_y = map_data["info.origin.position.y"]

        # Convert to grayscale: -1 to 255, 0 to 255, 100 to 0
        gray_map = np.where(
            map_array == -1,
            255,
            255 - (map_array * 2.55).astype(np.uint8),
        )

        # Robot pose
        x_r, y_r, theta_r = robot_pose  # theta_r in radians

        # Robot position in pixel coordinates
        robot_px_x = (x_r - origin_x) / resolution
        robot_px_y = (y_r - origin_y) / resolution

        # Output image size in pixels before resizing
        pixels_per_side = int(self.crop_region_size / resolution)

        # Affine transformation: rotate around robot, then center it
        theta_deg = np.degrees(theta_r) + 180
        affine_transformation_matrix = cv2.getRotationMatrix2D(
            (robot_px_x, robot_px_y),
            theta_deg,
            1.0,
        )
        affine_transformation_matrix = affine_transformation_matrix.astype(
            np.float32,
        )  # Ensure float32 for OpenCV compatibility

        # Translate robot to center
        center_px = pixels_per_side / 2.0
        affine_transformation_matrix[0, 2] += center_px - robot_px_x
        affine_transformation_matrix[1, 2] += center_px - robot_px_y

        # Warp the map
        warped_map = cv2.warpAffine(
            gray_map,
            affine_transformation_matrix,
            (pixels_per_side, pixels_per_side),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255,),  # Scalar tuple for grayscale
        )

        # Resize to final image size
        return cv2.resize(
            warped_map,
            (self.image_pixel_size, self.image_pixel_size),
            interpolation=cv2.INTER_NEAREST,
        )

    def apply(self, data: pl.LazyFrame) -> pl.LazyFrame:
        """Apply the transformation to generate map images.

        Args:
            data: LazyFrame containing map and pose data.

        Returns:
            pl.LazyFrame: Updated DataFrame with map images.
        """
        logger.debug("Processing map data to generate images.")

        region_str = str(self.crop_region_size).replace(".", "_")
        output_dir = f"map_images_{region_str}m_{self.image_pixel_size}p"
        if self.save_images:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        collected_data = data.collect()
        map_data = collected_data[0, self.map_topic]
        amcl_pose_data = collected_data[0, self.sensor_pose_topic]

        # Extract pose data
        pose_entries = []
        for entry in amcl_pose_data:
            pose_data = entry["value"]
            x = pose_data["pose.pose.position.x"]
            y = pose_data["pose.pose.position.y"]
            orientation_x = pose_data["pose.pose.orientation.x"]
            orientation_y = pose_data["pose.pose.orientation.y"]
            orientation_z = pose_data["pose.pose.orientation.z"]
            orientation_w = pose_data["pose.pose.orientation.w"]
            theta = Rotation.from_quat(
                [orientation_x, orientation_y, orientation_z, orientation_w],
            ).as_euler("xyz", degrees=False)[2]
            pose_entries.append((entry["time"], x, y, theta))
        pose_times = [entry[0] for entry in pose_entries]

        # Process each map message
        image_records = []
        for message_number, message_data in enumerate(
            tqdm(map_data, desc="Generating map images"),
        ):
            timestamp = message_data["time"]
            logger.debug(
                "Processing map message %d at time %s",
                message_number,
                timestamp,
            )

            # Find the latest pose before the map timestamp
            idx = bisect.bisect_right(pose_times, timestamp) - 1
            if idx < 0:
                logger.warning(
                    "No pose found before timestamp %s, skipping.",
                    timestamp,
                )
                continue
            _, x_robot, y_robot, theta_robot = pose_entries[idx]

            # Transform map to sensor frame
            map_value = message_data["value"]
            transformed_image = self._transform_map_to_sensor_frame(
                map_value,
                (x_robot, y_robot, theta_robot),
            )

            # Flip horizontally to match RViz orientation
            gray_image = transformed_image[:, ::-1]

            # Save image to disk if enabled
            if self.save_images:
                filename = (
                    Path(output_dir) / f"map_image_{message_number:04d}.png"
                )
                Image.fromarray(gray_image).save(filename)

            # Store image data
            image_records.append(
                {"time": timestamp, "value": gray_image.tolist()},
            )

        # Create DataFrame with images and combine with original data
        df_images = pl.DataFrame({"/map_image": [image_records]})
        logger.debug("Processed map images schema: %s", df_images.schema)
        return collected_data.hstack(df_images).lazy()
