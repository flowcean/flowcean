import logging
from pathlib import Path

import numpy as np
import polars as pl
from PIL import Image
from tqdm import tqdm

from flowcean.core.transform import Transform

logger = logging.getLogger(__name__)


class ScanImage(Transform):
    """Generates grayscale images from laser scan data in sensor coordinates.

    Processes 2D laser scan data to create grayscale images centered around the
    robot, using scan points in the sensor coordinate system. Optionally saves
    images to disk and embeds them in a Polars DataFrame under the feature name
    `/scan_image`. The images are cropped to a square region with a specified
    side length.
    """

    def __init__(
        self,
        scan_points_sensor_topic: str = "scan_points_sensor",
        crop_region_size: float = 15.0,
        image_pixel_size: int = 300,
        *,
        save_images: bool = False,
    ) -> None:
        """Initialize the ScanImage transform.

        Args:
            scan_points_sensor_topic: Topic name for scan points in
                sensor coordinates.
            crop_region_size: Side length of square region for cropping.
            image_pixel_size: Output image resolution (both width and height).
            save_images: Whether to save images to disk.
        """
        self.crop_region_size = crop_region_size
        self.image_pixel_size = image_pixel_size
        self.save_images = save_images
        self.scan_points_sensor_topic = scan_points_sensor_topic

    def apply(self, data: pl.LazyFrame) -> pl.LazyFrame:
        logger.debug("Processing scan data to generate images.")

        half_region = self.crop_region_size / 2.0
        region_str = str(self.crop_region_size).replace(".", "_")
        output_dir = f"scan_images_{region_str}m_{self.image_pixel_size}p"
        if self.save_images:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Collect data
        collected_data = data.collect()
        scan_data = collected_data[
            0,
            self.scan_points_sensor_topic,
        ]  # Sensor-frame points

        image_records = []
        for message_number, message_data in enumerate(
            tqdm(scan_data, desc="Generating scan images"),
        ):
            timestamp = message_data["time"]
            logger.debug(
                "Processing message %d at time %s",
                message_number,
                timestamp,
            )

            # Extract scan points in sensor coordinates (centered at robot)
            scan_points_sensor = np.array(message_data["value"])

            # Filter points within the cutting area
            mask = (np.abs(scan_points_sensor[:, 0]) <= half_region) & (
                np.abs(scan_points_sensor[:, 1]) <= half_region
            )
            filtered_positions = scan_points_sensor[mask]

            if filtered_positions.size == 0:
                gray_image = np.full(
                    (self.image_pixel_size, self.image_pixel_size),
                    255,  # White background
                    dtype=np.uint8,
                )
            else:
                # Define the cutting area bounds in sensor coordinates
                x_min, x_max = -half_region, half_region
                y_min, y_max = -half_region, half_region

                # Create histogram edges
                x_edges = np.linspace(x_min, x_max, self.image_pixel_size + 1)
                y_edges = np.linspace(y_min, y_max, self.image_pixel_size + 1)

                # Generate 2D histogram
                histogram, _, _ = np.histogram2d(
                    filtered_positions[:, 0],
                    filtered_positions[:, 1],
                    bins=[x_edges, y_edges],
                )

                # Rotate 90° clockwise so sensor x (forward) points up
                # Then flip horizontally so sensor y (left) points left
                rotated_histogram = np.rot90(histogram, k=1)  # 90° CW
                gray_image = np.where(rotated_histogram > 0, 0, 255).astype(
                    np.uint8,
                )
            if self.save_images:
                filename = (
                    Path(output_dir) / f"scan_image_{message_number:04d}.png"
                )
                Image.fromarray(gray_image).save(filename)

            image_records.append(
                {"time": timestamp, "value": gray_image.tolist()},
            )

        df_images = pl.DataFrame({"/scan_image": [image_records]})
        logger.debug("Processed images schema: %s", df_images.schema)
        return collected_data.hstack(df_images).lazy()
