import logging

import numpy as np
import polars as pl

from flowcean.core.transform import Transform

logger = logging.getLogger(__name__)


class ImagesToTensor(Transform):
    def __init__(
        self,
        image_columns: list[str],
        height: int,
        width: int,
        tensor_column: str = "image_tensor",
    ) -> None:
        """Initialize the transform to convert image columns to a tensor.

        Args:
            image_columns: List of column names containing grayscale images.
            height: Height of each image in pixels.
            width: Width of each image in pixels.
            tensor_column: Name of the new column to store the tensor.
        """
        self.image_columns = image_columns
        self.height = height
        self.width = width
        self.num_channels = len(image_columns)
        self.tensor_column = tensor_column

    def apply(self, data: pl.LazyFrame) -> pl.LazyFrame:
        logger.debug("Applying ImagesToTensor transform")

        collected_data = data.collect()
        if collected_data.height == 0:
            return collected_data.lazy()

        # Verify all specified columns exist
        missing_cols = [
            col
            for col in self.image_columns
            if col not in collected_data.columns
        ]
        if missing_cols:
            error_msg = f"Columns {missing_cols} not found in DataFrame"
            raise ValueError(error_msg)

        # Ensure each image has the correct 2D shape
        for col in self.image_columns:
            # Extract the first value from the Series as a Python object
            sample_series = collected_data.get_column(col)
            sample = sample_series[0].to_list()  # Get the first element
            # Check if sample is a list and has the correct dimensions
            if not isinstance(sample, list):
                error_msg = (
                    f"Image in column '{col}' is not a list, got "
                    f"{type(sample)}"
                )
                raise TypeError(error_msg)
            if len(sample) != self.height:
                error_msg = (
                    f"Image in column '{col}' has height {len(sample)}, "
                    f"expected {self.height}"
                )
                raise ValueError(error_msg)
            if not all(
                isinstance(row, list) and len(row) == self.width
                for row in sample
            ):
                error_msg = (
                    f"Image in column '{col}' has inconsistent width, "
                    f"expected {self.width}"
                )
                raise ValueError(error_msg)

        # Process each row to create tensors
        tensor_list = []
        for row in collected_data.rows(named=True):
            # Extract images for this row from specified columns
            images = []
            for col in self.image_columns:
                image_2d = row[col]  # 2D list like [[255, 255, …], …]
                # Convert to numpy array directly
                image_np = np.array(image_2d, dtype=np.uint8)
                if image_np.shape != (self.height, self.width):
                    error_msg = (
                        f"Image in column '{col}' has shape {image_np.shape}, "
                        f"expected ({self.height}, {self.width})",
                    )
                    raise ValueError(
                        error_msg,
                    )
                images.append(image_np)

            # Stack images
            multi_channel_image = np.stack(
                images,
                axis=0,
            )  # Shape: [num_channels, height, width]
            # normalize to [0, 1]
            multi_channel_image = (
                multi_channel_image.astype(np.float32) / 255.0
            )
            tensor_list.append(multi_channel_image)



        # Add the tensor list as a new column with Object dtype
        result_df = collected_data.with_columns(
            pl.Series(self.tensor_column, tensor_list, dtype=pl.Object),
        ).drop_nulls()

        return result_df.lazy()
