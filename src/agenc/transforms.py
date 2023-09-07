"""
Transforms are used to transform the data before it is used in the learner.
This is useful for scaling the data, or for adding new features to the data.
"""

import polars as pl
from abc import ABC, abstractmethod


class Transform(ABC):
    """Base class for all transforms."""

    @abstractmethod
    def __call__(self, data: pl.DataFrame) -> pl.DataFrame:
        """Transform the data.

        Args:
            data (pl.DataFrame): The data to transform.

        Returns:
            pl.DataFrame: The transformed data.
        """


class StandardScaler(Transform):
    def __call__(self, data: pl.DataFrame) -> pl.DataFrame:
        return data.select((pl.all() - pl.all().mean()) / pl.all().std())


class SlidingWindow(Transform):
    """Transforms the data with a sliding window.

    The sliding window transform transforms the data by creating a sliding
    window over the row dimension. The data is then transformed by creating a
    new column for each column in the original data. The new columns are
    named by appending the index of the row in the sliding window to the
    original column name.
    As an example, consider the following data:

    .. list-table:: Original data
        :header-rows: 1

        *   - x
            - y
            - z
        *   - 1
            - 10
            - 100
        *   - 2
            - 20
            - 200
        *   - 3
            - 30
            - 300
        *   - 4
            - 40
            - 400
        *   - 5
            - 50
            - 500

    If we apply a sliding window with a window size of 3, we get the following

    .. list-table:: Transformed data
        :header-rows: 1

        *   - x_0
            - x_1
            - x_2
            - y_0
            - y_1
            - y_2
            - z_0
            - z_1
            - z_2
        *   - 1
            - 2
            - 3
            - 10
            - 20
            - 30
            - 100
            - 200
            - 300
        *   - 2
            - 3
            - 4
            - 20
            - 30
            - 40
            - 200
            - 300
            - 400
        *   - 3
            - 4
            - 5
            - 30
            - 40
            - 50
            - 300
            - 400
            - 500

    Args:
        window_size: size of the sliding window.
    """

    def __init__(self, window_size: int) -> None:
        self.window_size = window_size

    def __call__(self, data: pl.DataFrame) -> pl.DataFrame:
        columns = data.columns
        return (
            data.with_row_count()
            .select(pl.col("row_nr").cast(pl.Int32), pl.exclude("row_nr"))
            .groupby_rolling(
                "row_nr", period=f"{self.window_size}i", closed="right"
            )
            .agg(pl.exclude("row_nr"))
            .tail(-(self.window_size - 1))
            .select(
                [
                    pl.col(column).list.to_struct(
                        n_field_strategy="max_width",
                        fields=[
                            f"{column}_{i}" for i in range(self.window_size)
                        ],
                    )
                    for column in columns
                ]
            )
            .unnest(columns)
        )


class Select(Transform):
    """Select a subset of features.

    Args:
        features (list[str]): The features to select.
    """

    def __init__(self, features: list[str]) -> None:
        self.features = features

    def __call__(self, data: pl.DataFrame) -> pl.DataFrame:
        return data.select(self.features)
