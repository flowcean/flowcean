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
