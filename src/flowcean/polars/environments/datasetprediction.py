from typing import override

import polars as pl

from flowcean.core import ActiveEnvironment, Finished

from .dataframe import DataFrame


class DatasetPredictionEnvironment(
    ActiveEnvironment[pl.LazyFrame, pl.DataFrame]
):
    """Dataset prediction environment."""

    environment: DataFrame
    batch_size: int
    data: pl.LazyFrame | None = None
    slice: pl.LazyFrame | None = None
    i: int = 0

    def __init__(
        self,
        environment: DataFrame,
        batch_size: int,
    ) -> None:
        """Initialize the dataset prediction environment.

        Args:
            environment: The dataset to use for prediction.
            batch_size: The batch size of the prediction.
        """
        super().__init__()
        if batch_size <= 0:
            message = "batch_size must be positive"
            raise ValueError(message)
        self.environment = environment
        self.batch_size = batch_size

    @override
    def _observe(self) -> pl.LazyFrame:
        if self.data is None:
            self.data = self.environment.observe()
        if self.slice is None:
            batch = self.data.slice(self.i, self.batch_size)
            if batch.head(1).collect().is_empty():
                raise Finished
            self.slice = batch
        print("Provided input for prediction is: ", self.slice)
        return self.slice.lazy()

    @override
    def step(self) -> None:
        if self.data is None:
            self.data = self.environment.observe()
        self.i += self.batch_size
        self.slice = None
        batch = self.data.slice(self.i, self.batch_size)
        if batch.head(1).collect().is_empty():
            raise Finished
        self.slice = batch

    @override
    def act(self, action: pl.DataFrame) -> None:
        print("The predicted output is: ", action)
