from typing import cast

import polars as pl
from typing_extensions import override

from flowcean.core import ActiveEnvironment, Finished

from .dataframe import DataFrame


class DatasetPredictionEnvironment(ActiveEnvironment):
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
        self.environment = environment
        self.batch_size = batch_size

    @override
    def _observe(self) -> pl.LazyFrame:
        if self.data is None:
            self.data = cast(pl.LazyFrame, self.environment.observe())
        if self.slice is None:
            self.slice = self.data.slice(self.i, self.batch_size)
        print("Provided input for prediction is: ", self.slice)
        return self.slice.lazy()

    @override
    def step(self) -> None:
        if self.data is None:
            self.data = cast(pl.LazyFrame, self.environment.observe())
        self.i += self.batch_size
        self.slice = self.data.slice(self.i, self.batch_size)
        if (
            self.slice.slice(0, 1)
            .collect(streaming=False)
            .select(pl.len())
            .item(0, 0)
            == 0
        ):
            raise Finished

    @override
    def act(self, action: pl.DataFrame) -> None:
        print("The predicted output is: ", action)
