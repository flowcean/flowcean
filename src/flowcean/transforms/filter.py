import polars as pl
from flowcean.core.transform import Transform
from typing import override



class Filter(Transform):
    """Filter rows based on a condition."""

    def __init__(self, condition) -> None:
        """Initializes the Filter transform.

        Args:
            condition: A lambda function that takes a DataFrame and returns a boolean Series.
        """
        # super().__init__()
        self.condition = condition

    @override
    def apply(self, data: pl.DataFrame) -> pl.DataFrame:
        return data.filter(self.condition(data))