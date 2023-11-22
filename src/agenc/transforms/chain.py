import polars as pl
from typing_extensions import override

from agenc.core.transform import Transform


class Chain(Transform):
    """Chain multiple transforms together."""

    def __init__(self, *transforms: Transform):
        """Initialize the Chain.

        Args:
            transforms: The transforms to chain together.
        """
        self.transforms = transforms

    @override
    def __call__(self, data: pl.DataFrame) -> pl.DataFrame:
        for transform in self.transforms:
            data = transform(data)
        return data
