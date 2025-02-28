import logging
from collections.abc import Callable

import polars as pl
from typing_extensions import override

from flowcean.core import Transform

logger = logging.getLogger(__name__)


class Lambda(Transform):
    """Apply a custom function to the data of an environment."""

    def __init__(self, fn: Callable[[pl.LazyFrame], pl.LazyFrame]) -> None:
        """Initializes the Lambda transform.

        Args:
            fn: Function handle to be applied to the data.
        """
        self.fn = fn

    @override
    def apply(self, data: pl.LazyFrame) -> pl.LazyFrame:
        return self.fn(data)
