import logging
from collections.abc import Callable
from typing import override

import polars as pl

from flowcean.core.transform import Transform

logger = logging.getLogger(__name__)


class Lambda(Transform):
    """Apply a custom function to the data of an environment."""

    def __init__(self, fn: Callable[[pl.DataFrame], pl.DataFrame]) -> None:
        """Initializes the Lambda transform.

        Args:
            fn: Function handle to be applied to the data.
        """
        self.fn = fn

    @override
    def apply(self, data: pl.DataFrame) -> pl.DataFrame:
        return self.fn(data)
