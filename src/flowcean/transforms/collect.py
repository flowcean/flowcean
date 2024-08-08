import logging
from typing import override

import polars as pl

from flowcean.core import Transform

logger = logging.getLogger(__name__)


class Collect(Transform):
    """Convert data to a materialized representation.

    Materialize data by performing any queued transform on it. Operations are
    optimized before execution for improved performance.
    """

    def __init__(self) -> None:
        """Initializes the Collect transform."""
        super().__init__()

    @override
    def transform(
        self, data: pl.DataFrame | pl.LazyFrame
    ) -> pl.DataFrame | pl.LazyFrame:
        return data if isinstance(data, pl.DataFrame) else data.collect()
