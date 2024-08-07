import logging
from typing import override

import polars as pl

from flowcean.core import Transform

logger = logging.getLogger(__name__)


class ToLazy(Transform):
    """Convert data to a lazy representation.

    Convert data into a lazy representation that can speed up subsequent
    transformations by optimizing them before execution. Most of the flowceans
    API supports lazy data. If a particular transform does not, use a
    `Collect' transform to convert the data back to a materialized
    representation. Multiple calls to this transform will not degrade
    performance.
    """

    def __init__(self) -> None:
        """Initializes the ToLazy transform."""
        super().__init__()

    @override
    def transform(
        self, data: pl.DataFrame | pl.LazyFrame
    ) -> pl.DataFrame | pl.LazyFrame:
        return data.lazy()
