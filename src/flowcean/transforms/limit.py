import logging
import math
from collections.abc import Iterable
from typing import override

import polars as pl

from flowcean.core import Transform

logger = logging.getLogger(__name__)


class Limit(Transform):
    """Limit time series feature lengths."""

    def __init__(
        self,
        *,
        features: Iterable[str] | None,
        t_start: float = 0.0,
        t_end: float = math.inf,
    ) -> None:
        """Initializes the Limit transform.

        Args:
            features: The features to limit. If `None`, all applicable features
                are limited.
            t_start: Limit start time. Defaults to zero. All data before this
                time will be removed from the time series.
            t_end: Limit end time. Defaults to infinite. All data after this
                time will be removed from the time series.
        """
        self.features = features
        self.t_start = t_start
        self.t_end = t_end

    @override
    def transform(self, data: pl.DataFrame) -> pl.DataFrame:
        return data.select(self.features)
