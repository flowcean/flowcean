import logging
import math
from collections.abc import Iterable
from typing import override

import polars as pl

from flowcean.core import Transform
from flowcean.utils import is_timeseries_feature

logger = logging.getLogger(__name__)


class TimeWindow(Transform):
    """Limit time series to a certain windows."""

    def __init__(
        self,
        *,
        features: Iterable[str] | None = None,
        t_start: float = 0.0,
        t_end: float = math.inf,
    ) -> None:
        """Initializes the TimeWindow transform.

        Args:
            features: The features to limit. If `None`, all applicable features
                are limited.
            t_start: Window start time. Defaults to zero. All data before this
                time will be removed from the time series when applying the
                transform.
            t_end: Window end time. Defaults to infinite. All data after this
                time will be removed from the time series when applying the
                transform.
        """
        self.features = features
        self.t_start = t_start
        self.t_end = t_end

    @override
    def transform(self, data: pl.DataFrame) -> pl.DataFrame:
        for feature in (
            self.features
            if self.features is not None
            else [
                feature
                for feature in data.collect_schema().names()
                if is_timeseries_feature(data, feature)
            ]
        ):
            time_expression = (
                pl.element().struct.field("time").cast(pl.Float64)
            )

            data = data.with_columns(
                pl.col(feature).list.eval(
                    pl.element().filter(
                        time_expression.ge(self.t_start).and_(
                            time_expression.le(self.t_end)
                        )
                    )
                )
            )
        return data
