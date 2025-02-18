import logging
import math
from collections.abc import Iterable

import polars as pl
from typing_extensions import override

from flowcean.core import Transform
from flowcean.polars.is_time_series import is_timeseries_feature

logger = logging.getLogger(__name__)


class TimeWindow(Transform):
    """Limit time series to a certain time window."""

    def __init__(
        self,
        *,
        features: Iterable[str] | None = None,
        time_start: float = 0.0,
        time_end: float = math.inf,
    ) -> None:
        """Initializes the TimeWindow transform.

        Args:
            features: The features to apply this transformation to. If `None`,
                all applicable features will be affected.
            time_start: Window start time. Defaults to zero. All data before
                this time will be removed from the time series when applying
                the transform.
            time_end: Window end time. Defaults to infinite. All data after
                this time will be removed from the time series when applying
                the transform.
        """
        self.features = features
        self.t_start = time_start
        self.t_end = time_end

    @override
    def apply(self, data: pl.LazyFrame) -> pl.LazyFrame:
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
                            time_expression.le(self.t_end),
                        ),
                    ),
                ),
            )
        return data
