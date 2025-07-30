import logging
from collections.abc import Iterable
from typing import cast

import polars as pl
from typing_extensions import override

from flowcean.core import Transform
from flowcean.polars.time_series_type import get_time_series_value_type

logger = logging.getLogger(__name__)


class Mode(Transform):
    """Mode finds the value that appears most often in time-series features."""

    def __init__(
        self,
        features: str | Iterable[str],
        *,
        replace: bool = False,
    ) -> None:
        """Initializes the Mode transform.

        Args:
            features: The features to apply this transform to.
            replace: Whether to replace the original features with the
                transformed ones. If set to False, the default, the value will
                be added as a new feature named `{feature}_mode`.
        """
        self.features = [features] if isinstance(features, str) else features
        self.replace = replace

    @override
    def apply(self, data: pl.LazyFrame) -> pl.LazyFrame:
        schema = data.collect_schema()
        for feature in self.features:
            # Check if the feature is a floating point number and issue a
            # warning as the mode is not well defined for those.
            time_series_type = get_time_series_value_type(
                cast("pl.DataType", schema.get(feature)),
            )
            if time_series_type in (pl.Float32, pl.Float64):
                logger.warning(
                    "Feature %s is a floating point number. "
                    "The mode is not well defined for floating point numbers.",
                    feature,
                )

            expr = (
                pl.col(feature)
                .list.eval(pl.element().struct.field("value"))
                # Unfortunately, `mode` is not implemented for lists, so we
                # have to use `map_elements` as a workaround.
                .map_elements(
                    lambda x: x.mode().max(),
                    return_dtype=time_series_type,
                )
            )
            data = data.with_columns(
                expr if self.replace else expr.alias(f"{feature}_mode"),
            )
        return data
