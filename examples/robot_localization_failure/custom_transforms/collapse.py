import logging

import polars as pl
from typing_extensions import override

from flowcean.core.transform import Transform

logger = logging.getLogger(__name__)


class Collapse(Transform):
    """Collapse a time series feature into a single value."""

    def __init__(
        self,
        feature: str,
        element: int = 0,
    ) -> None:
        """Initialize the Collapse transform.

        Args:
            feature: The name of the feature to collapse.
            element: The index of the element to extract from the time series.
                Defaults to 0, which extracts the first element.
        """
        super().__init__()
        self.feature = feature
        self.element = element

    @override
    def apply(self, data: pl.LazyFrame) -> pl.LazyFrame:
        logger.debug(
            "Collapsing feature '%s' at element %d",
            self.feature,
            self.element,
        )
        return data.with_columns(
            pl.col(self.feature)
            .list.get(self.element)
            .struct.field("value")
            .alias(self.feature),
        )
