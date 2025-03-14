import logging

import polars as pl
from typing_extensions import override

from flowcean.core import Transform

logger = logging.getLogger(__name__)


class Explode(Transform):
    """Explodes a Dataframe to long format by exploding the given features.

    If no features are specified, all columns will be exploded.

    Args:
        features (list[str] | None): List of features to explode. If None, all
            columns are exploded.

    The below example shows the usage of a `Explode` transform in an
    `experiment.yaml` file. Assuming the loaded data is represented by the
    table:

    time          | feature_a    | feature_b    | constant
    --------------|--------------|--------------|---------
     [0, 1]       | [2, 1]       | [9, 3]       | 1
     [0, 2]       | [3, 4]       | [8, 4]       | 2

    The resulting Dataframe after the transform is:

    time          | feature_a    | feature_b    | constant
    --------------|--------------|--------------|---------
     0            | 2            | 9            | 1
     1            | 1            | 3            | 1
     0            | 3            | 8            | 2
     2            | 4            | 4            | 2
    """

    def __init__(self, features: list[str] | None = None) -> None:
        """Initialize the Explode transform.

        Args:
            features: List of column names to explode, or None to explode all
                columns.
        """
        self.features = features

    @override
    def apply(self, data: pl.LazyFrame) -> pl.LazyFrame:
        """Apply the explosion transformation to the DataFrame.

        Args:
            data: Input LazyFrame to transform.

        Returns:
            Transformed LazyFrame with exploded columns.
        """
        logger.debug("Exploding timeseries")
        # If features is None, use all column names from the schema
        features_to_explode = (
            self.features
            if self.features is not None
            else data.collect_schema().names()
        )
        return data.explode(features_to_explode)
