import logging
from collections.abc import Iterable
from typing import override

import polars as pl

from flowcean.core import Transform

logger = logging.getLogger(__name__)


class OneHot(Transform):
    """Transforms integer features into a set of binary one-hot features.

    Transforms integer features into a set of binary one-hot features. The
    original integer features are dropped and are not part of the resulting
    data frame.

    As an example consider the following data

    feature |
    :------:|
     0      |
     1      |
     2      |
     1      |
     5      |

    When the one-hot transformation is applied, the result is as follows

    feature_0 | feature_1 | feature_2 | feature_5
    ----------|-----------|-----------|----------
     1        | 0         | 0         | 0
     0        | 1         | 0         | 0
     0        | 0         | 1         | 0
     0        | 1         | 0         | 0
     0        | 0         | 0         | 1
    """

    def __init__(self, features: Iterable[str]) -> None:
        """Initializes the One-Hot transform.

        Args:
            features: The features to transform.
        """
        self.features = features

    @override
    def transform(self, data: pl.DataFrame) -> pl.DataFrame:
        for feature in self.features:
            # Check if the feature is an integer feature
            if not data.schema[feature].is_integer():
                logger.error(
                    "Feature %s is of type %s but one-hot requires integer",
                    feature,
                    data.schema[feature],
                )
                continue
            data = data.with_columns(
                [
                    pl.col(feature)
                    .eq(entry)
                    .cast(pl.Int64)
                    .alias(f"{feature}_{entry}")
                    for entry in data.select(
                        pl.col(feature).unique()
                    ).to_series()
                ]
            ).drop(feature)

        return data
