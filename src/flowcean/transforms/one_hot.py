import logging
from collections.abc import Iterable
from typing import Any, override

import polars as pl

from flowcean.core import Transform
from flowcean.core.learner import UnsupervisedLearner

logger = logging.getLogger(__name__)


class OneHot(Transform, UnsupervisedLearner):
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

    categories: dict[str, dict[str, Any]]

    def __init__(self, features: Iterable[str]) -> None:
        """Initializes the One-Hot transform.

        Args:
            features: The features to transform.
        """
        self.features = features
        self.categories = {}

    @override
    def fit(self, data: pl.DataFrame) -> None:
        # Derive categories from the data frame
        for feature in self.features:
            if data.schema[feature].is_float():
                logger.warning(
                    (
                        "Feature %s is of type float. Applying a one-hot",
                        "transform may produce undesired results.",
                        "Check your datatypes and transforms.",
                    ),
                    feature,
                )
            self.categories[feature] = {
                f"{feature}_{value}": value
                for value in data.select(pl.col(feature).unique()).to_series()
            }

    @override
    def transform(self, data: pl.DataFrame) -> pl.DataFrame:
        for feature in self.features:
            data = data.with_columns(
                [
                    pl.col(feature).eq(value).cast(pl.Int64).alias(name)
                    for name, value in self.categories[feature].items()
                ]
            ).drop(feature)

        return data
