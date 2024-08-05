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

    def __init__(
        self,
        features: Iterable[str],
        *,
        categories: dict[str, list[Any]] | None = None,
    ) -> None:
        """Initializes the One-Hot transform.

        Args:
            features: The features to transform.
            categories: Dictionary of features and a list of categorical values
                to encode for each. If set to None, the categories must be
                determined by calling `fit` with a sufficient sample of data.
        """
        self.features = features
        if categories is None:
            self.categories = {}
        else:
            self.categories = {
                feature: {f"{feature}_{value}": value for value in values}
                for feature, values in categories.items()
            }

    @override
    def fit(self, data: pl.DataFrame) -> None:
        # Derive categories from the data frame
        for feature in self.features:
            if data.schema[feature].is_float():
                logger.warning(
                    (
                        "Feature %s is of type float. Applying a one-hot",
                        "transform to it may produce undesired results.",
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
        if len(self.categories) == 0:
            raise NoCategoriesError
        for feature in self.features:
            data = data.with_columns(
                [
                    pl.col(feature).eq(value).cast(pl.Int64).alias(name)
                    for name, value in self.categories[feature].items()
                ]
            ).drop(feature)

        return data


class NoCategoriesError(Exception):
    pass
