import logging
from collections.abc import Iterable
from typing import Any, Self, override

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

    feature_categrorie_mapping: dict[str, dict[str, Any]]

    def __init__(
        self,
        feature_categories: dict[str, list[Any]],
    ) -> None:
        """Initializes the One-Hot transform.

        Args:
            feature_categories: Dictionary of features and a list of
                categorical values to encode for each. If set to None, the
                categories must be determined by calling `fit` with a
                sufficient sample of data.
        """
        self.feature_categrorie_mapping = {
            feature: {f"{feature}_{value}": value for value in values}
            for feature, values in feature_categories.items()
        }

    @override
    def transform(self, data: pl.DataFrame) -> pl.DataFrame:
        if len(self.feature_categrorie_mapping) == 0:
            raise NoCategoriesError
        for (
            feature,
            category_mappings,
        ) in self.feature_categrorie_mapping.items():
            data = data.with_columns(
                [
                    pl.col(feature).eq(value).cast(pl.Int64).alias(name)
                    for name, value in category_mappings.items()
                ]
            ).drop(feature)

        return data

    @classmethod
    def from_dataframe(
        cls, data: pl.DataFrame, features: Iterable[str]
    ) -> Self:
        # Derive categories from the data frame
        feature_categories: dict[str, list[Any]] = {}
        for feature in features:
            if data.schema[feature].is_float():
                logger.warning(
                    (
                        "Feature %s is of type float. Applying a one-hot",
                        "transform to it may produce undesired results.",
                        "Check your datatypes and transforms.",
                    ),
                    feature,
                )
            feature_categories[feature] = (
                data.select(pl.col(feature).unique()).to_series().to_list()
            )
        return cls(feature_categories)


class NoCategoriesError(Exception):
    pass
