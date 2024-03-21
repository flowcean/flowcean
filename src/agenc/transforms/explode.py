import logging

import polars as pl
from typing_extensions import override

from agenc.core import Transform

logger = logging.getLogger(__name__)


class Explode(Transform):
    """Explodes a Dataframe to long format by exploding the given features.

    Args:
        features (list[str]): List of features to explode.

    The below example shows the usage of a `Explode` transform in an
    `experiment.yaml` file. Assuming the loaded data is represented by the
    table:

    .. list-table:: Original data
        :header-rows: 1

        *   - time
            - feature_a
            - feature_b
            - constant
        *   - [0, 1, 2, 3]
            - [2, 1, 7, 2]
            - [9, 3, 5, 0]
            - 1
        *   - [0, 2, 4, 6]
            - [3, 4, 1, 0]
            - [8, 4, 7, 2]
            - 2

    This transform can be used to explode the columns `time`,
    `feature_a`, and `feature_b`.

    The resulting Dataframe after the transform is:

    .. list-table:: Transformed data
        :header-rows: 1

        *   - time
            - feature_a
            - feature_b
            - constant
        *   - 0
            - 2
            - 9
            - 1
        *   - 1
            - 1
            - 3
            - 1
        *   - 2
            - 7
            - 5
            - 1
        *   - 3
            - 2
            - 0
            - 1
        *   - 0
            - 3
            - 8
            - 2
        *   - 2
            - 4
            - 4
            - 2
        *   - 4
            - 1
            - 7
            - 2
        *   - 6
            - 0
            - 2
            - 2
    """

    def __init__(self, features: list[str]) -> None:
        self.features = features

    @override
    def transform(self, data: pl.DataFrame) -> pl.DataFrame:
        logger.debug("Exploding timeseries")
        return data.explode(self.features)
