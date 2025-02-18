import logging

import polars as pl
from typing_extensions import override

from flowcean.core import Transform

logger = logging.getLogger(__name__)


class Explode(Transform):
    """Explodes a Dataframe to long format by exploding the given features.

    Args:
        features (list[str]): List of features to explode.

    The below example shows the usage of a `Explode` transform in an
    `experiment.yaml` file. Assuming the loaded data is represented by the
    table:

    time          | feature_a    | feature_b    | constant
    --------------|--------------|--------------|---------
     [0, 1]       | [2, 1]       | [9, 3]       | 1
     [0, 2]       | [3, 4]       | [8, 4]       | 2

    This transform can be used to explode the columns `time`,
    `feature_a`, and `feature_b`.

    The resulting Dataframe after the transform is:

    time          | feature_a    | feature_b    | constant
    --------------|--------------|--------------|---------
     0            | 2            | 9            | 1
     1            | 1            | 3            | 1
     0            | 3            | 8            | 2
     2            | 4            | 4            | 2
    """

    def __init__(self, features: list[str]) -> None:
        self.features = features

    @override
    def apply(self, data: pl.LazyFrame) -> pl.LazyFrame:
        logger.debug("Exploding timeseries")
        return data.explode(self.features)
