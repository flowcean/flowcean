import logging
from typing import Any

import polars as pl
from typing_extensions import override

from flowcean.core import Metric

logger = logging.getLogger(__name__)


class MeanEuclideanDistance(Metric):
    def __init__(self, features: list[str] | None = None) -> None:
        super().__init__()
        self.features = features
        self.multi_output = True

    @override
    def _compute(self, true: pl.LazyFrame, predicted: pl.LazyFrame) -> Any:
        columns = self.features if self.features is not None else pl.all()

        true_df = true.select(columns).collect(engine="streaming")
        pred_df = predicted.select(columns).collect(engine="streaming")

        squared_diff = (true_df - pred_df).select(pl.all() ** 2)
        euclidean_dists = squared_diff.sum_horizontal().sqrt()
        mean_dist = euclidean_dists.mean()
        logger.info(
            "Computed EuclideanDistance over columns %s: %f",
            columns,
            mean_dist,
        )
        return mean_dist
