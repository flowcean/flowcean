import logging
from typing import override

import polars as pl

from flowcean.core import Metric

logger = logging.getLogger(__name__)


class MeanEuclideanDistance(Metric[pl.LazyFrame, pl.LazyFrame, float]):
    def __init__(self, features: list[str] | None = None) -> None:
        super().__init__()
        self.features = features
        self.multi_output = True

    @override
    def prepare(self, data: pl.LazyFrame) -> pl.LazyFrame:
        return data

    @override
    def _compute(self, true: pl.LazyFrame, predicted: pl.LazyFrame) -> float:
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
        if not isinstance(mean_dist, float):
            message = "Euclidean distance requires non-empty numeric data"
            raise ValueError(message)
        return mean_dist
