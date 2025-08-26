import logging
from typing import Any

import polars as pl
from typing_extensions import override

from flowcean.core.metric import OfflineMetric

logger = logging.getLogger(__name__)


class MeanEuclideanDistance(OfflineMetric):
    def __init__(self, columns: list[str] | None = None) -> None:
        super().__init__()
        self.columns = columns
        self.multi_output = True

    @override
    def __call__(self, true: pl.LazyFrame, predicted: pl.LazyFrame) -> Any:
        columns = self.columns if self.columns is not None else pl.all()

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
