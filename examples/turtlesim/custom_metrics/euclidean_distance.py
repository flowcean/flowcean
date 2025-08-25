import logging
from typing import Any, override

import polars as pl

from flowcean.core.metric import OfflineMetric

logger = logging.getLogger(__name__)


class EuclideanDistance(OfflineMetric):
    def __init__(self, columns: list[str] | None = None) -> None:
        super().__init__()
        self.columns = columns
        self.multi_output = True

    @override
    def __call__(self, true: pl.LazyFrame, predicted: pl.LazyFrame) -> Any:
        true_df = true.collect(engine="streaming")
        pred_df = predicted.collect(engine="streaming")

        if true_df.is_empty() or pred_df.is_empty():
            logger.warning("Empty DataFrame detected; returning NaN")
            return float("nan")

        # Select specified columns or all columns if none specified
        columns = self.columns if self.columns is not None else true_df.columns
        if not all(col in true_df.columns for col in columns):
            logger.error(
                "Columns %s not found in true DataFrame columns %s",
                columns,
                true_df.columns,
            )
            raise ValueError(columns)
        if not all(col in pred_df.columns for col in columns):
            logger.error(
                "Columns %s not found in predicted DataFrame columns %s",
                columns,
                pred_df.columns,
            )
            raise ValueError(columns)

        true_df = true_df.select(columns)
        pred_df = pred_df.select(columns)

        if true_df.shape != pred_df.shape:
            logger.error(
                "Shape mismatch: true %s, predicted %s",
                true_df.shape,
                pred_df.shape,
            )
            raise ValueError

        squared_diff = (true_df - pred_df).select(pl.all() ** 2)
        euclidean_dists = squared_diff.sum_horizontal().sqrt()
        mean_dist = float(euclidean_dists.mean())
        logger.info(
            "Computed EuclideanDistance over columns %s: %f",
            columns,
            mean_dist,
        )
        return mean_dist
