from typing import cast

import polars as pl
from polars.type_aliases import IntoExpr

from .predicate import Predicate


class PolarsPredicate(Predicate):
    """Predicate for Polars DataFrame."""

    def __init__(self, expr: IntoExpr) -> None:
        self.expr = expr

    def __call__(
        self,
        input_data: pl.DataFrame | pl.LazyFrame,
        prediction: pl.DataFrame | pl.LazyFrame,
    ) -> bool:
        input_data = (
            input_data.collect()
            if isinstance(input_data, pl.LazyFrame)
            else input_data
        )
        prediction = (
            prediction.collect()
            if isinstance(prediction, pl.LazyFrame)
            else prediction
        )
        return cast(
            "bool",
            pl.concat(
                [input_data, prediction],
                how="horizontal",
            )
            .select(
                self.expr.cast(pl.Boolean).alias("predicate"),
            )
            .item(),
        )
