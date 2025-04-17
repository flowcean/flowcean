import logging

import polars as pl

from flowcean.core.transform import Transform

logger = logging.getLogger(__name__)


class ZeroOrderHoldMatching(Transform):
    def __init__(
        self,
        topics: list[str],
    ) -> None:
        """Initialize the ZeroOrderHoldMatching transform.

        Args:
            topics (list[str]): List of topics to join.
        """
        super().__init__()
        self.topics = topics

    def _explode_timeseries(
        self,
        df: pl.LazyFrame,
        column: str,
    ) -> pl.LazyFrame:
        exploded = df.select(column).with_row_index().explode(column)
        unnested = exploded.unnest(column)
        renamed = unnested.with_columns(
            pl.col("value").name.map_fields(lambda x: f"{column}/{x}"),
        )
        return renamed.unnest("value")

    def _join_topics(
        self,
        df: pl.LazyFrame,
        topics: list[str],
    ) -> pl.LazyFrame:
        it = iter(topics)
        joined = self._explode_timeseries(df, next(it))

        for topic in it:
            joined = joined.join(
                self._explode_timeseries(df, topic),
                on=["index", "time"],
                how="full",
                coalesce=True,
            )

        return joined.sort(["index", "time"])

    def apply(self, data: pl.LazyFrame) -> pl.LazyFrame:
        logger.debug("Applying ZeroOrderHoldMatching transform")
        joined = self._join_topics(data, self.topics)
        return joined.select(
            pl.all().forward_fill().over("index"),
        ).drop_nulls()
