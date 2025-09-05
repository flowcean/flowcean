from collections.abc import Iterable
from typing import cast

import polars as pl

from flowcean.core import Adapter, Finished
from flowcean.polars.environments.dataframe import DataFrame


class DataFrameAdapter(Adapter):
    """Adapter wrapper for a DataFrame.

    This class allows to use a DataFrame as an Adapter and an input source
    for a tool loop. The tool receives data from the source DataFrame row by
    row. The tool loops results are collected in a result DataFrame which is
    written to a CSV file at the end of the process.
    """

    df: pl.LazyFrame
    df_len: int
    result_df: pl.DataFrame
    count: int = 0
    result_path: str

    def __init__(
        self,
        source: DataFrame,
        input_features: Iterable[str],
        result_path: str,
    ) -> None:
        super().__init__()

        self.df = cast("pl.LazyFrame", source.observe()).select(input_features)
        self.df_len = self.df.select(pl.len()).collect().item()
        self.result_path = result_path

    def start(self) -> None:
        self.result_df = pl.DataFrame()

    def stop(self) -> None:
        self.result_df.write_csv(self.result_path)

    def get_data(self) -> pl.LazyFrame:
        self.count += 1
        if self.count > self.df_len:
            raise Finished
        return self.df.slice(self.count - 1, 1)

    def send_data(self, data: pl.LazyFrame) -> None:
        self.result_df = pl.concat(
            [self.result_df, data.collect()],
            how="vertical",
        )
