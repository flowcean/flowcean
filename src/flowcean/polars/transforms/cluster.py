from collections.abc import Iterable
from typing import Protocol

import numpy.typing as npt
import polars as pl

from flowcean.core.transform import Transform


class ClusterFunction(Protocol):
    def fit(self, x: npt.ArrayLike) -> None: ...
    def predict(self, x: npt.ArrayLike) -> Iterable[int]: ...


class Cluster(Transform):
    """A transform that applies clustering to the data.

    Args:
        cluster_function: A clustering function that implements fit and predict
            methods.
        input_features: The list of input features to use for clustering.
            Defaults to None, which uses all features.
        output_feature: The name of the output feature to store cluster labels.
    """

    def __init__(
        self,
        cluster_function: ClusterFunction,
        *,
        input_features: list[str] | None = None,
        output_feature: str = "cluster_label",
    ) -> None:
        super().__init__()
        self.cluster_function = cluster_function
        self.input_features = input_features
        self.output_feature = output_feature

    def apply(self, data: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame:
        if isinstance(data, pl.DataFrame):
            data = data.lazy()

        def map_df(df: pl.DataFrame) -> pl.DataFrame:
            features = df.to_numpy(writable=True)
            labels = self.cluster_function.predict(features)
            return pl.DataFrame({self.output_feature: labels}).cast(pl.Int64)

        return pl.concat(
            [
                data,
                data.select(self.input_features or pl.all()).map_batches(
                    map_df,
                    streamable=True,
                    schema={self.output_feature: pl.Int64},
                ),
            ],
            how="horizontal",
        )

    def fit(self, data: pl.LazyFrame | pl.DataFrame) -> "Cluster":
        if self.input_features is None:
            features = data.lazy().collect().to_numpy(writable=True)
        else:
            features = (
                data.select(self.input_features)
                .lazy()
                .collect()
                .to_numpy(writable=True)
            )

        self.cluster_function.fit(features)

        return self
