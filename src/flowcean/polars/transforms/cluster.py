import logging
from collections.abc import Iterable
from typing import Any, Protocol

import numpy.typing as npt
import polars as pl
from typing_extensions import override

from flowcean.core import Transform

logger = logging.getLogger(__name__)


class Clusterer(Protocol):
    def fit(self, X: npt.ArrayLike) -> Any: ...  # noqa: N803, scikit-learn style
    def predict(self, X: npt.ArrayLike) -> npt.ArrayLike: ...  # noqa: N803, scikit-learn style


class Cluster(Transform):
    """Cluster data using a clustering algorithm.

    This transform allows to cluster data using a specified clustering
    algorithm. The resulting cluster label is added as a new feature to the
    DataFrame.
    """

    def __init__(
        self,
        clusterer: Clusterer,
        *,
        cluster_feature_name: str = "cluster_label",
        features: Iterable[str] | None = None,
    ) -> None:
        """Initializes the Cluster transform.

        Args:
            clusterer: The clustering algorithm to use.
            cluster_feature_name: The name of the feature to store the cluster
                labels.
            features: The features to use for clustering. If None, all features
                are used.
        """
        self.clusterer = clusterer
        self.cluster_feature_name = cluster_feature_name
        self.features = features

    @override
    def apply(self, data: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame:
        data = data.select(
            self.features or pl.all(),
        )
        if isinstance(data, pl.LazyFrame):
            data = data.collect()

        cluster_features = data.to_numpy(writable=True)
        labels = self.clusterer.predict(cluster_features)
        data = data.with_columns(
            pl.Series(self.cluster_feature_name, labels),
        )
        return data.lazy()

    @override
    def fit(self, data: pl.LazyFrame | pl.DataFrame) -> "Cluster":
        data = data.select(self.features or pl.all())
        if isinstance(data, pl.LazyFrame):
            data = data.collect()

        cluster_features = data.to_numpy(writable=True)
        self.clusterer.fit(cluster_features)

        return self
