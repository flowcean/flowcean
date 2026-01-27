from typing import Protocol

import numpy.typing as npt
import polars as pl

from flowcean.core.learner import SupervisedLearner
from flowcean.core.model import Model


class ClusterProtocol(Protocol):
    def fit_predict(self, x: npt.ArrayLike) -> list[int]: ...
    def predict(self, x: npt.ArrayLike) -> list[int]: ...


class ClusterModel(Model):
    def __init__(
        self,
        clustering_function: ClusterProtocol,
        cluster_model_mapping: dict[int, Model],
    ) -> None:
        super().__init__()
        self.clustering_function = clustering_function
        self.cluster_model_mapping = cluster_model_mapping

    def _predict(
        self,
        input_features: pl.LazyFrame | pl.DataFrame,
    ) -> pl.DataFrame:
        if isinstance(input_features, pl.LazyFrame):
            input_features = input_features.collect()

        labels = self.clustering_function.predict(
            input_features.to_numpy(writable=True),
        )

        inputs_with_labels = input_features.with_columns(
            pl.Series("cluster_label", labels),
        )

        result: list[pl.DataFrame] = []
        for row in inputs_with_labels.iter_slices(1):
            label = row["cluster_label"][0]
            model = self.cluster_model_mapping[label]
            prediction = model.predict(row.drop("cluster_label").lazy())
            if isinstance(prediction, pl.LazyFrame):
                prediction = prediction.collect()
            result.append(prediction)

        return pl.concat(result)


class ClusterLearner(SupervisedLearner):
    def __init__(
        self,
        clustering_function: ClusterProtocol,
        learner: SupervisedLearner,
    ) -> None:
        super().__init__()
        self.clustering_function = clustering_function
        self.learner = learner

    def learn(
        self,
        inputs: pl.LazyFrame,
        outputs: pl.LazyFrame,
    ) -> ClusterModel:
        dfs = pl.collect_all([inputs, outputs])
        features = dfs[0].to_numpy(writable=True)
        group_labels = self.clustering_function.fit_predict(features)

        inputs_with_labels = dfs[0].with_columns(
            pl.Series("cluster_label", group_labels),
        )
        outputs_with_labels = dfs[1].with_columns(
            pl.Series("cluster_label", group_labels),
        )

        models = {}

        for label in set(group_labels):
            input_subset = inputs_with_labels.filter(
                pl.col("cluster_label") == label,
            ).drop("cluster_label")

            output_subset = outputs_with_labels.filter(
                pl.col("cluster_label") == label,
            ).drop("cluster_label")

            model = self.learner.learn(
                input_subset.lazy(),
                output_subset.lazy(),
            )
            models[label] = model

        return ClusterModel(self.clustering_function, models)
