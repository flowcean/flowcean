import polars as pl

from flowcean.core.learner import SupervisedLearner
from flowcean.core.model import Model


class ClusterModel(Model):
    def __init__(
        self,
        cluster_feature: str,
        cluster_model_mapping: dict[int, Model],
        *,
        exclude_cluster_feature: bool = False,
    ) -> None:
        super().__init__()
        self.cluster_feature = cluster_feature
        self.cluster_model_mapping = cluster_model_mapping
        self.exclude_cluster_feature = exclude_cluster_feature

    def _predict(
        self,
        input_features: pl.LazyFrame | pl.DataFrame,
    ) -> pl.DataFrame:
        if isinstance(input_features, pl.LazyFrame):
            input_features = input_features.collect()

        result: list[pl.DataFrame] = []
        for row in input_features.iter_slices(1):
            label = row[self.cluster_feature][0]
            model = self.cluster_model_mapping[label]
            prediction = model.predict(
                (
                    row.drop(self.cluster_feature)
                    if self.exclude_cluster_feature
                    else row
                ).lazy(),
            )
            if isinstance(prediction, pl.LazyFrame):
                prediction = prediction.collect()
            result.append(prediction)

        return pl.concat(result)


class ClusterLearner(SupervisedLearner):
    def __init__(
        self,
        cluster_feature: str,
        learner: SupervisedLearner,
        *,
        remove_cluster_feature: bool = True,
    ) -> None:
        super().__init__()
        self.cluster_feature = cluster_feature
        self.learner = learner
        self.exclude_cluster_feature = remove_cluster_feature

    def learn(
        self,
        inputs: pl.LazyFrame,
        outputs: pl.LazyFrame,
    ) -> ClusterModel:
        dfs = pl.collect_all([inputs, outputs])

        inputs_with_labels = dfs[0]
        outputs_with_labels = dfs[1].with_columns(
            pl.Series(
                self.cluster_feature,
                inputs_with_labels[self.cluster_feature],
            ),
        )

        labels = (
            inputs_with_labels.select(self.cluster_feature)
            .unique()
            .to_series()
            .to_list()
        )

        models = {}

        for label in labels:
            input_subset = inputs_with_labels.filter(
                pl.col(self.cluster_feature) == label,
            )

            output_subset = outputs_with_labels.filter(
                pl.col(self.cluster_feature) == label,
            )

            if self.exclude_cluster_feature:
                input_subset = input_subset.drop(self.cluster_feature)
                output_subset = output_subset.drop(self.cluster_feature)

            model = self.learner.learn(
                input_subset.lazy(),
                output_subset.lazy(),
            )
            models[label] = model

        return ClusterModel(
            self.cluster_feature,
            models,
            exclude_cluster_feature=self.exclude_cluster_feature,
        )
