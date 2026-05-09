from typing import Any, cast

import polars as pl
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier

from flowcean.core import Model, SupervisedLearner
from flowcean.hydra.selector.config import SelectorFeatureConfig
from flowcean.hydra.selector.features import (
    build_selector_dataset,
    validate_global_mode_labels,
)
from flowcean.hydra.selector.model import HybridDecisionTreeModel
from flowcean.utils import get_seed


class HybridDecisionTreeLearner(SupervisedLearner):
    classifier: DecisionTreeClassifier

    def __init__(
        self,
        feature_config: SelectorFeatureConfig,
        **tree_kwargs: Any,
    ) -> None:
        self.feature_config = feature_config
        classifier_kwargs: dict[str, Any] = dict(tree_kwargs)
        classifier_kwargs.setdefault("random_state", get_seed())
        self.classifier = DecisionTreeClassifier(**classifier_kwargs)

    def learn(
        self,
        inputs: pl.DataFrame | pl.LazyFrame,
        outputs: pl.DataFrame | pl.LazyFrame,
    ) -> HybridDecisionTreeModel:
        collected_inputs = (
            inputs.collect() if isinstance(inputs, pl.LazyFrame) else inputs
        )
        collected_outputs = (
            outputs.collect() if isinstance(outputs, pl.LazyFrame) else outputs
        )
        if len(collected_outputs.columns) != 1:
            message = (
                "HybridDecisionTreeLearner requires a single label column"
            )
            raise ValueError(message)

        labels = collected_outputs.to_series()
        try:
            labels = labels.cast(pl.Int64, strict=True)
        except pl.exceptions.InvalidOperationError as error:
            message = (
                "HybridDecisionTreeLearner requires integer-like mode IDs"
            )
            raise ValueError(message) from error

        unique_labels = labels.unique().drop_nulls().to_list()
        if not unique_labels:
            message = "HybridDecisionTreeLearner requires at least one label"
            raise ValueError(message)

        classifier = cast("DecisionTreeClassifier", clone(self.classifier))
        classifier.fit(collected_inputs, labels)
        return HybridDecisionTreeModel(
            classifier=classifier,
            feature_columns=tuple(collected_inputs.columns),
            feature_config=self.feature_config,
        )

    def learn_from_traces(
        self,
        traces: list[pl.DataFrame],
        mode_to_flow: dict[int, Model] | None = None,
    ) -> HybridDecisionTreeModel:
        validate_global_mode_labels(traces)
        dataset = build_selector_dataset(traces, self.feature_config)
        model = self.learn(
            dataset.features,
            pl.DataFrame({"mode": dataset.labels}),
        )
        return HybridDecisionTreeModel(
            classifier=model.classifier,
            feature_columns=model.feature_columns,
            feature_config=self.feature_config,
            mode_to_flow=mode_to_flow,
        )
