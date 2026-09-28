from dataclasses import dataclass, field
from pathlib import Path
from typing import override

import polars as pl
from sklearn.tree import DecisionTreeClassifier, export_text

from flowcean.core import Model

from .config import SelectorFeatureConfig
from .graph import build_selector_dot, render_dot_svg
from .inspection import (
    SelectorFlowInspection,
    SelectorInspection,
    SelectorLeafInspection,
    SelectorNodeInspection,
    summarize_flow_model,
)
from .text import (
    render_flow_summary_text,
    render_leaf_summary_text,
    render_prediction_debug_text,
    render_summary_text,
)


def _reconstruct_class_support(
    class_probabilities: list[float],
    weighted_sample_count: float,
    classes: tuple[int, ...],
) -> dict[int, float]:
    return {
        flow_id: round(probability * weighted_sample_count, 12)
        for flow_id, probability in zip(
            classes,
            class_probabilities,
            strict=True,
        )
    }


@dataclass(frozen=True)
class FlowPredictionResult:
    ready: bool
    flow_id: int | None
    probabilities: dict[int, float] = field(default_factory=dict)
    leaf_id: int | None = None
    flow_model: Model | None = None


class HybridDecisionTreeModel(Model):
    def __init__(
        self,
        classifier: DecisionTreeClassifier,
        feature_columns: tuple[str, ...],
        feature_config: SelectorFeatureConfig,
        flow_models_by_id: dict[int, Model] | None = None,
    ) -> None:
        self.classifier = classifier
        self.feature_columns = feature_columns
        self.feature_config = feature_config
        self.flow_models_by_id = flow_models_by_id or {}

    @override
    def _predict(
        self,
        input_features: pl.DataFrame | pl.LazyFrame,
    ) -> pl.LazyFrame:
        features = self._collect_features(input_features)
        if features.height == 0:
            return pl.DataFrame(schema={"flow_id": pl.Int64}).lazy()

        predictions = self.classifier.predict(features)
        return pl.DataFrame(
            {"flow_id": [int(flow_id) for flow_id in predictions]},
        ).lazy()

    def predict_details(
        self,
        input_features: pl.DataFrame | pl.LazyFrame,
    ) -> list[FlowPredictionResult]:
        features = self._collect_features(input_features)
        if features.height == 0:
            return []

        predicted_flows = self.classifier.predict(features)
        probabilities = self.classifier.predict_proba(features)
        leaf_ids = self.classifier.apply(features)
        classes = [int(flow_id) for flow_id in self.classifier.classes_]

        results: list[FlowPredictionResult] = []
        for flow_id, row_probabilities, leaf_id in zip(
            predicted_flows,
            probabilities,
            leaf_ids,
            strict=True,
        ):
            resolved_flow = int(flow_id)
            results.append(
                FlowPredictionResult(
                    ready=True,
                    flow_id=resolved_flow,
                    probabilities={
                        class_id: float(probability)
                        for class_id, probability in zip(
                            classes,
                            row_probabilities,
                            strict=True,
                        )
                    },
                    leaf_id=int(leaf_id),
                    flow_model=self.resolve_flow(resolved_flow),
                ),
            )

        return results

    def resolve_flow(self, flow_id: int) -> Model | None:
        return self.flow_models_by_id.get(flow_id)

    def feature_importances(self) -> dict[str, float]:
        return {
            column: float(importance)
            for column, importance in zip(
                self.feature_columns,
                self.classifier.feature_importances_,
                strict=True,
            )
        }

    def tree_text(self) -> str:
        return export_text(
            self.classifier,
            feature_names=list(self.feature_columns),
        )

    def summary_text(self) -> str:
        return render_summary_text(self.inspect())

    def leaf_summary_text(self) -> str:
        return render_leaf_summary_text(self.inspect())

    def flow_summary_text(self) -> str:
        return render_flow_summary_text(self.inspect())

    def to_dot(self) -> str:
        return build_selector_dot(self.inspect())

    def to_svg(self) -> str:
        return render_dot_svg(self.to_dot())

    def save_svg(self, path: str | Path) -> None:
        Path(path).write_text(self.to_svg(), encoding="utf-8")

    def debug_prediction_text(
        self,
        input_features: pl.DataFrame | pl.LazyFrame,
    ) -> str:
        features = self._collect_features(input_features)
        predictions = self.predict_details(features)
        return render_prediction_debug_text(
            input_rows=features,
            predictions=predictions,
            feature_columns=self.feature_columns,
        )

    def inspect(self) -> SelectorInspection:
        tree = self.classifier.tree_
        classes = tuple(int(flow_id) for flow_id in self.classifier.classes_)
        nodes: list[SelectorNodeInspection] = []
        leaves: list[SelectorLeafInspection] = []
        flow_sample_counts = dict.fromkeys(classes, 0.0)

        for node_id in range(tree.node_count):
            left_child_id = int(tree.children_left[node_id])
            right_child_id = int(tree.children_right[node_id])
            is_leaf = left_child_id == right_child_id
            sample_count = int(tree.n_node_samples[node_id])
            weighted_class_support = _reconstruct_class_support(
                class_probabilities=tree.value[node_id][0].tolist(),
                weighted_sample_count=float(
                    tree.weighted_n_node_samples[node_id],
                ),
                classes=classes,
            )
            predicted_flow_id = max(
                weighted_class_support,
                key=weighted_class_support.__getitem__,
            )

            node = SelectorNodeInspection(
                node_id=node_id,
                sample_count=sample_count,
                impurity=float(tree.impurity[node_id]),
                is_leaf=is_leaf,
                predicted_flow_id=predicted_flow_id,
                weighted_class_support=weighted_class_support,
                feature_index=None if is_leaf else int(tree.feature[node_id]),
                feature_name=None
                if is_leaf
                else self.feature_columns[int(tree.feature[node_id])],
                threshold=None if is_leaf else float(tree.threshold[node_id]),
                left_child_id=None if is_leaf else left_child_id,
                right_child_id=None if is_leaf else right_child_id,
            )
            nodes.append(node)

            if is_leaf:
                for flow_id, support in weighted_class_support.items():
                    flow_sample_counts[flow_id] += support
                leaves.append(
                    SelectorLeafInspection(
                        node_id=node_id,
                        flow_id=predicted_flow_id,
                        sample_count=node.sample_count,
                        weighted_class_support=weighted_class_support,
                        flow_summary=summarize_flow_model(
                            self.resolve_flow(predicted_flow_id),
                        ),
                    ),
                )

        flows = tuple(
            SelectorFlowInspection(
                flow_id=flow_id,
                weighted_support=flow_sample_counts[flow_id],
                flow_summary=summarize_flow_model(
                    self.resolve_flow(flow_id),
                ),
            )
            for flow_id in classes
        )

        return SelectorInspection(
            feature_columns=self.feature_columns,
            classes=classes,
            max_depth=int(tree.max_depth),
            n_leaves=int(tree.n_leaves),
            nodes=tuple(nodes),
            leaves=tuple(leaves),
            flows=flows,
        )

    def _collect_features(
        self,
        input_features: pl.DataFrame | pl.LazyFrame,
    ) -> pl.DataFrame:
        frame = (
            input_features.collect()
            if isinstance(input_features, pl.LazyFrame)
            else input_features
        )
        return frame.select(self.feature_columns)
