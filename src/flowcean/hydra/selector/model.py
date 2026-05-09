from dataclasses import dataclass, field
from pathlib import Path
from typing import override

import polars as pl
from sklearn.tree import DecisionTreeClassifier, export_text

from flowcean.core import Model
from flowcean.hydra.selector.config import SelectorFeatureConfig
from flowcean.hydra.selector.graph import build_selector_dot, render_dot_svg
from flowcean.hydra.selector.inspection import (
    SelectorInspection,
    SelectorLeafInspection,
    SelectorModeInspection,
    SelectorNodeInspection,
    summarize_flow_model,
)
from flowcean.hydra.selector.text import (
    render_leaf_summary_text,
    render_mode_summary_text,
    render_prediction_debug_text,
    render_summary_text,
)


def _reconstruct_class_support(
    class_probabilities: list[float],
    weighted_sample_count: float,
    classes: tuple[int, ...],
) -> dict[int, float]:
    return {
        mode_id: round(probability * weighted_sample_count, 12)
        for mode_id, probability in zip(
            classes,
            class_probabilities,
            strict=True,
        )
    }


@dataclass(frozen=True)
class ModePredictionResult:
    ready: bool
    mode_id: int | None
    probabilities: dict[int, float] = field(default_factory=dict)
    leaf_id: int | None = None
    flow_model: Model | None = None


class HybridDecisionTreeModel(Model):
    def __init__(
        self,
        classifier: DecisionTreeClassifier,
        feature_columns: tuple[str, ...],
        feature_config: SelectorFeatureConfig,
        mode_to_flow: dict[int, Model] | None = None,
    ) -> None:
        self.classifier = classifier
        self.feature_columns = feature_columns
        self.feature_config = feature_config
        self.mode_to_flow = mode_to_flow or {}

    @override
    def _predict(
        self,
        input_features: pl.DataFrame | pl.LazyFrame,
    ) -> pl.LazyFrame:
        features = self._collect_features(input_features)
        if features.height == 0:
            return pl.DataFrame(schema={"mode": pl.Int64}).lazy()

        predictions = self.classifier.predict(features)
        return pl.DataFrame(
            {"mode": [int(mode_id) for mode_id in predictions]},
        ).lazy()

    def predict_details(
        self,
        input_features: pl.DataFrame | pl.LazyFrame,
    ) -> list[ModePredictionResult]:
        features = self._collect_features(input_features)
        if features.height == 0:
            return []

        predicted_modes = self.classifier.predict(features)
        probabilities = self.classifier.predict_proba(features)
        leaf_ids = self.classifier.apply(features)
        classes = [int(mode_id) for mode_id in self.classifier.classes_]

        results: list[ModePredictionResult] = []
        for mode_id, row_probabilities, leaf_id in zip(
            predicted_modes,
            probabilities,
            leaf_ids,
            strict=True,
        ):
            resolved_mode = int(mode_id)
            results.append(
                ModePredictionResult(
                    ready=True,
                    mode_id=resolved_mode,
                    probabilities={
                        class_id: float(probability)
                        for class_id, probability in zip(
                            classes,
                            row_probabilities,
                            strict=True,
                        )
                    },
                    leaf_id=int(leaf_id),
                    flow_model=self.resolve_flow(resolved_mode),
                ),
            )

        return results

    def resolve_flow(self, mode_id: int) -> Model | None:
        return self.mode_to_flow.get(mode_id)

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

    def mode_summary_text(self) -> str:
        return render_mode_summary_text(self.inspect())

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
        classes = tuple(int(mode_id) for mode_id in self.classifier.classes_)
        nodes: list[SelectorNodeInspection] = []
        leaves: list[SelectorLeafInspection] = []
        mode_sample_counts = dict.fromkeys(classes, 0.0)

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
            predicted_mode_id = max(
                weighted_class_support,
                key=weighted_class_support.__getitem__,
            )

            node = SelectorNodeInspection(
                node_id=node_id,
                sample_count=sample_count,
                impurity=float(tree.impurity[node_id]),
                is_leaf=is_leaf,
                predicted_mode_id=predicted_mode_id,
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
                for mode_id, support in weighted_class_support.items():
                    mode_sample_counts[mode_id] += support
                leaves.append(
                    SelectorLeafInspection(
                        node_id=node_id,
                        mode_id=predicted_mode_id,
                        sample_count=node.sample_count,
                        weighted_class_support=weighted_class_support,
                        flow_summary=summarize_flow_model(
                            self.resolve_flow(predicted_mode_id),
                        ),
                    ),
                )

        modes = tuple(
            SelectorModeInspection(
                mode_id=mode_id,
                weighted_support=mode_sample_counts[mode_id],
                flow_summary=summarize_flow_model(
                    self.resolve_flow(mode_id),
                ),
            )
            for mode_id in classes
        )

        return SelectorInspection(
            feature_columns=self.feature_columns,
            classes=classes,
            max_depth=int(tree.max_depth),
            n_leaves=int(tree.n_leaves),
            nodes=tuple(nodes),
            leaves=tuple(leaves),
            modes=modes,
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
