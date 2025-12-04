from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from dynamic_hybrid_system import Bound, FeatureName, ModeName, build_modes
from sklearn.tree import (
    DecisionTreeClassifier,
    _tree,  # pyright: ignore[reportAttributeAccessIssue]
)

from flowcean.ode import HybridSystem

if TYPE_CHECKING:
    from collections.abc import Sequence

    from flowcean.ode.hybrid_system import FlowFn


@dataclass
class TreePath:
    conditions: dict[FeatureName, Bound]
    output: int


def tree_to_paths(
    tree: DecisionTreeClassifier,
    feature_names: list[str],
) -> list[TreePath]:
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    paths = []

    def recurse(node_id: int, conditions: dict[FeatureName, Bound]) -> None:
        if tree_.feature[node_id] != _tree.TREE_UNDEFINED:
            name = feature_name[node_id]
            threshold = tree_.threshold[node_id]

            bound = conditions.get(name, Bound())
            left = {
                **conditions,
                name: bound.with_updated_right(threshold),
            }
            recurse(tree_.children_left[node_id], left)

            right = {
                **conditions,
                name: bound.with_updated_left(threshold),
            }
            recurse(tree_.children_right[node_id], right)
        else:
            output = np.argmax(tree_.value[node_id])
            paths.append(TreePath(conditions, int(output)))

    recurse(0, {})
    return paths


def path_to_transitions(
    paths: list[TreePath],
    mode_feature_name: FeatureName,
    mode_decoding: dict[int, ModeName],
) -> dict[tuple[ModeName, ModeName], list[dict[FeatureName, Bound]]]:
    transitions: dict[
        tuple[ModeName, ModeName],
        list[dict[FeatureName, Bound]],
    ] = defaultdict(list)

    max_mode_index = max(mode_decoding.keys())

    for path in paths:
        target = mode_decoding[path.output]
        mode_bound = path.conditions.pop(mode_feature_name, Bound())
        start = math.ceil(mode_bound.left or 0)
        stop = math.ceil(mode_bound.right or (max_mode_index + 1))
        for i in range(start, stop):
            source = mode_decoding[i]
            transitions[(source, target)].append(path.conditions)

    return transitions


class HybridDecisionTree(HybridSystem):
    """A hybrid system where transitions are learned from a decision tree."""

    def __init__(
        self,
        flows: dict[ModeName, FlowFn],
        tree: DecisionTreeClassifier,
        input_names: list[str],
        mode_feature: FeatureName,
        mode_decoding: dict[int, ModeName],
        time_feature: FeatureName,
        features: Sequence[FeatureName],
    ) -> None:
        paths = tree_to_paths(tree, input_names)
        transitions = path_to_transitions(
            paths,
            mode_feature,
            mode_decoding,
        )
        self.mode_decoding = mode_decoding
        self.transitions = transitions
        super().__init__(
            modes=build_modes(
                flows,
                transitions,
                time_feature,
                features,
            ),
        )

    def print_transitions(self) -> None:
        for (source, target), conditions in self.transitions.items():
            if source == target:
                continue
            print(f"{source} -> {target}:")
            for cond in conditions:
                cond_str = " and ".join(
                    b.to_expression_str(f) for f, b in sorted(cond.items())
                )
                print(f"  {cond_str}")
