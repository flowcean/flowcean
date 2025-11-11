import math
from collections.abc import Iterator
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import PyTree
from sklearn.tree import (
    DecisionTreeClassifier,
    _tree,  # pyright: ignore[reportAttributeAccessIssue]
)

from flowcean.ode import Guard, HybridSystem, Mode
from flowcean.ode.hybrid_system import CondFn, FlowFn, RealScalarLike


@dataclass
class Bound:
    left: float | None = None
    right: float | None = None

    def update_left(self, value: float) -> None:
        if self.left is None or value > self.left:
            self.left = value

    def update_right(self, value: float) -> None:
        if self.right is None or value < self.right:
            self.right = value


FeatureName: TypeAlias = str


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

            left = deepcopy(conditions)
            left.setdefault(name, Bound()).update_right(threshold)
            recurse(tree_.children_left[node_id], left)

            right = deepcopy(conditions)
            right.setdefault(name, Bound()).update_left(threshold)
            recurse(tree_.children_right[node_id], right)
        else:
            output = np.argmax(tree_.value[node_id])
            paths.append(TreePath(conditions, int(output)))

    recurse(0, {})
    return paths


ModeName: TypeAlias = str


def path_to_transitions(
    paths: list[TreePath],
    mode_feature_name: str,
    mode_decoding: dict[int, ModeName],
) -> dict[tuple[ModeName, ModeName], list[dict[FeatureName, Bound]]]:
    transitions: dict[
        tuple[ModeName, ModeName],
        list[dict[FeatureName, Bound]],
    ] = {}

    max_mode_index = max(mode_decoding.keys())

    for path in paths:
        target = mode_decoding[path.output]
        mode_bound = path.conditions.pop(mode_feature_name, Bound())
        start = (
            math.ceil(mode_bound.left) if mode_bound.left is not None else 0
        )
        stop = (
            math.ceil(mode_bound.right)
            if mode_bound.right is not None
            else max_mode_index + 1
        )
        source_modes = [mode_decoding[i] for i in range(start, stop)]
        for source in source_modes:
            key = (source, target)
            transitions.setdefault(key, []).append(path.conditions)

    return transitions


def make_condition(
    conditions: list[dict[FeatureName, Bound]],
    time_feature_name: FeatureName,
    features: list[FeatureName],
) -> CondFn:
    """Create a condition function from a list of feature bounds.

    Args:
        conditions: A list of conditions, where each condition is a dictionary
            mapping feature names to their corresponding bounds.
        time_feature_name: The name of the time feature.
        features: A list of feature names. The order of features corresponds
            to the order of state variables in the hybrid system.

    Returns:
        A condition function that evaluates to True if any of the conditions
        are satisfied.
    """
    feature_to_index = {name: i for i, name in enumerate(features)}

    def condition_fn(
        t: RealScalarLike,
        x: PyTree,
        _args: Any,
        **_kwargs: Any,
    ) -> RealScalarLike | bool:
        # jax.debug.print("Evaluating condition at t={}, x={}", t, x)
        def generate_edge_conditions(
            condition: dict[FeatureName, Bound],
        ) -> Iterator[jax.Array]:
            for feature, bound in condition.items():
                if feature == time_feature_name:
                    value = t
                else:
                    index = feature_to_index[feature]
                    value = x[index]
                if bound.left is not None:
                    yield jnp.greater_equal(value, bound.left)
                if bound.right is not None:
                    yield jnp.less(value, bound.right)

        trigger = [
            jnp.all(
                jnp.array(list(generate_edge_conditions(condition))),
            )
            for condition in conditions
        ]
        return jnp.any(jnp.array(trigger))

    return condition_fn


class HybridDecisionTree(HybridSystem):
    def __init__(
        self,
        flows: dict[ModeName, FlowFn],
        tree: DecisionTreeClassifier,
        input_names: list[str],
        mode_feature_name: FeatureName,
        mode_decoding: dict[int, ModeName],
        time_feature_name: FeatureName,
        features: list[FeatureName],
    ) -> None:
        paths = tree_to_paths(tree, input_names)
        transitions = path_to_transitions(
            paths,
            mode_feature_name,
            mode_decoding,
        )
        modes = {}
        for mode_name, flow in flows.items():
            guards = []
            for (source, target), conditions in transitions.items():
                if source != mode_name:
                    continue
                if target == source:
                    continue

                guard = Guard(
                    condition=make_condition(
                        conditions,
                        time_feature_name=time_feature_name,
                        features=features,
                    ),
                    target_mode=target,
                )

                guards.append(guard)

            modes[mode_name] = Mode(flow=flow, guards=guards)
        self.transitions = transitions
        super().__init__(modes=modes)
