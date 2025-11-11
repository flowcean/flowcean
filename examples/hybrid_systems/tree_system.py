from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
from sklearn.tree import (
    DecisionTreeClassifier,
    _tree,  # pyright: ignore[reportAttributeAccessIssue]
)

from flowcean.ode import Guard, HybridSystem, Mode

if TYPE_CHECKING:
    from jaxtyping import PyTree

    from flowcean.ode.hybrid_system import CondFn, FlowFn, RealScalarLike


@dataclass(frozen=True, slots=True)
class Bound:
    left: float | None = None
    right: float | None = None

    def with_updated_left(self, value: float) -> Bound:
        if self.left is None or value > self.left:
            return Bound(left=value, right=self.right)
        return self

    def with_updated_right(self, value: float) -> Bound:
        if self.right is None or value < self.right:
            return Bound(left=self.left, right=value)
        return self


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


ModeName: TypeAlias = str


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
        def evaluate_condition(cond: dict[FeatureName, Bound]) -> jax.Array:
            vals = []
            for feature, bound in cond.items():
                value = (
                    t
                    if feature == time_feature_name
                    else x[feature_to_index[feature]]
                )
                if bound.left is not None:
                    vals.append(jnp.greater_equal(value, bound.left))
                if bound.right is not None:
                    vals.append(jnp.less(value, bound.right))
            return jnp.all(jnp.stack(vals))

        return jnp.any(jnp.stack([evaluate_condition(c) for c in conditions]))

    return condition_fn


class HybridDecisionTree(HybridSystem):
    """A hybrid system where transitions are learned from a decision tree."""

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
        modes = {
            name: Mode(
                flow=flow,
                guards=[
                    Guard(
                        condition=make_condition(
                            conditions,
                            time_feature_name=time_feature_name,
                            features=features,
                        ),
                        target_mode=target,
                    )
                    for (source, target), conditions in transitions.items()
                    if source == name and target != source
                ],
            )
            for name, flow in flows.items()
        }
        self.transitions = transitions
        super().__init__(modes=modes)
