from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeAlias

import jax
import jax.numpy as jnp

from flowcean.ode import Guard, Mode

if TYPE_CHECKING:
    from collections.abc import Sequence

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

    def to_expression_str(
        self,
        feature: str = "x",
        conjunction: str = "and",
    ) -> str:
        parts = []
        if self.left is not None:
            parts.append(f"{feature} >= {self.left:.3f}")
        if self.right is not None:
            parts.append(f"{feature} < {self.right:.3f}")
        return f" {conjunction} ".join(parts)


FeatureName: TypeAlias = str


def make_condition(
    conditions: list[dict[FeatureName, Bound]],
    time_feature_name: FeatureName,
    features: Sequence[FeatureName],
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


ModeName: TypeAlias = str


def build_modes(
    flows: dict[ModeName, FlowFn],
    transitions: dict[
        tuple[ModeName, ModeName],
        list[dict[FeatureName, Bound]],
    ],
    time_feature: FeatureName,
    features: Sequence[FeatureName],
) -> dict[str, Mode]:
    return {
        name: Mode(
            flow=flow,
            guards=[
                Guard(
                    condition=make_condition(
                        conditions,
                        time_feature_name=time_feature,
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
