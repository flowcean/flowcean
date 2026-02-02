from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

import polars as pl
from typing_extensions import override

from flowcean.core.model import Model

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray


class SupportsPredict(Protocol):
    """Protocol describing an object that has a `predict` method."""

    def predict(self, X: Any) -> NDArray: ...  # noqa: N803 X is a standard name in sklearn


class SciKitModel(Model):
    """A model that wraps a scikit-learn model."""

    estimator: SupportsPredict

    def __init__(
        self,
        estimator: SupportsPredict,
        *,
        input_features: Sequence[str],
        output_features: Sequence[str],
        name: str | None = None,
    ) -> None:
        """Initialize the model.

        Args:
            estimator: The scikit-learn estimator.
            input_features: The names of the input features.
            output_features: The names of the output features.
            name: The name of the model.
        """
        if name is None:
            name = estimator.__class__.__name__
        self._name = name
        self.estimator = estimator
        self.input_features = list(input_features)
        self.output_features = list(output_features)

    @override
    def _predict(
        self,
        input_features: pl.DataFrame | pl.LazyFrame,
    ) -> pl.LazyFrame:
        if isinstance(input_features, pl.LazyFrame):
            input_features = input_features.collect()

        outputs = self.estimator.predict(input_features)
        if len(self.output_features) == 1:
            data = {self.output_features[0]: outputs}
        else:
            data = {
                self.output_features[i]: outputs[:, i]
                for i in range(len(self.output_features))
            }
        return pl.LazyFrame(data)
