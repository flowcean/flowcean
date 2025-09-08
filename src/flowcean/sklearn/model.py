from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

import polars as pl
from typing_extensions import override

from flowcean.core.model import Model

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy.typing import NDArray


class SupportsPredict(Protocol):
    """Protocol describing an object that has a `predict` method."""

    def predict(self, X: Any) -> NDArray: ...  # noqa: N803 X is a standard name in sklearn


class SciKitModel(Model):
    """A model that wraps a scikit-learn model."""

    estimator: SupportsPredict
    output_names: list[str]

    def __init__(
        self,
        estimator: SupportsPredict,
        *,
        output_names: Iterable[str],
        name: str | None = None,
    ) -> None:
        """Initialize the model.

        Args:
            estimator: The scikit-learn estimator.
            output_names: The names of the output columns.
            name: The name of the model.
        """
        if name is None:
            name = estimator.__class__.__name__
        self._name = name
        self.estimator = estimator
        self.output_names = list(output_names)

    @override
    def _predict(
        self,
        input_features: pl.DataFrame | pl.LazyFrame,
    ) -> pl.LazyFrame:
        if isinstance(input_features, pl.LazyFrame):
            input_features = input_features.collect()

        outputs = self.estimator.predict(input_features)
        if len(self.output_names) == 1:
            data = {self.output_names[0]: outputs}
        else:
            data = {
                self.output_names[i]: outputs[:, i]
                for i in range(len(self.output_names))
            }
        return pl.LazyFrame(data)
