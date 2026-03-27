import polars as pl
from typing_extensions import override

from flowcean.core.model import Model


class HyDRAModel(Model):
    def __init__(
        self,
        modes: list[Model],
        *,
        input_features: list[str],
        output_features: list[str],
    ) -> None:
        super().__init__()
        self.modes = modes
        self.input_features = input_features
        self.output_features = output_features

    @override
    def _predict(
        self,
        input_features: pl.LazyFrame,
    ) -> pl.LazyFrame:
        return self.modes[0].predict(input_features)
