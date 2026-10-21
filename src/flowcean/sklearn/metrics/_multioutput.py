from __future__ import annotations

from abc import ABC
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import polars as pl
from numpy.typing import NDArray

from flowcean.polars import PolarsMetric


class MultiOutputMetric(PolarsMetric[float | dict[str, float]], ABC):
    """Prepare tabular data and finalize sklearn multioutput metrics."""

    def __init__(
        self,
        *,
        features: Sequence[str] | None = None,
        multioutput: Literal["raw_values", "uniform_average"] = "raw_values",
    ) -> None:
        super().__init__(features=features)
        self.multioutput = multioutput

    def _finalize_result(
        self,
        error: float | np.floating[Any] | NDArray[np.floating[Any]],
        true: pl.DataFrame,
    ) -> float | dict[str, float]:
        if isinstance(error, np.ndarray):
            return dict(zip(true.columns, map(float, error), strict=True))
        return float(error)
