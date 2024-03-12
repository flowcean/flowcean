from typing import Any

import polars as pl
from typing_extensions import override

from agenc.core import Metric


class DummyMetric(Metric):
    @override
    def __call__(self, true: pl.DataFrame, predicted: pl.DataFrame) -> Any:
        return 0
