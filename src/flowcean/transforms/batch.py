from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any

import polars as pl

from flowcean.core.transform import Transform


class RowTransform(ABC):
    @abstractmethod
    def apply(self, row: tuple[Any, ...]) -> pl.DataFrame:
        pass


class BatchTransform(Transform):
    def __init__(
        self, feature: str, child_transforms: Iterable[RowTransform]
    ) -> None:
        super().__init__()
        self.feature = feature
        self.child_transforms = child_transforms

    def apply(self, data: pl.DataFrame) -> pl.DataFrame:
        selected_data = data.select(self.feature)
        mapped_data = selected_data.map_rows(
            lambda pc: self.map_element(pc[0])
        )
        return pl.DataFrame({self.feature: mapped_data})

    def map_element(self, pc: tuple[Any, ...]) -> pl.DataFrame:
        return pl.concat(
            (transform.apply(pc) for transform in self.child_transforms),
            how="horizontal",
        )
