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
    """A BatchTransform applies a list of RowTransforms to a single feature.

    Args:
        feature (str): The feature to apply the RowTransforms to.
        child_transforms (Iterable[RowTransform]): The RowTransforms to apply.

    This transform can be used to apply a list of RowTransforms to the column
    `feature`.
    Assuming there are two RowTransforms `MeanTransform` and `MaxTransform`
    that calculate the mean and max of the input list, respectively.
    For this example, the loaded data is represented by the table:

    feature    | time
    -----------|-----
    [1, 2, 3]  | 0
    [4, 5, 6]  | 1
    [7, 8, 9]  | 2

    The resulting Dataframe after the transform is:

    feature    | mean | max | time
    -----------|------|------------
    [1, 2, 3]  | 2    | 3   | 0
    [4, 5, 6]  | 5    | 6   | 1
    [7, 8, 9]  | 8    | 9   | 2

    """

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
