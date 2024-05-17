from __future__ import annotations

import logging
from itertools import accumulate
from typing import TYPE_CHECKING

from .dataset import Dataset

if TYPE_CHECKING:
    from collections.abc import Sequence

    import polars as pl

    from flowcean.core import OfflineEnvironment

logger = logging.getLogger(__name__)


class TrainTestSplit:
    def __init__(
        self,
        ratio: float,
        *,
        shuffle: bool = False,
    ) -> None:
        if ratio < 0 or ratio > 1:
            message = "ratio must be between 0 and 1"
            raise ValueError(message)
        self.ratio = ratio
        self.shuffle = shuffle

    def split(
        self,
        environment: OfflineEnvironment,
    ) -> tuple[Dataset, Dataset]:
        logger.info("Splitting data into train and test sets")
        data = environment.get_data()
        pivot = int(len(data) * self.ratio)
        splits = _split(
            data,
            lengths=[pivot, len(data) - pivot],
            shuffle=self.shuffle,
        )
        return Dataset(splits[0]), Dataset(splits[1])


def _split(
    dataset: pl.DataFrame,
    lengths: Sequence[int],
    *,
    shuffle: bool = False,
) -> list[pl.DataFrame]:
    shuffled_dataset = dataset.sample(fraction=1.0, shuffle=shuffle)

    return [
        shuffled_dataset.slice(offset - length, length)
        for offset, length in zip(accumulate(lengths), lengths, strict=True)
    ]
