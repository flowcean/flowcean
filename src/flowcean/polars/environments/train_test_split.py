from __future__ import annotations

import logging
from itertools import accumulate
from typing import TYPE_CHECKING

from flowcean.utils.random import get_seed

from .dataframe import DataFrame

if TYPE_CHECKING:
    from collections.abc import Sequence

    import polars as pl

    from flowcean.core.environment.offline import OfflineEnvironment

logger = logging.getLogger(__name__)


class TrainTestSplit:
    """Split data into train and test sets."""

    def __init__(
        self,
        ratio: float,
        *,
        shuffle: bool = False,
    ) -> None:
        """Initialize the train-test splitter.

        Args:
            ratio: The ratio of the data to put in the training set.
            shuffle: Whether to shuffle the data before splitting.
        """
        if ratio < 0 or ratio > 1:
            message = "ratio must be between 0 and 1"
            raise ValueError(message)
        self.ratio = ratio
        self.shuffle = shuffle

    def split(
        self,
        environment: OfflineEnvironment,
    ) -> tuple[DataFrame, DataFrame]:
        """Split the data into train and test sets.

        Args:
            environment: The environment to split.
        """
        logger.info("Splitting data into train and test sets")
        data = environment.observe().collect(streaming=True)
        pivot = int(len(data) * self.ratio)
        splits = _split(
            data,
            lengths=[pivot, len(data) - pivot],
            shuffle=self.shuffle,
            seed=get_seed(),
        )
        return DataFrame(splits[0].lazy()), DataFrame(splits[1].lazy())


def _split(
    dataset: pl.DataFrame,
    lengths: Sequence[int],
    *,
    shuffle: bool = False,
    seed: int | None = None,
) -> list[pl.DataFrame]:
    shuffled_dataset = dataset.sample(fraction=1.0, shuffle=shuffle, seed=seed)

    return [
        shuffled_dataset.slice(offset - length, length)
        for offset, length in zip(accumulate(lengths), lengths, strict=True)
    ]
