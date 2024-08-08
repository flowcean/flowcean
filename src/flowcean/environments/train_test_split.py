from __future__ import annotations

import logging
from itertools import accumulate
from typing import TYPE_CHECKING

import polars as pl

from .dataset import Dataset

if TYPE_CHECKING:
    from collections.abc import Sequence

    from flowcean.core import OfflineEnvironment

logger = logging.getLogger(__name__)


class TrainTestSplit:
    """Split environments into a train and a test dataset."""

    def __init__(
        self,
        ratio: float,
        *,
        shuffle: bool = False,
    ) -> None:
        """Initialize the TrainTestSplit.

        Args:
            ratio: Split ratio between train and test set. If N is the total
                number of samples in the environment, N*ratio samples will be
                assigned to the training set, while N*(1-ratio) samples will be
                assigned to the test set.
            shuffle: If true, the samples from the environment are shuffled
                before being split.
        """
        if ratio < 0 or ratio > 1:
            message = "ratio must be between 0 and 1"
            raise ValueError(message)
        self.ratio = ratio
        self.shuffle = shuffle

    def split(
        self,
        environment: OfflineEnvironment,
    ) -> tuple[Dataset, Dataset]:
        """Split the provided environment into a train and test environment.

        Split the provided environment into a train and a test environment
        according to the ratio specified when this TrainTestSplit object was
        created. To perform the split, the environment will be materialized so
        that any outstanding transformations are applied.

        Args:
            environment: Environment to split.

        Returns:
            A tuple of two datasets that are split according to the ratio where
            the first dataset is used for training and the later dataset is
            used for testing.
        """
        logger.info("Splitting data into train and test sets")
        data = environment.get_data()
        data = data if isinstance(data, pl.DataFrame) else data.collect()
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
