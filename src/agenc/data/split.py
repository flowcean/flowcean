from collections.abc import Sequence
from itertools import accumulate

import polars as pl


def _split(
    dataset: pl.DataFrame, lengths: Sequence[int], shuffle: bool = False
) -> list[pl.DataFrame]:
    shuffled_dataset = dataset.sample(fraction=1.0, shuffle=shuffle)

    return [
        shuffled_dataset.slice(offset - length, length)
        for offset, length in zip(accumulate(lengths), lengths, strict=True)
    ]


class TrainTestSplit:
    """Split a dataset into training and test set."""

    def __init__(self, ratio: float, shuffle: bool = False):
        """Initialize the Split.

        Args:
            ratio: Ratio of the training set.
            shuffle: Whether to shuffle the dataset before splitting.
        """
        if ratio < 0.0 or ratio > 1.0:
            raise ValueError("ratio must be between 0 and 1")
        self.ratio = ratio
        self.shuffle = shuffle

    def __call__(
        self, dataset: pl.DataFrame
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        pivot = int(len(dataset) * self.ratio)
        splits = _split(
            dataset,
            lengths=[pivot, len(dataset) - pivot],
            shuffle=self.shuffle,
        )
        return splits[0], splits[1]
