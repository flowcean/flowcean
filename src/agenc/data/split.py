from collections.abc import Sequence
from itertools import accumulate
import polars as pl


def split(
    dataset: pl.DataFrame,
    lengths: Sequence[int],
    shuffle: bool = True,
) -> list[pl.DataFrame]:
    shuffled_dataset = dataset.sample(fraction=1.0, shuffle=shuffle)

    return [
        shuffled_dataset.slice(offset - length, length)
        for offset, length in zip(accumulate(lengths), lengths)
    ]


def train_test_split(
    dataset: pl.DataFrame,
    ratio: float,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    if ratio < 0.0 or ratio > 1.0:
        raise ValueError("ratio must be between 0 and 1")

    pivot = int(len(dataset) * ratio)
    splits = split(dataset, [pivot, len(dataset) - pivot], shuffle=True)
    return splits[0], splits[1]
