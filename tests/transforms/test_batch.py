import unittest
from collections.abc import Callable
from typing import Any

import polars as pl
from polars.testing import assert_frame_equal

from flowcean.transforms.batch import BatchTransform, RowTransform


class TestRowTransform(RowTransform):
    def __init__(self, name: str, transform_fn: Callable) -> None:
        self.name = name
        self.transform_fn = transform_fn

    def apply(self, row: tuple[Any, ...]) -> pl.DataFrame:
        return pl.DataFrame({self.name: [self.transform_fn(row)]})


class BatchTransformTest(unittest.TestCase):
    def test_single_transform(self) -> None:
        sum_transform = TestRowTransform("sum", lambda x: sum(x))

        transform = BatchTransform(
            feature="feature", child_transforms=[sum_transform]
        )

        data_frame = pl.DataFrame(
            {
                "feature": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                "time": [0, 1, 2],
            }
        )

        transformed_data = transform.apply(data_frame)

        expected_data = pl.DataFrame(
            {
                "feature": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                "time": [0, 1, 2],
                "sum": [6, 15, 24],
            }
        )

        assert_frame_equal(transformed_data, expected_data)

    def test_multiple_transforms(self) -> None:
        sum_transform = TestRowTransform("sum", lambda x: sum(x))
        max_transform = TestRowTransform("max", lambda x: max(x))

        transform = BatchTransform(
            feature="feature", child_transforms=[sum_transform, max_transform]
        )

        data_frame = pl.DataFrame(
            {
                "feature": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                "time": [0, 1, 2],
            }
        )

        transformed_data = transform.apply(data_frame)

        expected_data = pl.DataFrame(
            {
                "feature": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                "time": [0, 1, 2],
                "sum": [6, 15, 24],
                "max": [3, 6, 9],
            }
        )

        assert_frame_equal(transformed_data, expected_data)

    def test_empty_child_transforms(self) -> None:
        transform = BatchTransform(feature="feature", child_transforms=[])

        data_frame = pl.DataFrame(
            {
                "feature": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                "time": [0, 1, 2],
            }
        )

        transformed_data = transform.apply(data_frame)

        assert_frame_equal(transformed_data, data_frame)


if __name__ == "__main__":
    unittest.main()
