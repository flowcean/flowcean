import unittest
from typing import Any

import polars as pl
from polars.testing import assert_frame_equal

from flowcean.transforms.batch import BatchTransform, RowTransform


class MeanTransform(RowTransform):
    def apply(self, row: tuple[Any, ...]) -> pl.DataFrame:
        return pl.DataFrame({"mean": [sum(row) / len(row)]})


class MaxTransform(RowTransform):
    def apply(self, row: tuple[Any, ...]) -> pl.DataFrame:
        return pl.DataFrame({"max": [max(row)]})


class TestBatchTransform(unittest.TestCase):
    def setUp(self) -> None:
        self.mean_transform = MeanTransform()
        self.max_transform = MaxTransform()

        self.feature = "feature"
        self.child_transforms = [self.mean_transform, self.max_transform]
        self.batch_transform = BatchTransform(
            self.feature, self.child_transforms
        )

    def test_apply_single_column(self) -> None:
        data = pl.DataFrame({"feature": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]})

        expected_output = pl.DataFrame(
            {
                "feature": [
                    pl.concat(
                        [
                            self.mean_transform.apply((1, 2, 3)),
                            self.max_transform.apply((1, 2, 3)),
                        ],
                        how="horizontal",
                    ),
                    pl.concat(
                        [
                            self.mean_transform.apply((4, 5, 6)),
                            self.max_transform.apply((4, 5, 6)),
                        ],
                        how="horizontal",
                    ),
                    pl.concat(
                        [
                            self.mean_transform.apply((7, 8, 9)),
                            self.max_transform.apply((7, 8, 9)),
                        ],
                        how="horizontal",
                    ),
                ]
            }
        )

        output = self.batch_transform.apply(data)

        assert_frame_equal(output, expected_output)

    def test_map_element(self) -> None:
        pc = (1, 2, 3)

        expected_output = pl.concat(
            [self.mean_transform.apply(pc), self.max_transform.apply(pc)],
            how="horizontal",
        )

        output = self.batch_transform.map_element(pc)

        assert_frame_equal(output, expected_output)


if __name__ == "__main__":
    unittest.main()
