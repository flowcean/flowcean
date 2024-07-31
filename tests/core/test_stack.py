import unittest

import polars as pl
from polars.testing import assert_frame_equal

from flowcean.core.environment.stack import StackEnvironment
from flowcean.environments.dataset import Dataset


class TestStack(unittest.TestCase):
    def test_stack_environment(self) -> None:
        dataset1 = Dataset(
            pl.DataFrame(
                {
                    "A": [1, 2],
                    "B": [5, 6],
                },
            )
        )

        dataset2 = Dataset(
            pl.DataFrame(
                {
                    "A": [3, 4],
                    "B": [7, 8],
                },
            )
        )

        stack = StackEnvironment(dataset1, dataset2)

        assert_frame_equal(
            stack.get_data(),
            pl.DataFrame(
                {
                    "A": [1, 2, 3, 4],
                    "B": [5, 6, 7, 8],
                },
            ),
        )

    def test_stack_method(self) -> None:
        dataset1 = Dataset(
            pl.DataFrame(
                {
                    "A": [1, 2],
                    "B": [5, 6],
                },
            )
        )

        dataset2 = Dataset(
            pl.DataFrame(
                {
                    "A": [3, 4],
                    "B": [7, 8],
                },
            )
        )

        stack = dataset1.stack(dataset2)

        assert_frame_equal(
            stack.get_data(),
            pl.DataFrame(
                {
                    "A": [1, 2, 3, 4],
                    "B": [5, 6, 7, 8],
                },
            ),
        )


if __name__ == "__main__":
    unittest.main()
