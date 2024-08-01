import unittest

import polars as pl
from polars.testing import assert_frame_equal

from flowcean.core.environment import CombineEnvironment
from flowcean.environments.dataset import Dataset


class TestCombine(unittest.TestCase):
    def test_combine_environment(self) -> None:
        dataset1 = Dataset(
            pl.DataFrame(
                {
                    "A": [1, 2],
                },
            )
        )

        dataset2 = Dataset(
            pl.DataFrame(
                {
                    "B": [3, 4],
                },
            )
        )

        combine = CombineEnvironment(dataset1, dataset2)

        assert_frame_equal(
            combine.get_data(),
            pl.DataFrame(
                {
                    "A": [1, 2],
                    "B": [3, 4],
                },
            ),
        )

    def test_stack_method(self) -> None:
        dataset1 = Dataset(
            pl.DataFrame(
                {
                    "A": [1, 2],
                },
            )
        )

        dataset2 = Dataset(
            pl.DataFrame(
                {
                    "B": [3, 4],
                },
            )
        )

        combine = dataset1.extend(dataset2)

        assert_frame_equal(
            combine.get_data(),
            pl.DataFrame(
                {
                    "A": [1, 2],
                    "B": [3, 4],
                },
            ),
        )


if __name__ == "__main__":
    unittest.main()
