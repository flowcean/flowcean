import unittest

import polars as pl
from polars.testing import assert_frame_equal

from flowcean.polars import DataFrame, JoinedOfflineEnvironment


class TestCombine(unittest.TestCase):
    def test_combine_environment(self) -> None:
        dataset1 = DataFrame(
            pl.DataFrame(
                {
                    "A": [1, 2],
                },
            ),
        )

        dataset2 = DataFrame(
            pl.DataFrame(
                {
                    "B": [3, 4],
                },
            ),
        )

        combine = JoinedOfflineEnvironment((dataset1, dataset2))

        assert isinstance(combine, JoinedOfflineEnvironment)

    def test_join_results(self) -> None:
        dataset1 = DataFrame(
            pl.DataFrame(
                {
                    "A": [1, 2],
                },
            ),
        )

        dataset2 = DataFrame(
            pl.DataFrame(
                {
                    "B": [3, 4],
                },
            ),
        )

        combine = JoinedOfflineEnvironment((dataset1, dataset2))

        assert_frame_equal(
            combine.observe().collect(),
            pl.DataFrame(
                {
                    "A": [1, 2],
                    "B": [3, 4],
                },
            ),
        )


if __name__ == "__main__":
    unittest.main()
