import unittest

import polars as pl
from polars.testing import assert_frame_equal

from flowcean.core.environment.chained import ChainedOfflineEnvironments
from flowcean.environments.dataset import Dataset


class TestChain(unittest.TestCase):
    def test_chain_environment(self) -> None:
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

        chain = ChainedOfflineEnvironments([dataset1, dataset2])
        assert isinstance(chain, ChainedOfflineEnvironments)

    def test_chain_method(self) -> None:
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

        chain = dataset1.chain(dataset2)
        assert isinstance(chain, ChainedOfflineEnvironments)

    def test_step(self) -> None:
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

        chain = dataset1.chain(dataset2)

        assert_frame_equal(
            chain.observe(),
            dataset1.observe(),
        )
        chain.step()
        assert_frame_equal(
            chain.observe(),
            dataset2.observe(),
        )

    def test_collect(self) -> None:
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

        chain = dataset1.chain(dataset2)

        assert_frame_equal(
            chain.collect().observe(),
            pl.DataFrame({
                "A": [1,2,3,4],
                ""
            }),
        )
        chain.step()
        assert_frame_equal(
            chain.observe(),
            dataset2.observe(),
        )

if __name__ == "__main__":
    unittest.main()
