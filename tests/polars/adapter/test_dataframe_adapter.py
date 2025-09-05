import tempfile
import unittest

import polars as pl
from polars.testing import assert_frame_equal

from flowcean.core.environment.incremental import Finished
from flowcean.polars import DataFrame
from flowcean.polars.adapter import DataFrameAdapter


class TestDataFrameAdapter(unittest.TestCase):
    def test_adapter(self) -> None:
        data = pl.DataFrame(
            {
                "A": [1, 2, 3],
                "B": [4, 5, 6],
            },
        )
        send_data = pl.DataFrame(
            {
                "data": [1, 2, 3],
            },
        )

        pulled_data = pl.DataFrame()
        with tempfile.NamedTemporaryFile() as f:
            adapter = DataFrameAdapter(
                DataFrame(data),
                input_features=["A", "B"],
                result_path=f.name,
            )

            # Start the adapter
            adapter.start()

            try:
                while True:
                    pulled_data = pl.concat(
                        [pulled_data, adapter.get_data().collect()],
                        how="vertical",
                    )
            except Finished:
                pass

            # Send some data back to the adapter
            for row in send_data.iter_rows(named=True):
                adapter.send_data(pl.DataFrame(row).lazy())

            # Stop the adapter
            adapter.stop()

            assert_frame_equal(
                pulled_data,
                data,
            )

            assert_frame_equal(
                adapter.result_df,
                send_data,
            )


if __name__ == "__main__":
    unittest.main()
