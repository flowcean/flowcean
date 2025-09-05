import unittest

import polars as pl
from polars.testing import assert_frame_equal

from flowcean.polars import ScaleToRange


class ScaleToRangeTransform(unittest.TestCase):
    def test_scale_fit(self) -> None:
        transform = ScaleToRange()

        data_frame = pl.DataFrame(
            {
                "a": [0.0, 0.5, 1.0],
                "b": [2.0, 3.0, 1.0],
                "c": [-2.0, -3.0, -1.0],
            },
        )
        transform.fit(data_frame.lazy())
        transformed_data = transform(data_frame.lazy()).collect()

        assert_frame_equal(
            transformed_data,
            pl.DataFrame(
                {
                    "a": [-1.0, 0.0, 1.0],
                    "b": [0.0, 1.0, -1.0],
                    "c": [0.0, -1.0, 1.0],
                },
            ),
            check_column_order=False,
        )


if __name__ == "__main__":
    unittest.main()
