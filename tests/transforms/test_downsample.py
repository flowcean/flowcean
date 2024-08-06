import unittest
from datetime import datetime

import polars as pl
from polars.testing import assert_frame_equal

from flowcean.transforms import Downsample


class DownsampleTransform(unittest.TestCase):
    def test_upsample_first(self) -> None:
        transform = Downsample(
            time_column="time",
            sampling_rate="2s",
            value="first",
        )

        data_frame = pl.DataFrame(
            {
                "time": [
                    datetime(2024, 1, 1, 0, 0, 0),  # noqa: DTZ001
                    datetime(2024, 1, 1, 0, 0, 1),  # noqa: DTZ001
                    datetime(2024, 1, 1, 0, 0, 2),  # noqa: DTZ001
                    datetime(2024, 1, 1, 0, 0, 3),  # noqa: DTZ001
                    datetime(2024, 1, 1, 0, 0, 4),  # noqa: DTZ001
                ],
                "feature_a": [1.0, 2.0, 3.0, 4.0, 5.0],
                "feature_b": [0.0, 1.0, 2.0, 3.0, 4.0],
                "constant": [1.0, 1.0, 1.0, 1.0, 1.0],
            },
        ).set_sorted("time")
        transformed_data = transform.transform(data_frame)
        assert_frame_equal(
            transformed_data,
            pl.DataFrame(
                {
                    "time": [
                        datetime(2024, 1, 1, 0, 0, 0),  # noqa: DTZ001
                        datetime(2024, 1, 1, 0, 0, 2),  # noqa: DTZ001
                        datetime(2024, 1, 1, 0, 0, 4),  # noqa: DTZ001
                    ],
                    "feature_a": [1.0, 3.0, 5.0],
                    "feature_b": [0.0, 2.0, 4.0],
                    "constant": [1.0, 1.0, 1.0],
                },
            ),
        )

    def test_upsample_mean(self) -> None:
        transform = Downsample(
            time_column="time",
            sampling_rate="2s",
            value="mean",
        )

        data_frame = pl.DataFrame(
            {
                "time": [
                    datetime(2024, 1, 1, 0, 0, 0),  # noqa: DTZ001
                    datetime(2024, 1, 1, 0, 0, 1),  # noqa: DTZ001
                    datetime(2024, 1, 1, 0, 0, 2),  # noqa: DTZ001
                    datetime(2024, 1, 1, 0, 0, 3),  # noqa: DTZ001
                    datetime(2024, 1, 1, 0, 0, 4),  # noqa: DTZ001
                ],
                "feature_a": [1.0, 2.0, 3.0, 4.0, 5.0],
                "feature_b": [0.0, 1.0, 2.0, 3.0, 4.0],
                "constant": [1.0, 1.0, 1.0, 1.0, 1.0],
            },
        ).set_sorted("time")
        transformed_data = transform.transform(data_frame)
        assert_frame_equal(
            transformed_data,
            pl.DataFrame(
                {
                    "time": [
                        datetime(2024, 1, 1, 0, 0, 0),  # noqa: DTZ001
                        datetime(2024, 1, 1, 0, 0, 2),  # noqa: DTZ001
                        datetime(2024, 1, 1, 0, 0, 4),  # noqa: DTZ001
                    ],
                    "feature_a": [1.5, 3.5, 5.0],
                    "feature_b": [0.5, 2.5, 4.0],
                    "constant": [1.0, 1.0, 1.0],
                },
            ),
        )

    def test_upsample_last(self) -> None:
        transform = Downsample(
            time_column="time",
            sampling_rate="2s",
            value="last",
        )

        data_frame = pl.DataFrame(
            {
                "time": [
                    datetime(2024, 1, 1, 0, 0, 0),  # noqa: DTZ001
                    datetime(2024, 1, 1, 0, 0, 1),  # noqa: DTZ001
                    datetime(2024, 1, 1, 0, 0, 2),  # noqa: DTZ001
                    datetime(2024, 1, 1, 0, 0, 3),  # noqa: DTZ001
                    datetime(2024, 1, 1, 0, 0, 4),  # noqa: DTZ001
                ],
                "feature_a": [1.0, 2.0, 3.0, 4.0, 5.0],
                "feature_b": [0.0, 1.0, 2.0, 3.0, 4.0],
                "constant": [1.0, 1.0, 1.0, 1.0, 1.0],
            },
        ).set_sorted("time")
        transformed_data = transform.transform(data_frame)
        assert_frame_equal(
            transformed_data,
            pl.DataFrame(
                {
                    "time": [
                        datetime(2024, 1, 1, 0, 0, 0),  # noqa: DTZ001
                        datetime(2024, 1, 1, 0, 0, 2),  # noqa: DTZ001
                        datetime(2024, 1, 1, 0, 0, 4),  # noqa: DTZ001
                    ],
                    "feature_a": [2.0, 4.0, 5.0],
                    "feature_b": [1.0, 3.0, 4.0],
                    "constant": [1.0, 1.0, 1.0],
                },
            ),
        )


if __name__ == "__main__":
    unittest.main()
