import unittest

import numpy as np
import polars as pl

from flowcean.polars import SignalFilter


class SignalFilterTransform(unittest.TestCase):
    def test_lowpass(self) -> None:
        transform = SignalFilter(
            ["feature_a"],
            "lowpass",
            filter_frequency=60,
        )

        time_vector = np.arange(0, 1, 1 / 200)
        data_frame = pl.DataFrame(
            {
                "feature_a": [
                    [
                        {
                            "time": t,
                            "value": np.sin(2 * np.pi * 5 * t)
                            + np.sin(2 * np.pi * 80 * t),
                        }
                        for t in time_vector
                    ],
                ],
                "scalar": [42],
            },
        )

        transformed_values = transform(data_frame.lazy()).collect()
        # Because the filter introduces a delay into the signal (group delay),
        # we cannot simply compare the transformed data to a given data frame.
        # Instead, the error between the expected and actual data is
        # calculated. If it's within limits, the filter has worked.

        # Get the transformed values
        transformed_values = (
            transformed_values.select(
                pl.col("feature_a").list.eval(
                    pl.first().struct.field("value"),
                ),
            )
            .item()
            .to_numpy()
        )

        expected_values = np.array(
            [np.sin(2 * np.pi * 5 * t) for t in time_vector],
        )

        mean_square_error = np.sum(
            np.power(expected_values - transformed_values, 2),
        ) / len(expected_values)

        assert mean_square_error <= 0.1

    def test_highpass(self) -> None:
        transform = SignalFilter(
            ["feature_a"],
            "highpass",
            filter_frequency=10,
        )

        time_vector = np.arange(0, 1, 1 / 200)
        data_frame = pl.DataFrame(
            {
                "feature_a": [
                    [
                        {
                            "time": t,
                            "value": np.sin(2 * np.pi * 5 * t)
                            + np.sin(2 * np.pi * 80 * t),
                        }
                        for t in time_vector
                    ],
                ],
                "scalar": [42],
            },
        )

        transformed_values = transform(data_frame.lazy()).collect()
        # Because the filter introduces a delay into the signal (group delay),
        # we cannot simply compare the transformed data to a given data frame.
        # Instead, the error between the expected and actual data is
        # calculated. If it's within limits, the filter has worked.

        # Get the transformed values
        transformed_values = (
            transformed_values.select(
                pl.col("feature_a").list.eval(
                    pl.first().struct.field("value"),
                ),
            )
            .item()
            .to_numpy()
        )

        expected_values = np.array(
            [np.sin(2 * np.pi * 80 * t) for t in time_vector],
        )

        mean_square_error = np.sum(
            np.power(expected_values - transformed_values, 2),
        ) / len(expected_values)

        assert mean_square_error <= 0.1


if __name__ == "__main__":
    unittest.main()
