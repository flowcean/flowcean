import unittest

import numpy as np
import polars as pl

from flowcean.transforms import Filter


class FilterTransform(unittest.TestCase):
    def test_lowpass(self) -> None:
        transform = Filter(
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
            }
        )

        transformed_data = transform.transform(data_frame)
        # As the filter introduces a delay to the signal (group-delay), we
        # cannot simply compare the transformed data to a given dataframe.
        # Instead the error between the expected and the real data is computed.
        # If that's within limits, the filter did work.

        # Get the transformed values
        transformed_data = (
            transformed_data.select(
                pl.col("feature_a").list.eval(pl.first().struct.field("value"))
            )
            .item()
            .to_numpy()
        )

        expected_data = np.array(
            [np.sin(2 * np.pi * 5 * t) for t in time_vector]
        )

        mean_square_error = np.sum(
            np.power(expected_data - transformed_data, 2)
        ) / len(expected_data)

        assert mean_square_error <= 0.1

    def test_highpass(self) -> None:
        transform = Filter(
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
            }
        )

        transformed_data = transform.transform(data_frame)
        # As the filter introduces a delay to the signal (group-delay), we
        # cannot simply compare the transformed data to a given dataframe.
        # Instead the error between the expected and the real data is computed.
        # If that's within limits, the filter did work.

        # Get the transformed values
        transformed_data = (
            transformed_data.select(
                pl.col("feature_a").list.eval(pl.first().struct.field("value"))
            )
            .item()
            .to_numpy()
        )

        expected_data = np.array(
            [np.sin(2 * np.pi * 80 * t) for t in time_vector]
        )

        mean_square_error = np.sum(
            np.power(expected_data - transformed_data, 2)
        ) / len(expected_data)

        assert mean_square_error <= 0.1


if __name__ == "__main__":
    unittest.main()
