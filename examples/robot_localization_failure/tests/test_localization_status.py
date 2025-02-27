import unittest

import polars as pl
from custom_transforms.localization_status import (
    LocalizationStatus,
)


class TestLocalizationStatus(unittest.TestCase):
    def setUp(self) -> None:
        self.transform = LocalizationStatus(
            position_error_feature_name="/position_error",
            heading_error_feature_name="/heading_error",
            position_threshold=1.2,
            heading_threshold=1.2,
        )

        self.data_frame = pl.DataFrame(
            {
                "/position_error": [
                    [
                        {"time": 1000, "value": {"data": 0.9}},
                        {"time": 2000, "value": {"data": 1.5}},
                    ],
                ],
                "/heading_error": [
                    [
                        {"time": 1000, "value": {"data": 1.1}},
                        {"time": 2000, "value": {"data": 0.8}},
                    ],
                ],
                "const": [1],
            },
        )

    def test_apply(self) -> None:
        transformed_data = self.transform.apply(
            self.data_frame.lazy(),
        ).collect()

        assert "isDelocalized" in transformed_data.columns

        is_delocalized_list = transformed_data["isDelocalized"][0]
        assert len(is_delocalized_list) == 2

        assert is_delocalized_list[0]["time"] == 1000
        assert is_delocalized_list[0]["value"]["data"] == 0

        assert is_delocalized_list[1]["time"] == 2000
        assert is_delocalized_list[1]["value"]["data"] == 1


if __name__ == "__main__":
    unittest.main()
