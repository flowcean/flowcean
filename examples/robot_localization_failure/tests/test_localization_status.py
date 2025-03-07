import unittest
from unittest.mock import patch

import polars as pl
from custom_transforms.localization_status import LocalizationStatus


class TestLocalizationStatus(unittest.TestCase):
    def setUp(self) -> None:
        # Mock logger to avoid actual logging during tests
        with patch("logging.getLogger"):
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
                "const": [
                    1,
                ],
            },
        )

    def create_lazy_frame(
        self,
        position_data: list,
        heading_data: list,
    ) -> pl.LazyFrame:
        """Helper to create a Polars LazyFrame."""
        return pl.LazyFrame(
            {
                "/position_error": [position_data],
                "/heading_error": [heading_data],
            },
        )

    def test_apply(self) -> None:
        """Test from your original code: basic apply functionality."""
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

    def test_basic_localization_status(self) -> None:
        """Test basic computation with integer timestamps."""
        position_data = [
            {"time": 1000, "value": {"data": 0.9}},
            {"time": 2000, "value": {"data": 1.5}},
        ]
        heading_data = [
            {"time": 1000, "value": {"data": 1.1}},
            {"time": 2000, "value": {"data": 0.8}},
        ]
        data = self.create_lazy_frame(position_data, heading_data)

        result = self.transform.apply(data).collect()
        expected = [
            {"time": 1000, "value": {"data": 0}},  # Both below threshold
            {"time": 2000, "value": {"data": 1}},  # Position above threshold
        ]
        assert result["isDelocalized"].to_list() == [expected]

    def test_float_timestamps(self) -> None:
        """Test localization status with float timestamps."""
        position_data = [
            {"time": 1000.5, "value": {"data": 0.9}},
            {"time": 2000.7, "value": {"data": 1.5}},
        ]
        heading_data = [
            {"time": 1000.5, "value": {"data": 1.1}},
            {"time": 2000.7, "value": {"data": 0.8}},
        ]
        data = self.create_lazy_frame(position_data, heading_data)

        result = self.transform.apply(data).collect()
        expected = [
            {"time": 1000.5, "value": {"data": 0}},
            {"time": 2000.7, "value": {"data": 1}},
        ]
        assert result["isDelocalized"].to_list() == [expected]

    def test_empty_lists(self) -> None:
        """Test behavior with empty input lists."""
        position_data = []
        heading_data = []
        data = self.create_lazy_frame(position_data, heading_data)

        result = self.transform.apply(data).collect()
        expected = [[]]  # Empty list inside a list due to Polars structure
        assert result["isDelocalized"].to_list() == expected

    def test_single_threshold_exceeded(self) -> None:
        """Test when only one threshold is exceeded."""
        position_data = [
            {"time": 1000, "value": {"data": 1.3}},  # Above threshold
            {"time": 2000, "value": {"data": 0.5}},  # Below threshold
        ]
        heading_data = [
            {"time": 1000, "value": {"data": 0.8}},  # Below threshold
            {"time": 2000, "value": {"data": 1.3}},  # Above threshold
        ]
        data = self.create_lazy_frame(position_data, heading_data)

        result = self.transform.apply(data).collect()
        expected = [
            {"time": 1000, "value": {"data": 1}},  # Position exceeds
            {"time": 2000, "value": {"data": 1}},  # Heading exceeds
        ]
        assert result["isDelocalized"].to_list() == [expected]

    def test_custom_thresholds(self) -> None:
        """Test with custom thresholds."""
        with patch("logging.getLogger"):
            custom_transform = LocalizationStatus(
                position_threshold=1.0,
                heading_threshold=1.0,
            )
        position_data = [
            {"time": 1000, "value": {"data": 0.9}},
            {"time": 2000, "value": {"data": 1.1}},
        ]
        heading_data = [
            {"time": 1000, "value": {"data": 1.1}},
            {"time": 2000, "value": {"data": 0.8}},
        ]
        data = self.create_lazy_frame(position_data, heading_data)

        result = custom_transform.apply(data).collect()
        expected = [
            {"time": 1000, "value": {"data": 1}},  # Heading exceeds
            {"time": 2000, "value": {"data": 1}},  # Position exceeds
        ]
        assert result["isDelocalized"].to_list() == [expected]


if __name__ == "__main__":
    unittest.main()
