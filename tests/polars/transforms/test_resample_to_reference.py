import unittest

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from flowcean.polars.transforms.resample_to_reference import (
    ResampleToReference,
    resample_to_reference,
)


class TestResampleToReferenceFunction(unittest.TestCase):
    def test_basic_forward_fill(self) -> None:
        """Test basic resampling with forward-fill behavior."""
        data = pl.DataFrame(
            {
                "reference": [
                    [
                        {"time": 0.0, "value": {"x": 10.0}},
                        {"time": 5.0, "value": {"x": 50.0}},
                        {"time": 10.0, "value": {"x": 100.0}},
                    ],
                ],
                "sensor": [
                    [
                        {"time": 0.0, "value": {"x": 1.0}},
                        {"time": 3.0, "value": {"x": 2.0}},
                        {"time": 7.0, "value": {"x": 3.0}},
                    ],
                ],
            },
        )

        result = resample_to_reference(
            data.lazy(),
            features=["reference", "sensor"],
            reference="reference",
            name="aligned",
        ).collect()

        expected = pl.DataFrame(
            {
                "aligned": [
                    [
                        {
                            "time": 0.0,
                            "value": {
                                "reference/x": 10.0,
                                "sensor/x": 1.0,
                            },
                        },
                        {
                            "time": 5.0,
                            "value": {
                                "reference/x": 50.0,
                                "sensor/x": 2.0,
                            },
                        },
                        {
                            "time": 10.0,
                            "value": {
                                "reference/x": 100.0,
                                "sensor/x": 3.0,
                            },
                        },
                    ],
                ],
            },
        )

        assert_frame_equal(result, expected)

    def test_multiple_features(self) -> None:
        """Test resampling with multiple features."""
        data = pl.DataFrame(
            {
                "ref": [
                    [
                        {"time": 0.0, "value": {"x": 0.0}},
                        {"time": 2.0, "value": {"x": 2.0}},
                    ],
                ],
                "a": [
                    [
                        {"time": 0.0, "value": {"x": 100.0}},
                        {"time": 1.0, "value": {"x": 200.0}},
                    ],
                ],
                "b": [
                    [
                        {"time": 0.0, "value": {"x": 10.0}},
                        {"time": 3.0, "value": {"x": 30.0}},
                    ],
                ],
            },
        )

        result = resample_to_reference(
            data.lazy(),
            features=["ref", "a", "b"],
            reference="ref",
            name="output",
        ).collect()

        expected = pl.DataFrame(
            {
                "output": [
                    [
                        {
                            "time": 0.0,
                            "value": {
                                "ref/x": 0.0,
                                "a/x": 100.0,
                                "b/x": 10.0,
                            },
                        },
                        {
                            "time": 2.0,
                            "value": {
                                "ref/x": 2.0,
                                "a/x": 200.0,
                                "b/x": 10.0,
                            },
                        },
                    ],
                ],
            },
        )

        assert_frame_equal(result, expected)

    def test_2d_timeseries_data(self) -> None:
        """Test resampling with 2D time series data."""
        data = pl.DataFrame(
            {
                "ref": [
                    [
                        {"time": 0.0, "value": {"x": 0.0, "y": 0.0}},
                        {"time": 5.0, "value": {"x": 5.0, "y": 5.0}},
                        {"time": 10.0, "value": {"x": 10.0, "y": 10.0}},
                    ],
                ],
                "sensor": [
                    [
                        {"time": 0.0, "value": {"x": 1.0, "y": 10.0}},
                        {"time": 3.0, "value": {"x": 2.0, "y": 20.0}},
                        {"time": 12.0, "value": {"x": 4.0, "y": 40.0}},
                    ],
                ],
            },
        )

        result = resample_to_reference(
            data.lazy(),
            features=["ref", "sensor"],
            reference="ref",
            name="output",
        ).collect()

        expected = pl.DataFrame(
            {
                "output": [
                    [
                        {
                            "time": 0.0,
                            "value": {
                                "ref/x": 0.0,
                                "ref/y": 0.0,
                                "sensor/x": 1.0,
                                "sensor/y": 10.0,
                            },
                        },
                        {
                            "time": 5.0,
                            "value": {
                                "ref/x": 5.0,
                                "ref/y": 5.0,
                                "sensor/x": 2.0,
                                "sensor/y": 20.0,
                            },
                        },
                        {
                            "time": 10.0,
                            "value": {
                                "ref/x": 10.0,
                                "ref/y": 10.0,
                                "sensor/x": 2.0,
                                "sensor/y": 20.0,
                            },
                        },
                    ],
                ],
            },
        )

        assert_frame_equal(result, expected)

    def test_reference_not_in_features(self) -> None:
        """Test that error is raised when reference not in features."""
        data = pl.DataFrame(
            {
                "ref": [
                    [
                        {"time": 0.0, "value": {"x": 0.0}},
                    ],
                ],
                "sensor": [
                    [
                        {"time": 0.0, "value": {"x": 1.0}},
                    ],
                ],
            },
        )

        with pytest.raises(
            ValueError,
            match="reference 'ref' not in columns",
        ) as context:
            resample_to_reference(
                data.lazy(),
                features=["sensor"],
                reference="ref",
                name="output",
            ).collect()

        assert "reference 'ref' not in columns" in str(context.value)


class TestResampleToReferenceTransform(unittest.TestCase):
    def test_simple_transform(self) -> None:
        """Test the Transform class with basic data."""
        transform = ResampleToReference(
            reference="ref",
            features=["ref", "data"],
            name="resampled",
        )

        data = pl.DataFrame(
            {
                "ref": [
                    [
                        {"time": 0.0, "value": {"x": 0.0}},
                        {"time": 10.0, "value": {"x": 10.0}},
                    ],
                ],
                "data": [
                    [
                        {"time": 0.0, "value": {"x": 5.0}},
                        {"time": 5.0, "value": {"x": 15.0}},
                        {"time": 12.0, "value": {"x": 25.0}},
                    ],
                ],
            },
        )

        result = transform(data.lazy()).collect()

        expected = pl.DataFrame(
            {
                "resampled": [
                    [
                        {
                            "time": 0.0,
                            "value": {"ref/x": 0.0, "data/x": 5.0},
                        },
                        {
                            "time": 10.0,
                            "value": {"ref/x": 10.0, "data/x": 15.0},
                        },
                    ],
                ],
            },
        )

        assert_frame_equal(result, expected)

    def test_drop_behavior(self) -> None:
        """Test that original features are dropped when drop=True."""
        transform = ResampleToReference(
            reference="ref",
            features=["ref", "data"],
            name="resampled",
            drop=True,
        )

        data = pl.DataFrame(
            {
                "ref": [
                    [
                        {"time": 0.0, "value": {"x": 0.0}},
                    ],
                ],
                "data": [
                    [
                        {"time": 0.0, "value": {"x": 1.0}},
                    ],
                ],
                "other": [42],
            },
        )

        result = transform(data.lazy()).collect()

        assert "resampled" in result.columns
        assert "ref" not in result.columns
        assert "data" not in result.columns
        assert "other" in result.columns

    def test_no_drop_behavior(self) -> None:
        """Test that original features are kept when drop=False."""
        transform = ResampleToReference(
            reference="ref",
            features=["ref", "data"],
            name="resampled",
            drop=False,
        )

        data = pl.DataFrame(
            {
                "ref": [
                    [
                        {"time": 0.0, "value": {"x": 0.0}},
                    ],
                ],
                "data": [
                    [
                        {"time": 0.0, "value": {"x": 1.0}},
                    ],
                ],
            },
        )

        result = transform(data.lazy()).collect()

        assert "resampled" in result.columns
        assert "ref" in result.columns
        assert "data" in result.columns

    def test_features_none(self) -> None:
        """Test that all features are used when features=None."""
        transform = ResampleToReference(
            reference="ref",
            features=None,
            name="output",
        )

        data = pl.DataFrame(
            {
                "ref": [
                    [
                        {"time": 0.0, "value": {"x": 0.0}},
                    ],
                ],
                "a": [
                    [
                        {"time": 0.0, "value": {"x": 1.0}},
                    ],
                ],
                "b": [
                    [
                        {"time": 0.0, "value": {"x": 2.0}},
                    ],
                ],
            },
        )

        result = transform(data.lazy()).collect()

        assert "output" in result.columns

    def test_reference_not_in_features_transform(self) -> None:
        """Test that Transform raises error when reference not in features."""
        transform = ResampleToReference(
            reference="missing",
            features=["ref", "data"],
            name="output",
        )

        data = pl.DataFrame(
            {
                "ref": [
                    [
                        {"time": 0.0, "value": {"x": 0.0}},
                    ],
                ],
                "data": [
                    [
                        {"time": 0.0, "value": {"x": 1.0}},
                    ],
                ],
            },
        )

        with pytest.raises(
            ValueError,
            match="reference 'missing' must be included in features",
        ) as context:
            transform(data.lazy()).collect()

        assert "reference 'missing' must be included in features" in str(
            context.value,
        )

    def test_multiple_rows(self) -> None:
        """Test resampling with multiple rows in the dataset."""
        transform = ResampleToReference(
            reference="ref",
            features=["ref", "sensor"],
            name="aligned",
        )

        data = pl.DataFrame(
            {
                "ref": [
                    [
                        {"time": 0.0, "value": {"x": 0.0}},
                        {"time": 5.0, "value": {"x": 5.0}},
                    ],
                    [
                        {"time": 10.0, "value": {"x": 10.0}},
                        {"time": 13.0, "value": {"x": 13.0}},
                    ],
                ],
                "sensor": [
                    [
                        {"time": 0.0, "value": {"x": 100.0}},
                        {"time": 2.0, "value": {"x": 200.0}},
                    ],
                    [
                        {"time": 10.0, "value": {"x": 50.0}},
                        {"time": 14.0, "value": {"x": 150.0}},
                    ],
                ],
            },
        )

        result = transform(data.lazy()).collect()

        assert len(result) == 2
        assert "aligned" in result.columns


if __name__ == "__main__":
    unittest.main()
