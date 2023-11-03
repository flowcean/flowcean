import unittest

import polars as pl
import os
from pathlib import Path


from agenc.core import Experiment, Feature, Metadata


class TestFeature(unittest.TestCase):
    def test_feature_creation(self):
        feature = Feature(
            name="TestFeature",
            description="Description",
            kind="Scalar",
            minimum=0.1,
            maximum=0.9,
            quantity="Length",
            unit="meter",
        )
        self.assertEqual(feature.name, "TestFeature")
        self.assertEqual(feature.description, "Description")
        self.assertEqual(feature.kind, "Scalar")
        self.assertEqual(feature.minimum, 0.1)
        self.assertEqual(feature.maximum, 0.9)
        self.assertEqual(feature.quantity, "Length")
        self.assertEqual(feature.unit, "meter")


class TestMetadata(unittest.TestCase):
    def test_load_dataset_from_csv(self):
        # Create a temporary CSV file for testing
        data_file = "test_data.csv"
        data = pl.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        data.write_csv(data_file)

        metadata = Metadata(
            data_path=[data_file], test_data_path=[], features=[]
        )

        loaded_data = metadata.load_dataset()
        self.assertIsInstance(loaded_data, pl.DataFrame)
        self.assertEqual(len(loaded_data), 3)

    def test_load_test_dataset_from_csv(self):
        # Create temporary CSV files for testing
        test_data_file = "test_test_data.csv"
        data = pl.DataFrame({"A": [7, 8, 9], "B": [10, 11, 12]})
        data.write_csv(test_data_file)

        metadata = Metadata(
            data_path=[], test_data_path=[test_data_file], features=[]
        )

        loaded_data = metadata.load_test_dataset()
        self.assertIsInstance(loaded_data, pl.DataFrame)
        self.assertEqual(len(loaded_data), 3)


if __name__ == "__main__":
    unittest.main()
