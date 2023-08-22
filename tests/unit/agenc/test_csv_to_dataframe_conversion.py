import unittest

import polars as pl
import os

from agenc.data.metadata import AgencMetadata, AgencMetadatum, AgencFeature

# Get the current script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

class TestCSVToDataframeConversion(unittest.TestCase):
    def test_load_csvs(self):
        # test that the dataframes are loaded correctly
        metadata = AgencMetadata(
            data_paths=[os.path.join(script_dir, "..", "..", "..", "examples", "failure_time_prediction", "data", "processed_data.csv"),
                       os.path.join(script_dir, "..",  "..", "..", "examples", "failure_time_prediction", "data", "processed_data_2.csv")],
            columns=[
                AgencMetadatum(
                    name="a",
                    description="a description",
                    kind="continuous",
                    min=0.0,
                    max=1.0,
                    quantity="length",
                    unit="m",
                ),
                AgencMetadatum(
                    name="b",
                    description="b description",
                    kind="continuous",
                    min=0.0,
                    max=1.0,
                    quantity="length",
                    unit="m",
                ),
            ],
            features=[
                AgencFeature(
                    AgencMetadatum(
                        name="a",
                        description="a description",
                        kind="continuous",
                        min=0.0,
                        max=1.0,
                        quantity="length",
                        unit="m",
                    ),
                    import_str="agenc.features.identity.Identity",
                    params=[],
                ),
                AgencFeature(
                    AgencMetadatum(
                        name="b",
                        description="b description",
                        kind="continuous",
                        min=0.0,
                        max=1.0,
                        quantity="length",
                        unit="m",
                    ),
                    import_str="agenc.features.identity.Identity",
                    params=[],
                ),
            ],
        )
        data_frames = metadata.load_csvs()
        # check that the dataframes are loaded correctly
        self.assertEqual(len(data_frames), 2)
        # check that the dataframes have the correct data type
        self.assertIsInstance(data_frames[0], pl.DataFrame)



if __name__ == "__main__":
    unittest.main()
