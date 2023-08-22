import unittest

import polars as pl
import os
from pathlib import Path


from agenc.data.metadata import AgencMetadata, AgencMetadatum, AgencFeature

# Get the current script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))


class TestDataLoading(unittest.TestCase):
    def setUp(self):
        self.metadata = AgencMetadata(
            data_path=[
                Path(
                    os.path.join(
                        script_dir,
                        "..",
                        "..",
                        "..",
                        "examples",
                        "failure_time_prediction",
                        "data",
                        "processed_data.csv",
                    )
                ),
                Path(
                    os.path.join(
                        script_dir,
                        "..",
                        "..",
                        "..",
                        "examples",
                        "failure_time_prediction",
                        "data",
                        "processed_data_2.csv",
                    )
                ),
            ],
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

    def test_dataframe_concatenation(self):
        concatenated_data_frame = self.metadata.load_dataset()
        individual_data_frames = []
        for path in self.metadata.data_path:
            individual_data_frames.append(pl.read_csv(path))
        # check that the dataframes are concatenated correctly
        self.assertTrue(
            concatenated_data_frame.shape[0]
            == sum([df.shape[0] for df in individual_data_frames])
        )
        self.assertTrue(
            concatenated_data_frame.shape[1]
            == individual_data_frames[0].shape[1]
        )


if __name__ == "__main__":
    unittest.main()
