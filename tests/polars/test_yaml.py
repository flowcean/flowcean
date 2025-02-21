import tempfile
import unittest
from pathlib import Path

import polars as pl
from polars.testing import assert_frame_equal

from flowcean.polars.environments.dataframe import DataFrame


class TestYamlDataLoader(unittest.TestCase):
    def test_single_sample(self) -> None:
        yaml_content = """
        A: 1
        B: 2
        C: 3
        """

        data = pl.DataFrame({"A": 1, "B": 2, "C": 3})

        with tempfile.NamedTemporaryFile(delete=False) as f:
            try:
                f.write(bytes(yaml_content, "UTF-8"))
                f.close()

                dataloader = DataFrame.from_yaml(path=Path(f.name))
                loaded_data = dataloader.observe().collect()
                assert_frame_equal(loaded_data, data)
            finally:
                f.close()
                Path(f.name).unlink()

    def test_multiple_samples(self) -> None:
        yaml_content = """
        - A: 1
          B: 2
          C: 3
        - A: 4
          B: 5
          C: 6
        """

        data = pl.DataFrame({"A": [1, 4], "B": [2, 5], "C": [3, 6]})

        with tempfile.NamedTemporaryFile(delete=False) as f:
            try:
                f.write(bytes(yaml_content, "UTF-8"))
                f.close()

                dataloader = DataFrame.from_yaml(path=Path(f.name))
                loaded_data = dataloader.observe().collect()
                assert_frame_equal(loaded_data, data)
            finally:
                f.close()
                Path(f.name).unlink()


if __name__ == "__main__":
    unittest.main()
