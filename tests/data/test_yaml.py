import tempfile
import unittest
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from flowcean.core.environment import NotLoadedError
from flowcean.environments.yaml import YamlDataLoader


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

                dataloader = YamlDataLoader(path=Path(f.name))
                with pytest.raises(NotLoadedError):
                    dataloader.get_data()
                dataloader.load()
                loaded_data = dataloader.get_data()
                loaded_data = dataloader.get_data()
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

                dataloader = YamlDataLoader(path=Path(f.name))
                dataloader.load()
                loaded_data = dataloader.get_data()
                assert_frame_equal(loaded_data, data)
            finally:
                f.close()
                Path(f.name).unlink()


if __name__ == "__main__":
    unittest.main()
