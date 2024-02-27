import logging
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import polars as pl
from agenc.cli.main import load_data, main, parse_arguments
from agenc.core.data_loader import DataLoader
from agenc.data.split import TrainTestSplit
from polars.testing import assert_frame_equal

EXPERIMENT = Path(__file__).parent / "experiment.yaml"


class TestEntrypoint(unittest.TestCase):
    def test_entrypoint(self) -> None:
        result = subprocess.run(
            "agenc --help",
            shell=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0

    def test_arguments(self) -> None:
        sys.argv = [
            "agenc",
            "--configuration",
            "path/to/configuration",
            "--verbose",
            "--experiment",
            str(EXPERIMENT),
        ]
        arguments = parse_arguments()
        assert arguments.experiment == EXPERIMENT
        assert arguments.configuration == Path("path/to/configuration")
        assert arguments.verbose == logging.DEBUG

    def test_main(self) -> None:
        sys.argv = [
            "agenc",
            "--verbose",
            "--experiment",
            str(EXPERIMENT),
        ]
        main()

    def test_load_data_with_data_loader(self) -> None:
        data_loader_mock = MagicMock(spec=DataLoader)
        test_data_loader_mock = MagicMock(spec=DataLoader)

        data_loader_mock.load.return_value = pl.DataFrame(
            {"feature": [1, 2, 3]}
        )
        test_data_loader_mock.load.return_value = pl.DataFrame(
            {"target": [0, 1, 1]}
        )

        train_data, test_data = load_data(
            data_loader_mock, test_data_loader_mock
        )

        assert isinstance(train_data, pl.DataFrame)
        assert isinstance(test_data, pl.DataFrame)
        assert_frame_equal(
            train_data,
            pl.DataFrame({"feature": [1, 2, 3]}),
        )
        assert_frame_equal(
            test_data,
            pl.DataFrame({"target": [0, 1, 1]}),
        )

    def test_load_data_with_train_test_split(self) -> None:
        data_loader_mock = MagicMock(spec=DataLoader)
        train_test_split_mock = MagicMock(spec=TrainTestSplit)

        data_loader_mock.load.return_value = pl.DataFrame(
            {"feature": [1, 2, 3]}
        )
        train_test_split_mock.return_value = (
            pl.DataFrame({"train_feature": [4, 5, 6]}),
            pl.DataFrame({"test_target": [1, 0, 1]}),
        )

        train_data, test_data = load_data(
            data_loader_mock, train_test_split_mock
        )

        assert isinstance(train_data, pl.DataFrame)
        assert isinstance(test_data, pl.DataFrame)
        assert_frame_equal(
            train_data,
            pl.DataFrame({"train_feature": [4, 5, 6]}),
        )
        assert_frame_equal(
            test_data,
            pl.DataFrame({"test_target": [1, 0, 1]}),
        )


if __name__ == "__main__":
    unittest.main()
