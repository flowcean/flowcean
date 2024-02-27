import unittest
from pathlib import Path

from agenc.cli.runtime_configuration import (
    DEFAULT_BASE_CONFIG,
    get_configuration,
    load_from_file,
)

RUNTIME_CONFIGURATION = Path(__file__).parent / "runtime.yaml"


class TestRuntimeConfiguration(unittest.TestCase):
    def test_loading(self) -> None:
        configuration = get_configuration()
        assert (
            configuration["logging"]["version"]
            == DEFAULT_BASE_CONFIG["logging"]["version"]
        )

        load_from_file(RUNTIME_CONFIGURATION)
        assert configuration["logging"]["version"] == 2


if __name__ == "__main__":
    unittest.main()
