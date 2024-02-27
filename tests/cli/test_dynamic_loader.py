import unittest

import pytest
from agenc.cli._dynamic_loader import load_and_create
from agenc.cli.experiment import InstanceSpecification


class TestLoadInstanceFunction(unittest.TestCase):
    def test_load_instance_with_kwargs(self) -> None:
        instance = load_and_create(
            "mock.MyTransform",
            {"factor": 1},
        )

        assert instance.factor == 1

    def test_load_instance_handles_exception(self) -> None:
        with pytest.raises(ImportError):
            load_and_create(
                "nonexistent_module.NonExistentClass",
                {"arg": "value"},
            )

    def test_via_instance_specification(self) -> None:
        specification = InstanceSpecification(
            class_path="mock.MyTransform",
            arguments={"factor": 1},
        )
        instance = specification.create()

        assert instance.factor == 1


if __name__ == "__main__":
    unittest.main()
