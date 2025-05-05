import unittest

from flowcean.core.tool.testing.domain import Continuous


class TestContinuous(unittest.TestCase):
    def test_sampling(self) -> None:
        r = Continuous("feature", 0, 10)
        r.set_seed(0)

        sample = r()

        assert 0 <= sample <= 10, f"Sample {sample} is out of range [0, 10]"

    def test_to_discrete(self) -> None:
        r = Continuous("feature", 0, 10)
        r.set_seed(0)

        discrete = r.to_discrete(1)

        assert discrete.values[0] == 0
        assert discrete.values[-1] == 10

        assert len(discrete.values) == 11, (
            "Discrete domain should have 11 values"
        )

    def test_to_discrete_odd(self) -> None:
        r = Continuous("feature", 0, 10)
        r.set_seed(0)

        discrete = r.to_discrete(3)

        assert discrete.values[0] == 0
        assert discrete.values[-1] == 9

        assert len(discrete.values) == 4, (
            "Discrete domain should have 4 values",
        )


if __name__ == "__main__":
    unittest.main()
