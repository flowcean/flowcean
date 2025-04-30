import random

import polars as pl

from flowcean.core.data import Data
from flowcean.core.environment.incremental import IncrementalEnvironment
from flowcean.core.environment.stepable import Finished
from flowcean.core.tool.testing.generator.range import FeatureValue


class StochasticGenerator(IncrementalEnvironment):
    """A generator that produces random tests based on given values."""

    data: pl.DataFrame
    count: int

    def __init__(
        self,
        values: list[FeatureValue],
        *,
        number_test_cases: int | None = None,
        seed: int = 0,
    ) -> None:
        """Initialize the stochastic generator.

        Args:
            values: A list of feature values to generate random values from.
                Each entry must be associated with exactly one input feature
                of the model that shall be tested.
            number_test_cases: The number of test cases to generate. If None,
                the generator will run indefinitely.
            seed: The seed for the random number generator. The default is 0,
                which means a random seed will be used.
        """
        super().__init__()

        # Check if no duplicate feature names are present
        feature_names_count: dict[str, int] = {}
        for value in values:
            feature_names_count[value.feature_name] = (
                feature_names_count.get(
                    value.feature_name,
                    0,
                )
                + 1
            )
        if len(feature_names_count) != len(values):
            msg = "Duplicate feature names found in ranges: "
            msg += ", ".join(
                f"{feature_name}: {count}"
                for feature_name, count in feature_names_count.items()
                if count > 1
            )
            raise ValueError(msg)

        # Seed all ranges
        rng = random.Random(seed) if seed != 0 else random.Random()
        for value in values:
            value.set_seed(rng.randint(0, 2**32 - 1))

        self.values = values
        self.count = 0
        self.number_test_cases = number_test_cases
        # Perform the first step to initialize the generator
        self.step()

    def num_steps(self) -> int | None:
        return self.number_test_cases

    def step(self) -> None:
        self.count += 1
        if (
            self.number_test_cases is not None
            and self.count > self.number_test_cases
        ):
            raise Finished
        self.data = pl.DataFrame(
            {value.feature_name: value() for value in self.values},
        )

    def _observe(self) -> Data:
        return self.data.lazy()
