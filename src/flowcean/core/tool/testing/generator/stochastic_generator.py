import random

import polars as pl

from flowcean.core.data import Data
from flowcean.core.environment.incremental import IncrementalEnvironment
from flowcean.core.environment.stepable import Finished
from flowcean.core.tool.testing.generator.range import Range


class StochasticGenerator(IncrementalEnvironment):
    """A generator that produces random tests based on given ranges."""

    data: pl.DataFrame
    count: int

    def __init__(
        self,
        ranges: list[Range],
        *,
        number_test_cases: int | None = None,
        seed: int = 0,
    ) -> None:
        """Initialize the stochastic generator.

        Args:
            ranges: A list of ranges to generate random values from.
                Each range must be associated with exactly one input feature
                of the model that shall be tested.
            number_test_cases: The number of test cases to generate. If None,
                the generator will run indefinitely.
            seed: The seed for the random number generator. The default is 0,
                which means a random seed will be used.
        """
        super().__init__()

        # Check if no duplicate feature names are present
        feature_names_count: dict[str, int] = {}
        for range_ in ranges:
            feature_names_count[range_.feature_name] = (
                feature_names_count.get(
                    range_.feature_name,
                    0,
                )
                + 1
            )
        if len(feature_names_count) != len(ranges):
            msg = "Duplicate feature names found in ranges: "
            msg += ", ".join(
                f"{feature_name}: {count}"
                for feature_name, count in feature_names_count.items()
                if count > 1
            )
            raise ValueError(msg)

        # Seed all ranges
        rng = random.Random(seed) if seed != 0 else random.Random()
        for range_ in ranges:
            range_.set_seed(rng.randint(0, 2**32 - 1))

        self.ranges = ranges
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
            {range_.feature_name: range_() for range_ in self.ranges},
        )

    def _observe(self) -> Data:
        return self.data.lazy()
