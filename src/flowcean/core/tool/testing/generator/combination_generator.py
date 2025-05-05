import functools
import itertools

import polars as pl

from flowcean.core.data import Data
from flowcean.core.environment.incremental import IncrementalEnvironment
from flowcean.core.environment.stepable import Finished
from flowcean.core.tool.testing.generator.range import Discrete


class CombinationGenerator(IncrementalEnvironment):
    """A generator that produces tests based on combination of ranges.

    This generator creates a test case for each combination of the provided
    value ranges. Each value range must be associated with exactly one input
    feature of the model that shall be tested.
    """

    data: pl.DataFrame
    number_test_cases: int

    def __init__(self, *value_ranges: Discrete) -> None:
        """Initialize the combination generator.

        Args:
            value_ranges: A list of value ranges to generate test cases
                from. Each entry must be associated with exactly one input
                feature of the model that shall be tested.
        """
        super().__init__()

        # Check if no duplicate feature names are present
        feature_names_count: dict[str, int] = {}
        for value_range in value_ranges:
            feature_names_count[value_range.feature_name] = (
                feature_names_count.get(
                    value_range.feature_name,
                    0,
                )
                + 1
            )
        if len(feature_names_count) != len(value_ranges):
            msg = "Duplicate feature names found: "
            msg += ", ".join(
                f"{feature_name}: {count}"
                for feature_name, count in feature_names_count.items()
                if count > 1
            )
            raise ValueError(msg)

        self.value_ranges = value_ranges

        # Calculate the number of test cases
        self.number_test_cases = functools.reduce(
            lambda n, range_: n * len(range_),
            self.value_ranges,
            1,
        )

        # Build the product iterator
        self.product_iterator = itertools.product(
            *self.value_ranges,
        )

        # Perform the first step
        self.step()

    def num_steps(self) -> int | None:
        return self.number_test_cases

    def step(self) -> None:
        try:
            test_case = next(self.product_iterator)
        except StopIteration:
            raise Finished from None
        self.data = pl.DataFrame(
            dict(test_case),
        )

    def _observe(self) -> Data:
        return self.data.lazy()
