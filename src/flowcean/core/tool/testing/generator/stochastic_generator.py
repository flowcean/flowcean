import random

import polars as pl

from flowcean.core.data import Data
from flowcean.core.environment.incremental import Finished
from flowcean.core.tool.testing.domain import Domain

from .generator import TestcaseGenerator


class StochasticGenerator(TestcaseGenerator):
    """A generator that produces random tests based on given domains."""

    data: pl.DataFrame
    count: int

    def __init__(
        self,
        domains: list[Domain],
        *,
        test_case_count: int | None = None,
        seed: int = 0,
    ) -> None:
        """Initialize the stochastic generator.

        Args:
            domains: A list of domains to generate random values for.
                Each domain must be associated with exactly one input
                feature of the model that shall be tested.
            test_case_count: The number of test cases to generate. If None,
                the generator will run indefinitely.
            seed: The seed for the random number generator. The default is 0,
                which means a random seed will be used.
        """
        super().__init__()

        # Check if no duplicate feature names are present
        feature_names_count: dict[str, int] = {}
        for domain in domains:
            feature_names_count[domain.feature_name] = (
                feature_names_count.get(
                    domain.feature_name,
                    0,
                )
                + 1
            )
        if len(feature_names_count) != len(domains):
            msg = "Duplicate features found: "
            msg += ", ".join(
                f"{feature_name}: {count}"
                for feature_name, count in feature_names_count.items()
                if count > 1
            )
            raise ValueError(msg)

        # Seed all domains
        rng = random.Random(seed) if seed != 0 else random.Random()
        for domain in domains:
            domain.set_seed(rng.randint(0, 2**32 - 1))

        self.domains = domains
        self.count = 0
        self.number_test_cases = test_case_count
        # Perform the first step to initialize the generator
        self.step()

    def num_steps(self) -> int | None:
        return self.number_test_cases

    def reset(self) -> None:
        """Reset the generator to its initial state."""
        self.count = 0
        self.step()

    def step(self) -> None:
        self.count += 1
        if (
            self.number_test_cases is not None
            and self.count > self.number_test_cases
        ):
            raise Finished
        self.data = pl.DataFrame(
            {domain.feature_name: domain() for domain in self.domains},
        )

    def _observe(self) -> Data:
        return self.data.lazy()
