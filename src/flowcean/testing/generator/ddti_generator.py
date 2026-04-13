from pathlib import Path
from typing import Any

import polars as pl

from flowcean.core import Model
from flowcean.core.data import Data
from flowcean.polars.environments.dataframe import DataFrame
from flowcean.testing.generator.ddtig.application.test_pipeline import (
    TestPipeline,
)
from flowcean.utils import get_seed

from .generator import TestcaseGenerator


class DDTIGenerator(TestcaseGenerator):
    """Generates test inputs considering decision boundaries.

    Methods:
    -------
    save_hoeffding_tree()
        Saves the generated Hoeffding tree to a file.

    print_eqclasses()
        Prints the equivalence classes and their test input counts.

    print_testplans()
        Prints the test plans (intervals used to sample test inputs).

    print_hoeffding_tree()
        Prints the Hoeffding tree structure as a PNG.

    """

    def __init__(
        self,
        model: Model,
        *,
        n_testinputs: int,
        test_coverage_criterium: str,
        dataset: pl.DataFrame | None = None,
        specs_file: Path | None = None,
        classification: bool = False,
        inverse_alloc: bool = False,
        epsilon: float = 0.5,
        performance_threshold: float = 0.3,
        sample_limit: int = 50000,
        n_predictions: int = 50,
        max_depth: int = 5,
        hoeffding_tree_extra_params: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the stochastic generator.

        Args:
            model: The trained Flowcean model.
            n_testinputs: Number of test inputs to generate.
            test_coverage_criterium: Test coverage strategy identifier.
            dataset: Polars DataFrame containing the original dataset.
                Either the dataset or the specs_file must be provided.
            specs_file: Path to a file containing feature specifications.
                If you provide a dataset containing system inputs and
                outputs that already encodes the necessary specifications,
                then you do not need to supply a separate system
                specification file.
            classification: Whether the task is a classification problem.
            inverse_alloc: If True, allocate more tests to lower-priority
                equivalence classes.
            epsilon: Interval offset used for boundary value analysis.
            performance_threshold: Minimum performance needed before
                exporting the Hoeffding Tree.
            sample_limit: Maximum number of samples used to train the
                Hoeffding Tree.
            n_predictions: Number of consecutive correct predictions needed
                before exporting the Hoeffding Tree.
            max_depth: Maximum depth of the Hoeffding Tree.
            hoeffding_tree_extra_params: Extra keyword arguments forwarded
                to the Hoeffding Tree trainer.
        """
        super().__init__()
        self.n_testinputs = n_testinputs
        self.seed = get_seed()

        self.test_pipeline = TestPipeline(
            model,
            dataset=dataset,
            specs_file=specs_file,
            classification=classification,
            n_testinputs=self.n_testinputs,
            test_coverage_criterium=test_coverage_criterium,
            inverse_alloc=inverse_alloc,
            epsilon=epsilon,
            seed=self.seed,
            performance_threshold=performance_threshold,
            sample_limit=sample_limit,
            n_predictions=n_predictions,
            max_depth=max_depth,
            hoeffding_tree_extra_params=hoeffding_tree_extra_params,
        )
        self.data = DataFrame(self.test_pipeline.execute())
        self.reset()

    def num_steps(self) -> int | None:
        return self.n_testinputs

    def reset(self) -> None:
        self._streaming_env = self.data.to_incremental()

    def step(self) -> None:
        self._streaming_env.step()

    def print_eqclasses(self) -> None:
        self.test_pipeline.print_eqclasses()

    def print_testplans(self) -> None:
        self.test_pipeline.print_testplans()

    def print_hoeffding_tree(self) -> None:
        self.test_pipeline.print_hoeffding_tree()

    def save_hoeffding_tree(self, path: str | Path) -> None:
        self.test_pipeline.save_hoeffding_tree(path)

    def _observe(self) -> Data:
        return self._streaming_env.observe()
