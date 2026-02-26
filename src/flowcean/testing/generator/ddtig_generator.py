import polars as pl

from pathlib import Path
from typing import Any, TextIO, Iterator

from flowcean.testing.generator.ddtig.application.test_pipeline import TestPipeline
from .generator import TestcaseGenerator
from flowcean.core.data import Data
from flowcean.core import Model
from flowcean.core.environment.incremental import Finished



class ddtigGenerator(TestcaseGenerator):
    """A generator that produces random tests based on given domains."""

    data: pl.DataFrame
    count: int
    _row_iter: Iterator
    _columns: list[str]
    _df: pl.DataFrame

    def __init__(
        self,
        model: Model,
        *,
        n_testinputs: int,
        test_coverage_criterium: str,
        dataset: pl.DataFrame | None = None,
        specs_file: Path | TextIO | None = None,
        classification: bool = False,
        inverse_alloc: bool = False,
        epsilon: float = 0.5,
        seed: int = 42,

        performance_threshold: float = 0.3,
        sample_limit: int = 50000,
        n_predictions: int = 50,
        max_depth: int = 5,
        hoeffding_tree_extra_params: dict[str, Any] | None = None,
        

    ) -> None:
        """Initialize the stochastic generator.

        Args:
            model: The trained Flowcean model.
            reqs_file: Path to the test requirements file.
            dataset: Optional Polars DataFrame containing the original dataset.
            specs_file: Path to a file containing feature specifications. If you provide a dataset containing system inputs and outputs that already encodes the necessary specifications, then you do not need to supply a separate system specification file.
            classification: Whether the task is a classification problem (default: False).
            inverse_alloc (optional) : If true, use inverse test allocation strategy.
            epsilon (optional) : Size of interval around boundaries for BVA testing.

            For Surrogate model training (only applicable for black-box models):
                performance_threshold (optional) : Minimum performance required to export the Hoeffding Tree (only applicable for black-box models).
                sample_limit (optional) : Maximum number of samples used to train the Hoeffding Tree (only applicable for black-box models).
                n_predictions (optional) : Number of correct predictions required before exporting the Hoeffding Tree (only applicable for black-box models).
                max_depth (optional) : Maximum depth of the Hoeffding tree.
                hoeffding_tree_extra_params (optional) : Additional parameters for training the Hoeffding Tree (only applicable for black-box models).
        """
        super().__init__()
        self.n_testinputs = n_testinputs
    
        self.test_pipeline = TestPipeline(
            model,
            dataset=dataset,
            specs_file=specs_file,
            classification=classification,
            n_testinputs=self.n_testinputs,
            test_coverage_criterium=test_coverage_criterium,
            inverse_alloc=inverse_alloc,
            epsilon=epsilon,
            seed=seed,
            performance_threshold=performance_threshold,
            sample_limit=sample_limit,
            n_predictions=n_predictions,
            max_depth=max_depth,
            hoeffding_tree_extra_params=hoeffding_tree_extra_params,
        )

        self.reset()

    def num_steps(self) -> int | None:
        return self.n_testinputs

    def reset(self) -> None:
        self._df = self.test_pipeline.execute()
        self._columns = self._df.columns
        self.n_testinputs = self._df.height
        self._row_iter = self._df.iter_rows()
        self.step()

    def step(self) -> None:
        try:
            row = next(self._row_iter)
        except StopIteration:
            raise Finished from None
        self.data = pl.DataFrame([row], schema=self._columns, orient="row")

    def _observe(self) -> Data:
        return self.data.lazy()

        

