import polars as pl

from pathlib import Path
from typing import TextIO, Iterator

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
        reqs_file: Path | TextIO,
        dataset: pl.DataFrame | None = None,
        specs_file: Path | TextIO | None = None,
        classification: bool = False,
        n_testinputs: int  = 1000,
        test_coverage_criterium: str | None = None,
        

    ) -> None:
        """Initialize the stochastic generator.

        Args:
            model: The trained Flowcean model.
            reqs_file: Path to the test requirements file.
            dataset: Optional Polars DataFrame containing the original dataset.
            specs_file: Path to a file containing feature specifications. If you provide a dataset containing system inputs and outputs that already encodes the necessary specifications, then you do not need to supply a separate system specification file.
            classification: Whether the task is a classification problem (default: False).
        """
        super().__init__()
        self.n_testinputs = n_testinputs

        self.test_pipeline = TestPipeline(
            model,
            reqs_file,
            dataset=dataset,
            specs_file=specs_file,
            classification=classification,
            n_testinputs=self.n_testinputs,
            test_coverage_criterium=test_coverage_criterium,
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

        

