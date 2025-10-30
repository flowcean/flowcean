import logging

import polars as pl
from pysr import PySRRegressor
from typing_extensions import override

from flowcean.core import Model, SupervisedIncrementalLearner

logger = logging.getLogger(__name__)


class PySRModel(Model):
    def __init__(self, model: PySRRegressor) -> None:
        self.model = model

    @override
    def _predict(self, input_features: pl.LazyFrame) -> pl.LazyFrame:
        return pl.LazyFrame(
            self.model.predict(input_features.collect()),
        )


class PySRLearner(SupervisedIncrementalLearner):
    """Wrapper for PySR symbolic regression learner."""

    def __init__(self, model: PySRRegressor) -> None:
        self.model = model
        model.warm_start = True

    def learn_incremental(
        self,
        inputs: pl.LazyFrame,
        outputs: pl.LazyFrame,
    ) -> Model:
        dfs = pl.collect_all([inputs, outputs])
        collected_inputs = dfs[0]
        collected_outputs = dfs[1]
        self.model.fit(collected_inputs, collected_outputs)

        # Return the trained PySRModel
        return PySRModel(self.model)


logger = logging.getLogger(__name__)
