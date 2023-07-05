from typing import Tuple, cast
from typing import List

import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split

from agenc.experiment import Experiment


class Dataset:
    def __init__(
        self,
        data: pl.DataFrame,
        input_columns: List[str],
        output_columns: List[str],
    ):
        self.data = data
        self.input_columns = input_columns
        self.output_columns = output_columns

    @classmethod
    def from_experiment(cls, experiment: Experiment) -> "Dataset":
        columns = [column.name for column in experiment.metadata.columns]
        data = pl.read_csv(experiment.metadata.data_path, columns=columns)
        return cls(
            data=data,
            input_columns=experiment.data.inputs,
            output_columns=experiment.data.outputs,
        )

    def __len__(self) -> int:
        return len(self.data)

    def inputs(self) -> np.ndarray:
        return self.data.select(self.input_columns).to_numpy()

    def outputs(self) -> np.ndarray:
        return self.data.select(self.output_columns).to_numpy()

    def train_test_split(
        self,
        train_size: float,
        random_state: int,
    ) -> Tuple["Dataset", "Dataset"]:
        data_train, data_test = train_test_split(
            self.data,
            train_size=train_size,
            random_state=random_state,
        )

        return Dataset(
            cast(pl.DataFrame, data_train),
            self.input_columns,
            self.output_columns,
        ), Dataset(
            cast(pl.DataFrame, data_test),
            self.input_columns,
            self.output_columns,
        )
