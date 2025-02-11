import polars as pl
import torch
from torch import nn
from torch.optim.sgd import SGD
from typing_extensions import override

from flowcean.core import SupervisedIncrementalLearner

from .model import PyTorchModel


class LinearRegression(SupervisedIncrementalLearner):
    """Linear regression learner."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        learning_rate: float = 1e-3,
        loss: nn.Module | None = None,
    ) -> None:
        """Initialize the learner.

        Args:
            input_size: The size of the input.
            output_size: The size of the output.
            learning_rate: The learning rate.
            loss: The loss function.
        """
        self.model = nn.Linear(input_size, output_size)
        self.loss = loss or nn.MSELoss()
        self.optimizer = SGD(
            self.model.parameters(),
            lr=learning_rate,
        )

    @override
    def learn_incremental(
        self,
        inputs: pl.LazyFrame,
        outputs: pl.LazyFrame,
    ) -> PyTorchModel:
        self.optimizer.zero_grad()
        features = torch.from_numpy(inputs.collect().to_numpy(writable=True))
        labels = torch.from_numpy(outputs.collect().to_numpy(writable=True))
        prediction = self.model(features)
        loss = self.loss(prediction, labels)
        loss.backward()
        self.optimizer.step()
        return PyTorchModel(self.model, outputs.columns)
