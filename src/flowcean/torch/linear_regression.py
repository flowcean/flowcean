from typing import override

import polars as pl

from flowcean._optional import raise_for_missing_optional_dependency

try:
    import torch
    from torch import Tensor, nn
    from torch.optim.sgd import SGD
except ModuleNotFoundError as error:
    raise_for_missing_optional_dependency(
        error,
        extra="torch",
        module="flowcean.torch.linear_regression",
        missing_dependencies={"torch"},
    )

from flowcean.core import SupervisedIncrementalLearner

from .model import PyTorchModel


class LinearRegression(SupervisedIncrementalLearner):
    """Linear regression learner."""

    def __init__(
        self,
        *,
        output_size: int,
        learning_rate: float = 1e-3,
        loss: nn.Module | None = None,
        a: Tensor | None = None,
        b: Tensor | None = None,
    ) -> None:
        """Initialize the learner.

        Args:
            output_size: The size of the output.
            learning_rate: The learning rate.
            loss: The loss function.
            a: Initial weights. If None (the default), random weights are used.
            b: Initial bias. If None (the default), random bias is used.
        """
        self.model = nn.LazyLinear(output_size)
        if a is not None:
            self.model.weight.data = a
        if b is not None:
            self.model.bias.data = b
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
        dfs = pl.collect_all([inputs, outputs])
        features = torch.from_numpy(  # pyright: ignore[reportPrivateImportUsage]
            dfs[0].to_numpy(writable=True),
        )
        labels = torch.from_numpy(  # pyright: ignore[reportPrivateImportUsage]
            dfs[1].to_numpy(writable=True),
        )
        prediction = self.model(features)
        loss = self.loss(prediction, labels)
        loss.backward()
        self.optimizer.step()
        return PyTorchModel(self.model, outputs.collect_schema().names())
