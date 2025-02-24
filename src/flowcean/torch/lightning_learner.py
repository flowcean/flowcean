import os
import platform
from typing import TYPE_CHECKING, Any

import lightning
import polars as pl
import torch
from torch import Tensor
from torch.optim.adam import Adam
from torch.utils.data import DataLoader
from typing_extensions import override

from flowcean.core import SupervisedLearner

from .dataset import TorchDataset
from .model import PyTorchModel

if TYPE_CHECKING:
    from torch.nn import Module


class LightningLearner(SupervisedLearner):
    """A learner that uses PyTorch Lightning."""

    def __init__(
        self,
        module: lightning.LightningModule,
        num_workers: int | None = None,
        batch_size: int = 32,
        max_epochs: int = 100,
    ) -> None:
        """Initialize the learner.

        Args:
            module: The PyTorch Lightning module.
            num_workers: The number of workers to use for the DataLoader.
            batch_size: The batch size to use for training.
            max_epochs: The maximum number of epochs to train for.
        """
        self.module = module
        self.num_workers = num_workers or os.cpu_count() or 0
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.optimizer = None

    @override
    def learn(
        self,
        inputs: pl.LazyFrame,
        outputs: pl.LazyFrame,
    ) -> PyTorchModel:
        collected_inputs = inputs.collect()
        collected_outputs = outputs.collect()
        dataset = TorchDataset(collected_inputs, collected_outputs)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=platform.system() == "Windows",
        )
        trainer = lightning.Trainer(
            max_epochs=self.max_epochs,
        )
        trainer.fit(self.module, dataloader)
        return PyTorchModel(self.module, collected_outputs.columns)


class MultilayerPerceptron(lightning.LightningModule):
    """A multilayer perceptron."""

    def __init__(
        self,
        learning_rate: float,
        input_size: int,
        output_size: int,
        hidden_dimensions: list[int] | None = None,
    ) -> None:
        """Initialize the model.

        Args:
            learning_rate: The learning rate.
            input_size: The size of the input.
            output_size: The size of the output.
            hidden_dimensions: The dimensions of the hidden layers.
        """
        super().__init__()
        if hidden_dimensions is None:
            hidden_dimensions = []
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        layers: list[Module] = []
        hidden_size = input_size
        for dimension in hidden_dimensions:
            layers.extend(
                (
                    torch.nn.Linear(hidden_size, dimension),
                    torch.nn.LeakyReLU(),
                ),
            )
            hidden_size = dimension
        layers.append(torch.nn.Linear(hidden_size, output_size))
        self.model = torch.nn.Sequential(*layers)

    @override
    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        y: Tensor = self.model(*args, **kwargs)
        return y

    @override
    def training_step(self, batch: Any) -> Tensor:
        inputs, targets = batch
        outputs = self(inputs)
        return torch.nn.functional.mse_loss(outputs, targets)

    @override
    def configure_optimizers(self) -> Any:
        optimizer = Adam(
            self.parameters(),
            lr=self.learning_rate,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=40,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }


class ConvolutionalNeuralNetwork(lightning.LightningModule):
    """A convolutional neural network."""

    def __init__(
        self,
        learning_rate: float,
        conv_configs: list[tuple[int, int, int]] | None = None,
        fully_connected_layer_sizes: list[int] | None = None,
        output_size: int = 10,
        dropout_rates: list[float] | None = None,
        input_shape: tuple[int, int, int] = (1, 28, 28),
    ) -> None:
        """Initialize the model.

        Args:
            learning_rate: The learning rate.
            conv_configs: The convolutional layer configurations.
            fully_connected_layer_sizes: The fully connected layer sizes.
            output_size: The size of the output.
            dropout_rates: The dropout rates.
            input_shape: The shape of the input.
        """
        super().__init__()
        if conv_configs is None:
            conv_configs = [(1, 32, 3)]
        if fully_connected_layer_sizes is None:
            fully_connected_layer_sizes = [128]
        if dropout_rates is None:
            dropout_rates = [0.25, 0.5]
        self.input_shape = input_shape
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        self.conv_layers = torch.nn.ModuleList()
        for input_channels, output_channels, kernel_size in conv_configs:
            self.conv_layers.append(
                torch.nn.Conv2d(
                    input_channels,
                    output_channels,
                    kernel_size=kernel_size,
                ),
            )

        self.dropout_layers = torch.nn.ModuleList(
            torch.nn.Dropout2d(rate) for rate in dropout_rates
        )

        self.fully_connected_layers = torch.nn.ModuleList()
        prev_size = self._compute_flat_features(conv_configs)
        for size in fully_connected_layer_sizes:
            self.fully_connected_layers.append(
                torch.nn.Linear(prev_size, size),
            )
            prev_size = size
        self.output_layer = torch.nn.Linear(prev_size, output_size)

    def _compute_flat_features(
        self,
        conv_configs: list[tuple[int, int, int]],
    ) -> int:
        channels, height, width = self.input_shape
        for _, output_channels, kernel_size in conv_configs:
            height = height - kernel_size + 1
            width = width - kernel_size + 1
            height //= 2
            width //= 2
            channels = output_channels
        return channels * height * width

    def forward(self, x: Tensor) -> Tensor:
        for i, conv in enumerate(self.conv_layers):
            x = torch.relu(conv(x))
            if i < len(self.dropout_layers):
                x = self.dropout_layers[i](x)
            x = torch.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        for fc in self.fully_connected_layers:
            x = torch.relu(fc(x))
        return self.output_layer(x)

    @override
    def training_step(self, batch: Any) -> Tensor:
        inputs, targets = batch
        outputs = self(inputs)
        return torch.nn.functional.mse_loss(outputs, targets)

    @override
    def configure_optimizers(self) -> Any:
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=40,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class LongShortTermMemoryNetwork(lightning.LightningModule):
    """An LSTM network with configurable layers."""

    def __init__(
        self,
        learning_rate: float,
        input_size: int,
        output_size: int,
        hidden_sizes: list[int] | None = None,
        fully_connected_layer_sizes: list[int] | None = None,
    ) -> None:
        """Initialize the model.

        Args:
            learning_rate: The learning rate.
            input_size: The size of the input.
            output_size: The size of the output.
            hidden_sizes: The sizes of the hidden layers.
            fully_connected_layer_sizes: The sizes of the fully connected
                layers.
        """
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [128]
        if fully_connected_layer_sizes is None:
            fully_connected_layer_sizes = [64]
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        self.lstm_layers = torch.nn.ModuleList()
        prev_size = input_size
        for size in hidden_sizes:
            self.lstm_layers.append(
                torch.nn.LSTM(prev_size, size, batch_first=True),
            )
            prev_size = size

        self.fully_connected_layers = torch.nn.ModuleList()
        for size in fully_connected_layer_sizes:
            self.fully_connected_layers.append(
                torch.nn.Linear(prev_size, size),
            )
            prev_size = size
        self.output_layer = torch.nn.Linear(prev_size, output_size)

    @override
    def forward(self, x: Tensor) -> Tensor:
        for lstm in self.lstm_layers:
            x, _ = lstm(x)
        x = x[:, -1, :]
        for fc in self.fully_connected_layers:
            x = torch.relu(fc(x))
        return self.output_layer(x)

    @override
    def training_step(self, batch: Any) -> Tensor:
        inputs, targets = batch
        outputs = self(inputs)
        return torch.nn.functional.mse_loss(outputs, targets)

    @override
    def configure_optimizers(self) -> Any:
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=40,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class LongTermRecurrentConvolutionalNetwork(lightning.LightningModule):
    """An LRCN.

    Combines configurable convolutional and LSTM layers.
    """

    def __init__(
        self,
        learning_rate: float,
        conv_configs: list[tuple[int, int, int]] | None = None,
        lstm_hidden_sizes: list[int] | None = None,
        fully_connected_layer_sizes: list[int] | None = None,
        output_size: int = 10,
        dropout_rates: list[float] | None = None,
        input_shape: tuple[int, int, int] = (1, 28, 28),
    ) -> None:
        """Initialize the model.

        Args:
            learning_rate: The learning rate.
            conv_configs: The convolutional layer configurations.
            lstm_hidden_sizes: The sizes of the hidden LSTM layers.
            fully_connected_layer_sizes: The sizes of the fully connected
                layers.
            output_size: The size of the output.
            dropout_rates: The dropout rates.
            input_shape: The shape of the input.
        """
        super().__init__()
        if conv_configs is None:
            conv_configs = [(1, 32, 3), (32, 64, 3)]
        if lstm_hidden_sizes is None:
            lstm_hidden_sizes = [128]
        if fully_connected_layer_sizes is None:
            fully_connected_layer_sizes = [64]
        if dropout_rates is None:
            dropout_rates = [0.25, 0.5]
        self.input_shape = input_shape
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        self.conv_layers = torch.nn.ModuleList()
        for in_ch, out_ch, kernel_size in conv_configs:
            self.conv_layers.append(
                torch.nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size),
            )

        self.dropout_layers = torch.nn.ModuleList(
            torch.nn.Dropout2d(rate) for rate in dropout_rates
        )

        self.lstm_layers = torch.nn.ModuleList()
        prev_size = conv_configs[-1][1] if conv_configs else 1
        for size in lstm_hidden_sizes:
            self.lstm_layers.append(
                torch.nn.LSTM(prev_size, size, batch_first=True),
            )
            prev_size = size

        self.fully_connected_layers = torch.nn.ModuleList()
        for size in fully_connected_layer_sizes:
            self.fully_connected_layers.append(
                torch.nn.Linear(prev_size, size),
            )
            prev_size = size
        self.output_layer = torch.nn.Linear(prev_size, output_size)

    def forward(self, x: Tensor) -> Tensor:
        for i, conv in enumerate(self.conv_layers):
            x = torch.relu(conv(x))
            if i < len(self.dropout_layers):
                x = self.dropout_layers[i](x)
            x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1, self.conv_layers[-1].out_channels)
        for lstm in self.lstm_layers:
            x, _ = lstm(x)
        x = x[:, -1, :]
        for fc in self.fully_connected_layers:
            x = torch.relu(fc(x))
        return self.output_layer(x)

    @override
    def training_step(self, batch: Any) -> Tensor:
        inputs, targets = batch
        outputs = self(inputs)
        return torch.nn.functional.mse_loss(outputs, targets)

    @override
    def configure_optimizers(self) -> Any:
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=40,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
