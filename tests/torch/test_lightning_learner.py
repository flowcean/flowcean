"""Unit tests for PyTorch Lightning learners and models."""

import unittest
from unittest.mock import patch

import polars as pl
import torch

from flowcean.torch import (
    ConvolutionalNeuralNetwork,
    LightningLearner,
    LongShortTermMemoryNetwork,
    LongTermRecurrentConvolutionalNetwork,
    MultilayerPerceptron,
    PyTorchModel,
)


def _create_sample_data() -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """Creates sample input and output data for testing.

    Returns:
        Tuple containing input and output LazyFrames.
    """
    inputs = pl.LazyFrame(
        {
            "x1": [1.0, 2.0, 3.0],
            "x2": [4.0, 5.0, 6.0],
        },
    )
    outputs = pl.LazyFrame(
        {
            "y": [0.0, 1.0, 2.0],
        },
    )
    return inputs, outputs


def _create_simple_module() -> MultilayerPerceptron:
    """Creates a simple MultilayerPerceptron for testing.

    Returns:
        A configured MultilayerPerceptron instance.
    """
    return MultilayerPerceptron(
        learning_rate=0.001,
        input_size=2,
        output_size=1,
    )


class TestLightningLearner(unittest.TestCase):
    """Tests for the LightningLearner class."""

    def setUp(self) -> None:
        """Sets up test fixtures."""
        self.simple_module = _create_simple_module()
        self.inputs, self.outputs = _create_sample_data()

    def test_init(self) -> None:
        """Tests initialization of LightningLearner."""
        learner = LightningLearner(
            module=self.simple_module,
            num_workers=2,
            batch_size=32,
            max_epochs=10,
        )
        assert learner.module == self.simple_module
        assert learner.num_workers == 2
        assert learner.batch_size == 32
        assert learner.max_epochs == 10

    def test_default_num_workers(self) -> None:
        """Tests default num_workers assignment."""
        learner = LightningLearner(module=self.simple_module)
        assert isinstance(learner.num_workers, int)
        assert learner.num_workers >= 0

    def test_learn(self) -> None:
        """Tests the learn method."""
        learner = LightningLearner(module=self.simple_module, batch_size=2)

        with patch("lightning.Trainer") as mock_trainer:
            mock_trainer.return_value.fit.return_value = None
            result = learner.learn(self.inputs, self.outputs)

        assert isinstance(result, PyTorchModel)
        assert result.output_names == [
            "y",
        ]
        assert result.module == self.simple_module


class TestMultilayerPerceptron(unittest.TestCase):
    """Tests for the MultilayerPerceptron class."""

    def setUp(self) -> None:
        """Sets up test fixtures."""
        self.mlp = MultilayerPerceptron(
            learning_rate=0.001,
            input_size=2,
            output_size=1,
            hidden_dimensions=[4, 3],
        )

    def test_init(self) -> None:
        """Tests initialization of MultilayerPerceptron."""
        assert self.mlp.learning_rate == 0.001
        assert len(self.mlp.model) == 5  # 2 hidden layers + output layer
        assert isinstance(self.mlp.model[0], torch.nn.Linear)
        assert isinstance(self.mlp.model[-1], torch.nn.Linear)

    def test_forward(self) -> None:
        """Tests the forward pass."""
        x = torch.randn(1, 2)
        output = self.mlp(x)
        assert output.shape == torch.Size([1, 1])

    def test_training_step(self) -> None:
        """Tests the training step."""
        batch = (torch.randn(2, 2), torch.randn(2, 1))
        loss = self.mlp.training_step(batch)
        assert isinstance(loss, torch.Tensor)
        assert loss.shape == torch.Size([])


class TestConvolutionalNeuralNetwork(unittest.TestCase):
    """Tests for the ConvolutionalNeuralNetwork class."""

    def setUp(self) -> None:
        """Sets up test fixtures."""
        self.cnn = ConvolutionalNeuralNetwork(
            learning_rate=0.001,
            conv_configs=[(1, 32, 3)],
            fully_connected_layer_sizes=[128],
            output_size=10,
            input_shape=(1, 28, 28),
        )

    def test_init(self) -> None:
        """Tests initialization of ConvolutionalNeuralNetwork."""
        assert len(self.cnn.conv_layers) == 1
        assert len(self.cnn.fully_connected_layers) == 1
        assert self.cnn.output_layer.out_features == 10

    def test_forward(self) -> None:
        """Tests the forward pass."""
        x = torch.randn(1, 1, 28, 28)
        output = self.cnn(x)
        assert output.shape == torch.Size([1, 10])

    def test_compute_flat_features(self) -> None:
        """Tests the flat features computation."""
        conv_configs = [(1, 32, 3)]
        flat_size = self.cnn._compute_flat_features(conv_configs)  # noqa: SLF001
        assert isinstance(flat_size, int)
        assert flat_size > 0


class TestLongShortTermMemoryNetwork(unittest.TestCase):
    """Tests for the LongShortTermMemoryNetwork class."""

    def setUp(self) -> None:
        """Sets up test fixtures."""
        self.lstm = LongShortTermMemoryNetwork(
            learning_rate=0.001,
            input_size=2,
            output_size=1,
            hidden_sizes=[128],
            fully_connected_layer_sizes=[64],
        )

    def test_init(self) -> None:
        """Tests initialization of LongShortTermMemoryNetwork."""
        assert len(self.lstm.lstm_layers) == 1
        assert len(self.lstm.fully_connected_layers) == 1
        assert self.lstm.output_layer.out_features == 1

    def test_forward(self) -> None:
        """Tests the forward pass."""
        x = torch.randn(1, 10, 2)  # batch, seq_len, input_size
        output = self.lstm(x)
        assert output.shape == torch.Size([1, 1])


class TestLongTermRecurrentConvolutionalNetwork(unittest.TestCase):
    """Tests for the LongTermRecurrentConvolutionalNetwork class."""

    def setUp(self) -> None:
        """Sets up test fixtures."""
        self.lrcn = LongTermRecurrentConvolutionalNetwork(
            learning_rate=0.001,
            conv_configs=[(1, 32, 3)],
            lstm_hidden_sizes=[128],
            fully_connected_layer_sizes=[64],
            output_size=10,
            input_shape=(1, 28, 28),
        )

    def test_init(self) -> None:
        """Tests initialization of LongTermRecurrentConvolutionalNetwork."""
        assert len(self.lrcn.conv_layers) == 1
        assert len(self.lrcn.lstm_layers) == 1
        assert len(self.lrcn.fully_connected_layers) == 1
        assert self.lrcn.output_layer.out_features == 10

    def test_forward(self) -> None:
        """Tests the forward pass."""
        x = torch.randn(1, 1, 28, 28)
        output = self.lrcn(x)
        assert output.shape == torch.Size([1, 10])


class TestConfigureOptimizers(unittest.TestCase):
    """Tests for the configure_optimizers method across all models."""

    def test_configure_optimizers(self) -> None:
        """Tests optimizer configuration for all models."""
        models = [
            MultilayerPerceptron(0.001, 2, 1),
            ConvolutionalNeuralNetwork(0.001),
            LongShortTermMemoryNetwork(0.001, 2, 1),
            LongTermRecurrentConvolutionalNetwork(0.001),
        ]

        for model in models:
            optimizers = model.configure_optimizers()
            assert "optimizer" in optimizers
            assert "lr_scheduler" in optimizers
            assert isinstance(optimizers["optimizer"], torch.optim.Adam)
            assert isinstance(
                optimizers["lr_scheduler"],
                torch.optim.lr_scheduler.CosineAnnealingLR,
            )


if __name__ == "__main__":
    unittest.main()
