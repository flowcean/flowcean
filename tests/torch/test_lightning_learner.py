"""Unit tests for PyTorch Lightning learners and models."""

import unittest
from typing import cast
from unittest.mock import patch

import polars as pl
import torch

from flowcean.torch import LightningLearner, MultilayerPerceptron, PyTorchModel


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

    def test_learn_sets_example_input_and_preserves_default_logger(
        self,
    ) -> None:
        """Tests LightningLearner keeps Lightning fit defaults intact."""
        learner = LightningLearner(module=self.simple_module, batch_size=2)

        with patch("lightning.Trainer") as mock_trainer:
            mock_trainer.return_value.fit.return_value = None

            learner.learn(self.inputs, self.outputs)

        example_input_array = cast(
            "torch.Tensor",
            self.simple_module.example_input_array,
        )
        assert torch.equal(
            example_input_array,
            torch.tensor([1.0, 4.0]),
        )
        trainer_kwargs = mock_trainer.call_args.kwargs
        assert trainer_kwargs["enable_progress_bar"] is False
        assert "logger" not in trainer_kwargs
