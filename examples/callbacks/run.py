#!/usr/bin/env python

"""Example demonstrating learner progress callbacks.

Run this example with:
    uv run --directory ./examples/callbacks/ run.py
"""

import logging
import os
import tempfile

import lightning
import polars as pl
import torch
from river.linear_model import LinearRegression
from torch import nn

import flowcean.cli
from flowcean.core.callbacks import (
    LoggingCallback,
    RichCallback,
    SilentCallback,
)
from flowcean.river import RiverLearner
from flowcean.sklearn import RandomForestRegressorLearner
from flowcean.torch import LightningLearner
from flowcean.xgboost import XGBoostRegressorLearner

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")


def create_sample_data(
    n_samples: int = 1000,
) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """Create sample data for demonstration."""
    import numpy as np

    rng = np.random.default_rng(42)

    x1 = rng.standard_normal(n_samples)
    x2 = rng.standard_normal(n_samples)
    x3 = rng.standard_normal(n_samples)
    y = 2 * x1 + 3 * x2 - x3 + rng.standard_normal(n_samples) * 0.1

    inputs = pl.LazyFrame({"x1": x1, "x2": x2, "x3": x3})
    outputs = pl.LazyFrame({"y": y})

    return inputs, outputs


def demo_rich_callback() -> None:
    """Demo 1: RichCallback via explicit opt-in."""
    print("\n" + "=" * 60)
    print("Demo 1: RichCallback via explicit opt-in")
    print("=" * 60)

    inputs, outputs = create_sample_data(n_samples=1000)

    learner = RandomForestRegressorLearner(
        n_estimators=50,
        callbacks=RichCallback(),
    )

    model = learner.learn(inputs, outputs)
    print(f"Model trained! Type: {type(model).__name__}\n")


def demo_logging_callback() -> None:
    """Demo 2: LoggingCallback - Standard Python logging."""
    print("\n" + "=" * 60)
    print("Demo 2: LoggingCallback")
    print("=" * 60)

    inputs, outputs = create_sample_data(n_samples=1000)

    learner = RandomForestRegressorLearner(
        n_estimators=50,
        callbacks=LoggingCallback(),
    )

    model = learner.learn(inputs, outputs)
    print(f"Model trained! Type: {type(model).__name__}\n")


def demo_silent_callback() -> None:
    """Demo 3: SilentCallback - No progress output."""
    print("\n" + "=" * 60)
    print("Demo 3: SilentCallback - No progress output")
    print("=" * 60)

    inputs, outputs = create_sample_data(n_samples=1000)

    learner = RandomForestRegressorLearner(
        n_estimators=50,
        callbacks=SilentCallback(),
    )

    model = learner.learn(inputs, outputs)
    print(f"Model trained! Type: {type(model).__name__}\n")


def demo_silent_default() -> None:
    """Demo 4: callbacks=None stays silent by default."""
    print("\n" + "=" * 60)
    print("Demo 4: Default behavior is silent")
    print("=" * 60)

    inputs, outputs = create_sample_data(n_samples=1000)

    learner = RandomForestRegressorLearner(n_estimators=50)

    model = learner.learn(inputs, outputs)
    print(f"Model trained silently! Type: {type(model).__name__}\n")


def demo_multiple_callbacks() -> None:
    """Demo 5: Multiple callbacks at once."""
    print("\n" + "=" * 60)
    print("Demo 5: RichCallback and LoggingCallback together")
    print("=" * 60)

    inputs, outputs = create_sample_data(n_samples=1000)

    learner = RandomForestRegressorLearner(
        n_estimators=50,
        callbacks=[RichCallback(), LoggingCallback()],
    )

    model = learner.learn(inputs, outputs)
    print(f"Model trained! Type: {type(model).__name__}\n")


def demo_lightning_with_progress() -> None:
    """Demo 6: Lightning learner with live progress bar."""
    print("\n" + "=" * 60)
    print("Demo 6: PyTorch Lightning with Rich progress")
    print("=" * 60)

    class SimpleNet(lightning.LightningModule):
        def __init__(self) -> None:
            super().__init__()
            self.layer1 = nn.Linear(3, 64)
            self.layer2 = nn.Linear(64, 32)
            self.layer3 = nn.Linear(32, 1)
            self.relu = nn.ReLU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.relu(self.layer1(x))
            x = self.relu(self.layer2(x))
            return self.layer3(x)

        def training_step(
            self,
            batch: tuple[torch.Tensor, torch.Tensor],
            _batch_idx: int,
        ) -> torch.Tensor:
            x, y = batch
            y_pred = self(x)
            loss = nn.functional.mse_loss(y_pred, y)
            self.log("train_loss", loss)
            return loss

        def configure_optimizers(self) -> torch.optim.Optimizer:
            return torch.optim.Adam(self.parameters(), lr=0.01)

    inputs, outputs = create_sample_data(n_samples=256)

    learner = LightningLearner(
        module=SimpleNet(),
        batch_size=4,
        max_epochs=5,
        callbacks=RichCallback(),
    )

    current_directory = os.getcwd()
    with tempfile.TemporaryDirectory() as temp_directory:
        os.chdir(temp_directory)
        try:
            model = learner.learn(inputs, outputs)
        finally:
            os.chdir(current_directory)

    print(f"Model trained! Type: {type(model).__name__}")


def demo_xgboost_with_progress() -> None:
    """Demo 7: XGBoost with per-iteration progress updates."""
    print("\n" + "=" * 60)
    print("Demo 7: XGBoost Regressor with Rich progress")
    print("=" * 60)

    inputs, outputs = create_sample_data(n_samples=256)

    learner = XGBoostRegressorLearner(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        callbacks=RichCallback(),
    )

    model = learner.learn(inputs, outputs)
    print(f"Model trained! Type: {type(model).__name__}\n")


def demo_river_incremental() -> None:
    """Demo 8: River with incremental learning progress."""
    print("\n" + "=" * 70)
    print("Demo 8: River with Rich progress updates")
    print("=" * 70)

    inputs, outputs = create_sample_data(n_samples=5000)

    learner = RiverLearner(
        model=LinearRegression(),
        callbacks=RichCallback(),
        progress_interval=50,
    )

    model = learner.learn_incremental(inputs, outputs)
    print(f"Model trained! Type: {type(model).__name__}\n")


def main() -> None:
    """Run all callback demonstrations."""
    flowcean.cli.initialize()

    print("\n")
    print("╔" + "=" * 58 + "╗")
    print(
        "║" + " " * 10 + "Flowcean Learner Callback Examples" + " " * 14 + "║",
    )
    print("╚" + "=" * 58 + "╝")
    print(
        "callbacks=None is silent by default. Rich output requires RichCallback()."
    )

    demo_rich_callback()
    demo_logging_callback()
    demo_silent_callback()
    demo_silent_default()
    demo_multiple_callbacks()
    demo_lightning_with_progress()
    demo_xgboost_with_progress()
    demo_river_incremental()

    print("\n" + "=" * 60)
    print("All demos completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
