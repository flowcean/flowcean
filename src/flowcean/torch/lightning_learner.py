import os
import platform
from typing import Any

import lightning
import polars as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader
from typing_extensions import override

from flowcean.core import (
    LearnerCallback,
    SupervisedLearner,
    create_callback_manager,
)
from flowcean.core.named import Named

from .dataset import TorchDataset
from .model import PyTorchModel


class LightningCallbackBridge(lightning.Callback):
    """Bridge between PyTorch Lightning callbacks and flowcean callbacks.

    This adapter forwards Lightning training events to flowcean callbacks.
    """

    def __init__(
        self,
        learner: Named,
        callback_manager: Any,
    ) -> None:
        super().__init__()
        self.learner = learner
        self.callback_manager = callback_manager

    def on_train_start(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,  # noqa: ARG002
    ) -> None:
        """Called when training starts."""
        context = {
            "max_epochs": trainer.max_epochs,
            "batch_size": (
                trainer.train_dataloader.batch_size  # type: ignore[union-attr]
                if trainer.train_dataloader
                else None
            ),
        }
        self.callback_manager.on_learning_start(self.learner, context)

    def on_train_batch_end(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
        outputs: Any,  # noqa: ARG002
        batch: Any,  # noqa: ARG002
        batch_idx: int,
    ) -> None:
        """Called after each training batch."""
        # Calculate progress based on current epoch and batch
        if trainer.max_epochs and trainer.num_training_batches:
            current_epoch = trainer.current_epoch
            progress = (
                current_epoch + (batch_idx + 1) / trainer.num_training_batches
            ) / trainer.max_epochs
        else:
            progress = None

        # Extract metrics from logged values
        metrics = {
            "epoch": trainer.current_epoch + 1,
            "batch": batch_idx + 1,
        }

        # Add loss if available
        has_callback_metrics = (
            hasattr(pl_module, "trainer")
            and pl_module.trainer.callback_metrics
        )
        if has_callback_metrics:
            for key, value in pl_module.trainer.callback_metrics.items():
                if hasattr(value, "item"):
                    metrics[key] = value.item()  # type: ignore[assignment]

        self.callback_manager.on_learning_progress(
            self.learner,
            progress=progress,
            metrics=metrics,
        )

    def on_train_end(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
    ) -> None:
        """Called when training ends."""
        # We'll call on_learning_end from the learn method
        # to have access to the wrapped model


class LightningLearner(SupervisedLearner):
    """A learner that uses PyTorch Lightning.

    Args:
        module: The PyTorch Lightning module.
        num_workers: The number of workers to use for the DataLoader.
        batch_size: The batch size to use for training.
        max_epochs: The maximum number of epochs to train for.
        accelerator: The accelerator to use.
        callbacks: Optional callbacks for progress feedback. Defaults to
            RichCallback if not specified.
    """

    def __init__(
        self,
        module: lightning.LightningModule,
        num_workers: int | None = None,
        batch_size: int = 32,
        max_epochs: int = 100,
        accelerator: str = "auto",
        callbacks: list[LearnerCallback] | LearnerCallback | None = None,
    ) -> None:
        """Initialize the learner.

        Args:
            module: The PyTorch Lightning module.
            num_workers: The number of workers to use for the DataLoader.
            batch_size: The batch size to use for training.
            max_epochs: The maximum number of epochs to train for.
            accelerator: The accelerator to use.
            callbacks: Optional callbacks for progress feedback. Defaults to
                RichCallback if not specified.
        """
        self.module = module
        self.num_workers = num_workers or os.cpu_count() or 0
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.optimizer = None
        self.accelerator = accelerator
        self.callback_manager = create_callback_manager(callbacks)

    @override
    def learn(
        self,
        inputs: pl.LazyFrame,
        outputs: pl.LazyFrame,
    ) -> PyTorchModel:
        dfs = pl.collect_all([inputs, outputs])
        collected_inputs = dfs[0]
        collected_outputs = dfs[1]
        dataset = TorchDataset(collected_inputs, collected_outputs)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=platform.system() == "Windows",
        )

        # Create the bridge callback and add it to Lightning callbacks
        bridge_callback = LightningCallbackBridge(
            learner=self,
            callback_manager=self.callback_manager,
        )

        lightning_callbacks: list[lightning.Callback] = [
            bridge_callback,
            EarlyStopping(
                monitor="train_loss",
                patience=10,
                mode="min",
            ),
        ]

        try:
            trainer = lightning.Trainer(
                accelerator=self.accelerator,
                max_epochs=self.max_epochs,
                callbacks=lightning_callbacks,
                # Disable Lightning's progress bar to avoid conflicts
                enable_progress_bar=False,
                # Disable default logging to avoid clutter
                logger=False,
            )
            trainer.fit(self.module, dataloader)

            # Create the model
            model = PyTorchModel(self.module, collected_outputs.columns)

            # Notify callbacks that learning is complete
            final_metrics = {}
            if hasattr(trainer, "callback_metrics"):
                for key, value in trainer.callback_metrics.items():
                    if hasattr(value, "item"):
                        final_metrics[key] = value.item()

            self.callback_manager.on_learning_end(
                self,
                model,
                metrics=final_metrics,
            )
        except Exception as e:
            # Notify callbacks of the error
            self.callback_manager.on_learning_error(self, e)
            raise
        else:
            return model
