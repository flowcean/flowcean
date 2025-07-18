from __future__ import annotations

import logging
import os
import platform
from collections.abc import Callable
from io import BytesIO
from typing import Any

import lightning
import polars as pl
import torch
from feature_images import FeatureImagesData, FeatureImagesPredictionData
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader, Dataset
from typing_extensions import override

from flowcean.core import Model, SupervisedLearner

DatasetFactory = Callable[[pl.DataFrame], Dataset]

logger = logging.getLogger(__name__)


class ImageBasedLightningLearner(SupervisedLearner):
    """A learner that uses PyTorch Lightning for image-based datasets."""

    def __init__(
        self,
        module: lightning.LightningModule,
        num_workers: int | None = None,
        batch_size: int = 32,
        max_epochs: int = 100,
        accelerator: str = "auto",
        dataset_factory: DatasetFactory = lambda data: FeatureImagesData(
            data,
            image_size=32,
            width_meters=15.0,
        ),
        dataset_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the learner.

        Args:
            module: The PyTorch Lightning module.
            num_workers: The number of workers to use for the DataLoader.
            batch_size: The batch size to use for training.
            max_epochs: The maximum number of epochs to train for.
            accelerator: The accelerator to use.
            dataset_factory: Takes a DataFrame and returns a FeatureImagesData Dataset.
            dataset_kwargs: Keyword arguments to pass to the dataset factory (optional).
        """
        if dataset_kwargs is None:
            dataset_kwargs = {}
        self.module = module
        self.num_workers = num_workers or os.cpu_count() or 0
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.accelerator = accelerator
        self.dataset_factory = dataset_factory
        self.dataset_kwargs = dataset_kwargs

    @override
    def learn(
        self,
        inputs: pl.DataFrame,
        outputs: pl.DataFrame,
    ) -> ImageBasedPyTorchModel:
        """Learn from the input and output data.

        Args:
            inputs: Input features as a DataFrame with structured columns.
            outputs: Output labels as a DataFrame.

        Returns:
            An ImageBasedPyTorchModel instance.
        """
        data = inputs.with_columns(
            outputs.to_series(0).alias("is_delocalized"),
        )
        dataset = self.dataset_factory(data)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=platform.system() == "Linux",
        )
        trainer = lightning.Trainer(
            accelerator=self.accelerator,
            max_epochs=self.max_epochs,
            callbacks=[
                EarlyStopping(
                    monitor="train_loss",
                    patience=10,
                    mode="min",
                ),
            ],
        )
        trainer.fit(self.module, dataloader)
        return ImageBasedPyTorchModel(
            self.module,
            image_size=self.dataset_kwargs.get("image_size", 32),
            width_meters=self.dataset_kwargs.get("width_meters", 15.0),
            output_names=outputs.columns,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )


class ImageBasedPyTorchModel(Model):
    """PyTorch model wrapper for image-based datasets."""

    def __init__(
        self,
        module: lightning.LightningModule,
        image_size: int,
        width_meters: float,
        output_names: list[str],
        batch_size: int = 32,
        num_workers: int = 1,
    ) -> None:
        """Initialize the model.

        Args:
            module: The PyTorch Lightning module.
            image_size: The size of the image (height and width).
            width_meters: The width of the image in meters.
            output_names: The names of the output columns.
            batch_size: The batch size to use for predictions.
            num_workers: The number of workers to use for the DataLoader.
        """
        self.module = module
        self.image_size = image_size
        self.width_meters = width_meters
        self.output_names = output_names
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.BINARY_CLASSIFICATION_THRESHOLD = 0.5

    @override
    def predict(self, input_features: pl.LazyFrame) -> pl.LazyFrame:
        """Predict outputs for the given input features.

        Args:
            input_features: The input features as a LazyFrame with structured columns.

        Returns:
            A LazyFrame containing the predicted outputs.
        """
        # Validate required columns
        required_columns = [
            "/map",
            "/scan",
            "/particle_cloud",
            "/amcl_pose",
        ]
        input_columns = input_features.columns
        missing_columns = [
            col for col in required_columns if col not in input_columns
        ]
        if missing_columns:
            logger.error(
                f"Missing required columns in input_features: {missing_columns}",
            )
            raise ValueError(f"Missing required columns: {missing_columns}")

        dataset = FeatureImagesPredictionData(
            input_features.collect(),
            image_size=self.image_size,
            width_meters=self.width_meters,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=platform.system() == "Linux",
        )
        predictions = []
        self.module.eval()
        with torch.no_grad():
            for batch in dataloader:
                outputs = self.module(batch)
                preds = (
                    outputs > self.BINARY_CLASSIFICATION_THRESHOLD
                ).float()
                predictions.append(preds)
        predictions = torch.cat(predictions, dim=0).numpy()
        return pl.DataFrame(predictions, schema=self.output_names).lazy()

    @override
    def save_state(self) -> dict[str, Any]:
        """Save the model state to a dictionary.

        Returns:
            A dictionary containing the model state.
        """
        model_bytes = BytesIO()
        torch.save(self.module.state_dict(), model_bytes)
        model_bytes.seek(0)
        return {
            "data": model_bytes.read(),
            "output_names": self.output_names,
            "image_size": self.image_size,
            "width_meters": self.width_meters,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
        }

    @override
    @classmethod
    def load_from_state(cls, state: dict[str, Any]) -> ImageBasedPyTorchModel:
        """Load the model from a state dictionary.

        Args:
            state: A dictionary containing the model state.

        Returns:
            An Instance of ImageBasedPyTorchModel.
        """
        from architectures.cnn import (
            CNN,  # Lazy import to avoid circular dependencies
        )

        module = CNN(
            image_size=state["image_size"],
            in_channels=3,
            learning_rate=0.0001,  # Default value, adjust if needed
        )
        module.load_state_dict(
            torch.load(BytesIO(state["data"]), weights_only=True),
        )
        return cls(
            module,
            image_size=state["image_size"],
            width_meters=state["width_meters"],
            output_names=state["output_names"],
            batch_size=state["batch_size"],
            num_workers=state["num_workers"],
        )
