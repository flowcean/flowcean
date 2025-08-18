from __future__ import annotations

from tqdm import tqdm
import logging
import os
import platform
from collections.abc import Callable
from io import BytesIO
from typing import Any

import lightning
import polars as pl
import torch
from feature_images import FeatureImagesData
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
        image_size: int = 32,
        width_meters: float = 15.0,
    ) -> None:
        """Initialize the learner.

        Args:
            module: The PyTorch Lightning module.
            num_workers: The number of workers to use for the DataLoader.
            batch_size: The batch size to use for training.
            max_epochs: The maximum number of epochs to train for.
            accelerator: The accelerator to use.
            image_size: The size of the image (height and width).
            width_meters: The width of the image in meters.
        """
        self.module = module
        self.num_workers = num_workers or os.cpu_count() or 0
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.accelerator = accelerator
        self.image_size = image_size
        self.width_meters = width_meters

    @override
    def learn(
        self,
        inputs: pl.DataFrame,
        outputs: pl.DataFrame,
    ) -> ImageBasedPyTorchModel:
        dataset = FeatureImagesData(
            inputs,
            outputs,
            image_size=self.image_size,
            width_meters=self.width_meters,
        )
        # Preload the dataset to ensure all images are loaded into memory
        for i in tqdm(range(len(dataset)), desc="Preloading dataset"):
            dataset[i]

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
            devices=1,
        )
        trainer.fit(self.module, dataloader)
        return ImageBasedPyTorchModel(
            self.module,
            image_size=self.image_size,
            width_meters=self.width_meters,
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
        binary_classification_threshold: float = 0.5,
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
        self.binary_classification_threshold = binary_classification_threshold

    @override
    def predict(self, input_features: pl.LazyFrame) -> pl.LazyFrame:
        dataset = FeatureImagesData(
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
                    outputs > self.binary_classification_threshold
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
        # Lazy import to avoid circular dependencies
        from architectures.cnn import CNN

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
