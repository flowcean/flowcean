from __future__ import annotations

import importlib
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from io import BytesIO
from typing import Any, BinaryIO, cast

import polars as pl
from typing_extensions import override

from flowcean.core.transform import Transform


class Model(ABC):
    """Base class for models.

    A model is used to predict outputs for given inputs.
    """

    @abstractmethod
    def predict(
        self,
        input_features: pl.LazyFrame,
    ) -> pl.LazyFrame:
        """Predict outputs for the given inputs.

        Args:
            input_features: The inputs for which to predict the outputs.

        Returns:
            The predicted outputs.
        """

    @abstractmethod
    def save(self, file: BinaryIO) -> None:
        """Save the model to the file.

        Args:
            file: The file like object to save the model to.
        """

    @classmethod
    @abstractmethod
    def load(cls, file: BinaryIO) -> Model:
        """Load the model from file.

        Args:
            file: The file like object to load the model from.
        """


@dataclass
class ModelWithTransform(Model):
    """Model that carries a transform.

    Attributes:
        model: Model
        transform: Transform
    """

    model: Model
    input_transform: Transform
    output_transform: Transform

    @override
    def predict(
        self,
        input_features: pl.LazyFrame,
    ) -> pl.LazyFrame:
        transformed = self.input_transform.apply(input_features)
        return self.output_transform.apply(self.model.predict(transformed))

    @override
    def save(self, file: BinaryIO) -> None:
        model_bytes = BytesIO()
        self.model.save(model_bytes)
        model_bytes.seek(0)

        input_bytes = BytesIO()
        pickle.dump(self.input_transform, input_bytes)
        input_bytes.seek(0)

        output_bytes = BytesIO()
        pickle.dump(self.output_transform, output_bytes)
        output_bytes.seek(0)

        data = {
            "model": model_bytes.read(),
            "model_type": fullname(self.model),
            "input_transform": input_bytes.read(),
            "output_transform": output_bytes.read(),
        }
        pickle.dump(
            data,
            file,
        )

    @override
    @classmethod
    def load(cls, file: BinaryIO) -> ModelWithTransform:
        data = pickle.load(file)  # noqa: S301

        # Create a model based on the type
        model_type = data["model_type"]
        module_name, class_name = model_type.rsplit(".", 1)
        module = importlib.import_module(module_name)
        model_class = getattr(module, class_name)

        model = cast(Model, model_class.load(BytesIO(data["model"])))
        input_transform = pickle.load(BytesIO(data["input_transform"]))  # noqa: S301
        output_transform = pickle.load(BytesIO(data["output_transform"]))  # noqa: S301

        return cls(model, input_transform, output_transform)


def fullname(o: Any) -> str:
    klass = o.__class__
    module = klass.__module__
    if module == "builtins":
        return klass.__qualname__  # avoid outputs like 'builtins.str'
    return module + "." + klass.__qualname__
