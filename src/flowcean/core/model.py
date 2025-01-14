from __future__ import annotations

import importlib
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, cast

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
    # TODO: Use typing.BinaryIO instead of path...?!
    def save(self, path: Path) -> None:
        """Save the model to path.

        Args:
            path: The path to save the model to.
        """

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> Model:
        """Load the model from path.

        Args:
            path: The path to load the model from.
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
    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.model.save(path.joinpath("model"))
        # TODO: This is not nice at all, switch it!
        with path.joinpath("model_type").open("w") as f:
            f.write(fullname(self.model))
        with path.joinpath("input_transform").open("wb") as f:
            pickle.dump(
                self.input_transform,
                f,
            )
        with path.joinpath("output_transform").open("wb") as f:
            pickle.dump(
                self.output_transform,
                f,
            )

    @override
    @classmethod
    def load(cls, path: Path) -> ModelWithTransform:
        # Create a model based on the type
        model_type = path.joinpath("model_type").read_text()
        module_name, class_name = model_type.rsplit(".", 1)
        module = importlib.import_module(module_name)
        model_class = getattr(module, class_name)

        model = cast(Model, model_class.load(path.joinpath("model")))
        with path.joinpath("input_transform").open("rb") as f:
            input_transform = pickle.load(f)  # noqa: S301
        with path.joinpath("output_transform").open("rb") as f:
            output_transform = pickle.load(f)  # noqa: S301

        return cls(model, input_transform, output_transform)


def fullname(o: Any) -> str:
    klass = o.__class__
    module = klass.__module__
    if module == "builtins":
        return klass.__qualname__  # avoid outputs like 'builtins.str'
    return module + "." + klass.__qualname__
