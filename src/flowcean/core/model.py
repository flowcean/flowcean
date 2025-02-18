from __future__ import annotations

import importlib
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO, cast, final

from typing_extensions import override

if TYPE_CHECKING:
    from .data import Data
    from .transform import Transform


class Model(ABC):
    """Base class for models.

    A model is used to predict outputs for given inputs.
    """

    @abstractmethod
    def predict(self, input_features: Data) -> Data:
        """Predict outputs for the given inputs.

        Args:
            input_features: The inputs for which to predict the outputs.

        Returns:
            The predicted outputs.
        """

    @final
    def save(self, file: Path | BinaryIO) -> None:
        """Save the model to the file.

        This method can be used to save a flowcean model to a file or a
        file-like object. To save a model to a file use

        ```python
        with open("model.fml", "wb") as f:
            model.save(f)
        ```

        The resulting file will contain the model any any attached transforms.
        It can be loaded again using the `load` method from the `Model` class.

        Args:
            file: The file like object to save the model to.
        """
        data = {
            "model": self.save_state(),
            "model_type": fullname(self),
        }
        if isinstance(file, Path):
            with file.open("wb") as f:
                pickle.dump(data, f)
        else:
            pickle.dump(data, file)

    @staticmethod
    def load(file: Path | BinaryIO) -> Model:
        """Load a model from file.

        This method can be used to load a previously saved flowcean model from
        a file or a file-like object.
        To load a model from a file use

        ```python
        with open("model.fml", "rb") as f:
            model = Model.load(f)
        ```

        The `load` method will automatically determine the model type and and
        any attached transforms and will load them into the correct model
        class.

        As this method uses the `pickle` module to load the model, it is not
        safe to load models from untrusted sources as this could lead to
        arbitrary code execution!

        Args:
            file: The file like object to load the model from.
        """
        # Read the general model from the file
        if isinstance(file, Path):
            with file.open("rb") as f:
                data = pickle.load(f)  # noqa: S301
        else:
            data = pickle.load(file)  # noqa: S301
        if not isinstance(data, dict):
            msg = "Invalid model file"
            raise ValueError(msg)  # noqa: TRY004, it's really a value error and not a type error
        data = cast(dict[str, Any], data)

        # Create a model based on the type
        model_type = data["model_type"]
        module_name, class_name = model_type.rsplit(".", 1)
        module = importlib.import_module(module_name)
        model_class = getattr(module, class_name)

        return model_class.load_from_state(data["model"])

    @abstractmethod
    def save_state(self) -> dict[str, Any]:
        """Save the model state to a dictionary.

        To save the model to a file, use the `save` method.
        To create a model from a state dictionary, use the `load_from_state`
        method.

        Returns:
            A dictionary containing the model state.
        """

    @classmethod
    @abstractmethod
    def load_from_state(cls, state: dict[str, Any]) -> Model:
        """Load the model from a state dictionary.

        To load a model from a file use the `load` method.
        To save the model state to a dictionary, use the `save_state` method.

        Args:
            state: A dictionary containing the model state.
        """


@dataclass
class ModelWithTransform(Model):
    """Model that carries a transform.

    Attributes:
        model: Model
        transform: Transform
    """

    model: Model
    input_transform: Transform | None
    output_transform: Transform | None

    @override
    def predict(self, input_features: Data) -> Data:
        if self.input_transform is not None:
            input_features = self.input_transform.apply(input_features)

        prediction = self.model.predict(input_features)

        if self.output_transform is not None:
            return self.output_transform.apply(prediction)

        return prediction

    @override
    def save_state(self) -> dict[str, Any]:
        model_bytes = BytesIO()
        self.model.save(model_bytes)
        model_bytes.seek(0)

        input_bytes = BytesIO()
        pickle.dump(self.input_transform, input_bytes)
        input_bytes.seek(0)

        output_bytes = BytesIO()
        pickle.dump(self.output_transform, output_bytes)
        output_bytes.seek(0)

        return {
            "model": model_bytes.read(),
            "input_transform": input_bytes.read(),
            "output_transform": output_bytes.read(),
        }

    @override
    @classmethod
    def load_from_state(cls, state: dict[str, Any]) -> ModelWithTransform:
        model = Model.load(BytesIO(state["model"]))
        input_transform = pickle.load(BytesIO(state["input_transform"]))  # noqa: S301
        output_transform = pickle.load(BytesIO(state["output_transform"]))  # noqa: S301

        return cls(model, input_transform, output_transform)


def fullname(o: Any) -> str:
    klass = o.__class__
    module = klass.__module__
    if module == "builtins":
        return klass.__qualname__  # avoid outputs like 'builtins.str'
    return module + "." + klass.__qualname__
