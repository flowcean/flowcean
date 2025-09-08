from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, final

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
    def save(self, file: Path | str | BinaryIO) -> None:
        """Save the model to the file.

        This method can be used to save a flowcean model to a file or a
        file-like object. To save a model to a file use

        ```python
        model.save("model.fml")
        ```

        The resulting file will contain the model any any attached transforms.
        It can be loaded again using the `load` method from the `Model` class.

        This method uses pickle to serialize the model, so child classes should
        ensure that all attributes are pickleable. If this is not the case, the
        child class should override this method to implement custom
        serialization logic, or use the `__getstate__` and `__setstate__`
        methods to control what is serialized (see https://docs.python.org/3/library/pickle.html#pickling-class-instances).

        Args:
            file: The file like object to save the model to.
        """
        if isinstance(file, Path | str):
            path = Path(file)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as f:
                pickle.dump(self, f)
        else:
            pickle.dump(self, file)

    @staticmethod
    def load(file: Path | str | BinaryIO) -> Model:
        """Load a model from file.

        This method can be used to load a previously saved flowcean model from
        a file or a file-like object. To load a model from a file use

        ```python
        model = Model.load("model.fml")
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
        if isinstance(file, Path | str):
            with Path(file).open("rb") as f:
                instance = pickle.load(f)
        else:
            instance = pickle.load(file)

        return instance


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
