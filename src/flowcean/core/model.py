from __future__ import annotations

import pickle
from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, Protocol, final, runtime_checkable

from .named import Named
from .transform import Identity, Transform

if TYPE_CHECKING:
    from .data import Data


@runtime_checkable
class Model(Named, Protocol):
    """Base class for models.

    A model is used to predict outputs for given inputs.
    """

    pre_transform: Transform = Identity()
    post_transform: Transform = Identity()

    def preprocess(self, input_features: Data) -> Data:
        """Preprocess pipeline step."""
        return self.pre_transform.apply(input_features)

    @abstractmethod
    def _predict(self, input_features: Data) -> Data:
        """Predict outputs for the given inputs.

        Args:
            input_features: The inputs for which to predict the outputs.

        Returns:
            The predicted outputs.
        """

    def predict(self, input_features: Data) -> Data:
        """Predict outputs for given inputs, applying transforms and hooks."""
        input_features = self.preprocess(input_features)
        result = self._predict(input_features)
        return self.postprocess(result)

    @final
    def __call__(self, input_features: Data) -> Data:
        """Predict outputs for given inputs, applying transforms and hooks."""
        return self.predict(input_features)

    def postprocess(self, output: Data) -> Data:
        """Postprocess pipeline step."""
        return self.post_transform.apply(output)

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
