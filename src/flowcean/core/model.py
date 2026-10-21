from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import BinaryIO, Protocol, Self, runtime_checkable

from .named import Named
from .transform import Identity, Transform


class Model[Input, Output](Named, ABC):
    """A predictor with type-preserving preprocessing and postprocessing.

    Frozen subclasses can predict with lazily initialized identity transforms.
    Learning strategies that attach transforms require a mutable model.
    """

    _pre_transform: Transform[Input, Input]
    _post_transform: Transform[Output, Output]

    @property
    def pre_transform(self) -> Transform[Input, Input]:
        """Per-instance preprocessing, initialized lazily for dataclasses."""
        if "_pre_transform" not in self.__dict__:
            identity: Transform[Input, Input] = Identity()
            object.__setattr__(self, "_pre_transform", identity)
        return self._pre_transform

    @pre_transform.setter
    def pre_transform(self, transform: Transform[Input, Input]) -> None:
        self._pre_transform = transform

    @property
    def post_transform(self) -> Transform[Output, Output]:
        """Per-instance postprocessing."""
        if "_post_transform" not in self.__dict__:
            identity: Transform[Output, Output] = Identity()
            object.__setattr__(self, "_post_transform", identity)
        return self._post_transform

    @post_transform.setter
    def post_transform(self, transform: Transform[Output, Output]) -> None:
        self._post_transform = transform

    def preprocess(self, input_features: Input) -> Input:
        """Preprocess the model input."""
        return self.pre_transform(input_features)

    @abstractmethod
    def _predict(self, input_features: Input) -> Output:
        """Predict from preprocessed input."""

    def predict(self, input_features: Input) -> Output:
        """Predict, applying preprocessing and postprocessing hooks."""
        return self.postprocess(self._predict(self.preprocess(input_features)))

    def __call__(self, input_features: Input) -> Output:
        return self.predict(input_features)

    def postprocess(self, output: Output) -> Output:
        """Postprocess the prediction."""
        return self.post_transform(output)

    def save(self, file: Path | str | BinaryIO) -> None:
        """Pickle the model and its transforms to a path or binary stream."""
        if isinstance(file, Path | str):
            path = Path(file)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as stream:
                pickle.dump(self, stream)
        else:
            pickle.dump(self, file)

    @classmethod
    def load(cls, file: Path | str | BinaryIO) -> Self:
        """Load a trusted pickle, checking its concrete model class.

        Call on the expected concrete class to retain its prediction types.
        Pickle can execute arbitrary code: never load untrusted files.
        """
        if isinstance(file, Path | str):
            with Path(file).open("rb") as stream:
                instance: object = pickle.load(stream)
        else:
            instance = pickle.load(file)
        if not isinstance(instance, cls):
            message = (
                f"Expected a {cls.__name__}, got {type(instance).__name__}"
            )
            raise TypeError(message)
        return instance


class Predictor[Input, Output](Protocol):
    """Read-only prediction contract for evaluation and deployment consumers."""

    @property
    def name(self) -> str: ...

    def predict(self, input_features: Input) -> Output: ...


@runtime_checkable
class ClassifierModel[Input, Output](Protocol):
    """Structural consumer contract for threshold-based classifier tuning."""

    threshold: float | None

    @property
    def name(self) -> str: ...

    def predict(self, input_features: Input) -> Output: ...

    def predict_proba(self, input_features: Input) -> Output: ...
