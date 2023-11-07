from typing import Any

from numpy.typing import NDArray

from agenc.core import Learner


class DummyLearner(Learner):
    def train(self, inputs: NDArray[Any], outputs: NDArray[Any]) -> None:
        super().train(inputs, outputs)
        pass

    def predict(self, inputs: NDArray[Any]) -> NDArray[Any]:
        return inputs
