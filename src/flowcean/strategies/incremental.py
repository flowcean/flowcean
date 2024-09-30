from abc import abstractmethod
from collections.abc import Sequence
from typing import Self

from flowcean.core import (
    IncrementalEnvironment,
    Model,
    ModelWithTransform,
    SupervisedIncrementalLearner,
    Transform,
    UnsupervisedIncrementalLearner,
)


class WithFeatures:
    @abstractmethod
    def select(self, features: Sequence[str]) -> Self:
        pass


def learn_incremental[Observation: WithFeatures](
    environment: IncrementalEnvironment[Observation],
    learner: SupervisedIncrementalLearner,
    inputs: list[str],
    outputs: list[str],
    input_transform: Transform | None = None,
) -> Model:
    """Learn from a incremental environment.

    Learn from a incremental environment by incrementally learning from
    the input-output pairs. The learning process stops when the environment
    ends.

    Args:
        environment: The incremental environment.
        learner: The supervised incremental learner.
        inputs: The input feature names.
        outputs: The output feature names.
        input_transform: The transform to apply to the input features.

    Returns:
        The model learned from the environment.
    """
    model = None
    for data in environment:
        input_features = data.select(inputs)
        output_features = data.select(outputs)

        if input_transform is not None:
            if isinstance(input_transform, UnsupervisedIncrementalLearner):
                input_transform.fit_incremental(input_features)
            input_features = input_transform.transform(input_features)

        model = learner.learn_incremental(input_features, output_features)

    if model is None:
        message = "No data found in environment."
        raise ValueError(message)
    if input_transform is not None:
        return ModelWithTransform(model=model, transform=input_transform)
    return model
