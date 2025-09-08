from flowcean.core.environment.incremental import IncrementalEnvironment
from flowcean.core.learner import SupervisedIncrementalLearner
from flowcean.core.model import Model
from flowcean.core.transform import Identity, InvertibleTransform, Transform


def learn_incremental(
    environment: IncrementalEnvironment,
    learner: SupervisedIncrementalLearner,
    inputs: list[str],
    outputs: list[str],
    input_transform: Transform | None = None,
    output_transform: InvertibleTransform | None = None,
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
        output_transform: The transform to apply to the output features.
            Its inverse will be part of the final model.

    Returns:
        The model learned from the environment.
    """
    if input_transform is None:
        input_transform = Identity()
    if output_transform is None:
        output_transform = Identity()

    model = None
    for data in environment:
        input_features = data.select(inputs)
        output_features = data.select(outputs)

        input_transform.fit_incremental(input_features)
        input_features = input_transform.apply(input_features)

        output_transform.fit_incremental(output_features)
        output_features = output_transform.apply(output_features)

        model = learner.learn_incremental(
            input_features,
            output_features,
        )

    if model is None:
        message = "No data found in environment."
        raise ValueError(message)

    model.post_transform |= output_transform.inverse()

    return model
