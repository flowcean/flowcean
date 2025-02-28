from flowcean.core.environment.incremental import IncrementalEnvironment
from flowcean.core.learner import SupervisedIncrementalLearner
from flowcean.core.model import Model, ModelWithTransform
from flowcean.core.transform import FitIncremetally, Transform


def learn_incremental(
    environment: IncrementalEnvironment,
    learner: SupervisedIncrementalLearner,
    inputs: list[str],
    outputs: list[str],
    input_transform: Transform | None = None,
    output_transform: Transform | None = None,
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
    model = None
    for data in environment:
        input_features = data.select(inputs)
        output_features = data.select(outputs)

        if isinstance(input_transform, FitIncremetally):
            input_transform.fit_incremental(input_features)

        if input_transform is not None:
            input_features = input_transform.apply(input_features)

        if isinstance(output_transform, FitIncremetally):
            output_transform.fit_incremental(output_features)

        if output_transform is not None:
            output_features = output_transform.apply(output_features)

        model = learner.learn_incremental(
            input_features,
            output_features,
        )

    if model is None:
        message = "No data found in environment."
        raise ValueError(message)

    if input_transform is None and output_transform is None:
        return model

    if output_transform is not None:
        output_transform = output_transform.inverse()

    return ModelWithTransform(
        model=model,
        input_transform=input_transform,
        output_transform=output_transform,
    )
