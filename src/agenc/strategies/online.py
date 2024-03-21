from agenc.core import (
    Model,
    ModelWithTransform,
    PassiveOnlineEnvironment,
    SupervisedIncrementalLearner,
    Transform,
    UnsupervisedIncrementalLearner,
)


def learn_incremental(
    environment: PassiveOnlineEnvironment,
    learner: SupervisedIncrementalLearner,
    inputs: list[str],
    outputs: list[str],
    input_transform: Transform | None = None,
) -> Model:
    """Learn from a passive online environment.

    Learn from a passive online environment by incrementally learning from
    the input-output pairs. The learning process stops when the environment
    ends.

    Args:
        environment: The passive online environment.
        learner: The supervised incremental learner.
        inputs: The input feature names.
        outputs: The output feature names.
        input_transform: The transform to apply to the input features.

    Returns:
        The model learned from the environment.
    """
    model = None
    for data in environment.get_next_data():
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
