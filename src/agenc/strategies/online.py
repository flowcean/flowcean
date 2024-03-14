from agenc.core import (
    Model,
    ModelWithTransform,
    PassiveOnlineDataLoader,
    SupervisedIncrementalLearner,
    Transform,
    UnsupervisedIncrementalLearner,
)


def learn_incremental(
    environment: PassiveOnlineDataLoader,
    learner: SupervisedIncrementalLearner,
    inputs: list[str],
    outputs: list[str],
    input_transform: Transform | None = None,
    # feature transform
    # label transform
    # output transform
) -> Model:
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
