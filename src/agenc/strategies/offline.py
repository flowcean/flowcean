from typing import Any

from agenc.core import (
    Metric,
    Model,
    ModelWithTransform,
    OfflineDataLoader,
    SupervisedLearner,
    Transform,
    UnsupervisedLearner,
)


def learn_offline(
    environment: OfflineDataLoader,
    learner: SupervisedLearner,
    inputs: list[str],
    outputs: list[str],
    input_transform: Transform | None = None,
    # feature transform
    # label transform
    # output transform
) -> Model:
    data = environment.get_data()
    input_features = data.select(inputs)
    output_features = data.select(outputs)

    if input_transform is not None:
        if isinstance(input_transform, UnsupervisedLearner):
            input_transform.fit(input_features)
        input_features = input_transform.transform(input_features)

    model = learner.learn(input_features, output_features)

    if input_transform is not None:
        return ModelWithTransform(model=model, transform=input_transform)
    return model


def evaluate(
    model: Model,
    environment: OfflineDataLoader,
    inputs: list[str],
    outputs: list[str],
    metrics: list[Metric],
) -> dict[str, Any]:
    data = environment.get_data()
    input_features = data.select(inputs)
    output_features = data.select(outputs)
    predictions = model.predict(input_features)
    return {
        metric.name: metric(output_features, predictions) for metric in metrics
    }
