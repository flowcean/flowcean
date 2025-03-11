import logging

from flowcean.core.environment.offline import OfflineEnvironment
from flowcean.core.learner import SupervisedLearner
from flowcean.core.metric import OfflineMetric
from flowcean.core.model import Model, ModelWithTransform
from flowcean.core.report import Report
from flowcean.core.transform import FitIncremetally, FitOnce, Transform

logger = logging.getLogger(__name__)


def learn_offline(
    environment: OfflineEnvironment,
    learner: SupervisedLearner,
    inputs: list[str],
    outputs: list[str],
    *,
    input_transform: Transform | None = None,
    output_transform: Transform | None = None,
) -> Model:
    """Learn from an offline environment.

    Learn from an offline environment by learning from the input-output pairs.

    Args:
        environment: The offline environment.
        learner: The supervised learner.
        inputs: The input feature names.
        outputs: The output feature names.
        input_transform: The transform to apply to the input features.
            Will be part of the final model.
        output_transform: The transform to apply to the output features.
            Its inverse will be part of the final model.

    Returns:
        The model learned from the environment.
    """
    logger.info("Learning with offline strategy")
    data = environment.observe()
    logger.info("Selecting input and output features")
    input_features = data.select(inputs)
    output_features = data.select(outputs)

    if isinstance(input_transform, FitOnce):
        input_transform.fit(input_features)
    elif isinstance(input_transform, FitIncremetally):
        input_transform.fit_incremental(input_features)

    if input_transform is not None:
        input_features = input_transform.apply(input_features)

    if isinstance(output_transform, FitOnce):
        output_transform.fit(output_features)
    elif isinstance(output_transform, FitIncremetally):
        output_transform.fit_incremental(output_features)

    if output_transform is not None:
        output_features = output_transform.apply(output_features)

    logger.info("Learning model")
    model = learner.learn(inputs=input_features, outputs=output_features)

    if input_transform is None and output_transform is None:
        return model

    if output_transform is not None:
        output_transform = output_transform.inverse()

    return ModelWithTransform(
        model,
        input_transform,
        output_transform,
    )


def evaluate_offline(
    model: Model,
    environment: OfflineEnvironment,
    inputs: list[str],
    outputs: list[str],
    metrics: list[OfflineMetric],
) -> Report:
    """Evaluate a model on an offline environment.

    Evaluate a model on an offline environment by predicting the outputs from
    the inputs and comparing them to the true outputs.

    Args:
        model: The model to evaluate.
        environment: The offline environment.
        inputs: The input feature names.
        outputs: The output feature names.
        metrics: The metrics to evaluate the model with.

    Returns:
        The evaluation report.
    """
    data = environment.observe()
    input_features = data.select(inputs)
    output_features = data.select(outputs)
    predictions = model.predict(input_features)
    return Report(
        {
            metric.name: metric(output_features, predictions.lazy())
            for metric in metrics
        },
    )
