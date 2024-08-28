import logging

from flowcean.core.learner import SupervisedLearner, UnsupervisedLearner
from flowcean.core.metric import OfflineMetric
from flowcean.core.model import Model, ModelWithTransform
from flowcean.core.transform import Transform
from flowcean.environments.dataset import OfflineEnvironment
from flowcean.metrics.report import Report

logger = logging.getLogger(__name__)


def learn_offline(
    environment: OfflineEnvironment,
    learner: SupervisedLearner,
    inputs: list[str],
    outputs: list[str],
    input_transform: Transform | None = None,
) -> Model:
    """Learn from an offline environment.

    Learn from an offline environment by learning from the input-output pairs.

    Args:
        environment: The offline environment.
        learner: The supervised learner.
        inputs: The input feature names.
        outputs: The output feature names.
        input_transform: The transform to apply to the input features.

    Returns:
        The model learned from the environment.
    """
    logger.info("Learning with offline strategy")
    data = environment.observe()
    logger.info("Selecting input and output features")
    input_features = data.select(inputs)
    output_features = data.select(outputs)

    if input_transform is not None:
        if isinstance(input_transform, UnsupervisedLearner):
            logger.info("Learning input transform")
            input_transform.fit(input_features)
        logger.info("Applying input transform")
        input_features = input_transform.transform(input_features)

    logger.info("Learning model")
    model = learner.learn(input_features, output_features)

    if input_transform is not None:
        return ModelWithTransform(model=model, transform=input_transform)
    return model


def evaluate_offline(
    model: Model,
    environment: OfflineEnvironment,
    inputs: list[str],
    outputs: list[str],
    metrics: list[OfflineMetric],
) -> Report:
    data = environment.observe()
    input_features = data.select(inputs)
    output_features = data.select(outputs)
    predictions = model.predict(input_features)
    return Report(
        {
            metric.name: metric(output_features, predictions)
            for metric in metrics
        },
    )
