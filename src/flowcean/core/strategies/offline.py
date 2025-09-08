import logging
from collections.abc import Iterable, Sequence

from flowcean.core.environment.offline import OfflineEnvironment
from flowcean.core.learner import SupervisedLearner
from flowcean.core.metric import Metric
from flowcean.core.model import Model
from flowcean.core.report import Report, ReportEntry
from flowcean.core.transform import Identity, InvertibleTransform, Transform

logger = logging.getLogger(__name__)


def learn_offline(
    environment: OfflineEnvironment,
    learner: SupervisedLearner,
    inputs: list[str],
    outputs: list[str],
    *,
    input_transform: Transform | None = None,
    output_transform: InvertibleTransform | None = None,
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
    if input_transform is None:
        input_transform = Identity()
    if output_transform is None:
        output_transform = Identity()

    logger.info("Learning with offline strategy")
    data = environment.observe()

    logger.info("Selecting input and output features")
    input_features = data.select(inputs)
    output_features = data.select(outputs)

    logger.info("Fitting transforms and applying them to features")
    input_transform.fit(input_features)
    input_features = input_transform.apply(input_features)

    logger.info("Fitting output transform and applying it to output features")
    output_transform.fit(output_features)
    output_features = output_transform.apply(output_features)

    logger.info("Learning model")
    model = learner.learn(inputs=input_features, outputs=output_features)

    model.post_transform |= output_transform.inverse()

    return model


def evaluate_offline(
    models: Model | Iterable[Model],
    environment: OfflineEnvironment,
    inputs: Sequence[str],
    outputs: Sequence[str],
    metrics: Sequence[Metric],
) -> Report:
    """Evaluate a model on an offline environment.

    Evaluate a model on an offline environment by predicting the outputs from
    the inputs and comparing them to the true outputs.

    Args:
        models: The models to evaluate.
        environment: The offline environment.
        inputs: The input feature names.
        outputs: The output feature names.
        metrics: The metrics to evaluate the model with.

    Returns:
        The evaluation report.
    """
    if not isinstance(models, Iterable):
        models = [models]
    data = environment.observe()
    input_features = data.select(inputs)
    output_features = data.select(outputs)
    entries: dict[str, ReportEntry] = {}

    for model in models:
        predictions = model.predict(input_features)

        entries[model.name] = ReportEntry(
            {
                metric.name: metric(output_features, predictions.lazy())
                for metric in metrics
            },
        )
    return Report(entries)
