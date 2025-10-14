import logging
import numbers
from collections.abc import Iterable, Mapping, Sequence
from typing import Literal, cast

from flowcean.core.environment.offline import OfflineEnvironment
from flowcean.core.learner import SupervisedLearner
from flowcean.core.metric import Metric
from flowcean.core.model import Model
from flowcean.core.report import Report, Reportable, ReportEntry
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

    report = Report(entries)
    # Attach models by name for later retrieval (used by select_best_model)
    report.models_by_name = {m.name: m for m in models}  # type: ignore[attr-defined]
    return report


# Helper to convert various reportable values to float


def _to_float(value: object) -> float:
    if isinstance(value, numbers.Real):
        return float(value)
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as err:
        msg = f"Cannot convert metric value {value!r} to float"
        raise TypeError(msg) from err


def _score_from_entry(entry: ReportEntry, metric_name: str) -> float | None:
    if metric_name not in entry:
        return None
    raw = entry[metric_name]
    if isinstance(raw, Mapping):
        # Average over sub-metrics
        mapped = cast("Mapping[str, Reportable]", raw)
        values = [_to_float(v) for v in mapped.values()]
        return None if not values else sum(values) / len(values)
    return _to_float(raw)


def select_best_model(
    report: Report,
    metric_name: str,
    *,
    mode: Literal["max", "min"] = "max",
) -> Model:
    """Select the best model from a report.

    Args:
        report: The evaluation report.
        metric_name: The metric name to select by.
        mode: Whether to select the maximum ("max") or minimum ("min").

    Returns:
        The best model according to the metric and mode.
    """
    factor = 1.0 if mode == "max" else -1.0
    best_name: str | None = None
    best_score = float("-inf")

    for model_name, entry in report.items():
        score = _score_from_entry(entry, metric_name)
        if score is None:
            continue
        norm = factor * score
        if norm > best_score:
            best_score = norm
            best_name = model_name

    if best_name is None:
        msg = f"Metric '{metric_name}' not found in report for any model."
        raise ValueError(msg)

    models_by_name = getattr(report, "models_by_name", None)
    if isinstance(models_by_name, dict) and best_name in models_by_name:
        model = models_by_name[best_name]
        return cast("Model", model)

    msg = (
        "The report does not contain attached models. "
        "Re-run evaluation with evaluate_offline so that models are attached, "
        "or select using the model name from report keys."
    )
    raise ValueError(msg)
