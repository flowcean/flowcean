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


def tune_threshold(
    model: Model,
    environment: OfflineEnvironment,
    inputs: Sequence[str],
    outputs: Sequence[str],
    metric: Metric,
    *,
    thresholds: Sequence[float] | None = None,
    num_thresholds: int = 19,
) -> tuple[float, dict[float, float]]:
    """Find optimal decision threshold for a classifier.

    Evaluates the model at multiple threshold values and returns the threshold
    that maximizes the given metric. Only works with models that support
    probability predictions (i.e., have a threshold attribute and
    predict_proba method).

    Args:
        model: The classifier model to tune. Must have a threshold attribute.
        environment: The offline environment with validation/test data.
        inputs: The input feature names.
        outputs: The output feature names.
        metric: The metric to optimize (e.g., FBetaScore, Accuracy).
        thresholds: Specific thresholds to evaluate. If None, generates
            num_thresholds evenly spaced values between 0.05 and 0.95.
        num_thresholds: Number of thresholds to evaluate if thresholds is
            None (default: 19).

    Returns:
        Tuple of (best_threshold, results_dict) where results_dict maps
        each threshold to its metric score.

    Raises:
        AttributeError: If model does not have a threshold attribute.
        ValueError: If model does not support probability predictions.

    Example:
        >>> from flowcean.sklearn.metrics.classification import FBetaScore
        >>> metric = FBetaScore(beta=1.0)
        >>> best_thr, results = tune_threshold(
        ...     model, eval_env, inputs, outputs, metric
        ... )
        >>> print(f"Best threshold: {best_thr:.3f}")
        >>> model.threshold = best_thr  # Apply the best threshold
    """
    import numpy as np

    # Check if model supports thresholds
    if not hasattr(model, "threshold"):
        msg = (
            f"Model {model.__class__.__name__} does not have a threshold "
            "attribute. Only classifier models support threshold tuning."
        )
        raise AttributeError(msg)

    if not hasattr(model, "predict_proba"):
        msg = (
            f"Model {model.__class__.__name__} does not implement "
            "predict_proba. Threshold tuning requires probability predictions."
        )
        raise ValueError(msg)

    # Generate thresholds if not provided
    thresholds_to_test: Sequence[float]
    if thresholds is None:
        thresholds_to_test = list(np.linspace(0.05, 0.95, num_thresholds))
    else:
        thresholds_to_test = thresholds

    # Store original threshold
    original_threshold = model.threshold

    # Evaluate at each threshold
    results: dict[float, float] = {}
    best_threshold = 0.5
    best_score = -float("inf")

    logger.info(
        "Tuning threshold for %s using %s",
        model.name,
        metric.name,
    )

    try:
        for threshold in thresholds_to_test:
            # Set threshold and evaluate
            model.threshold = float(threshold)
            report = evaluate_offline(
                model,
                environment,
                inputs,
                outputs,
                [metric],
            )

            # Extract metric score from report
            # Report is dict[str, ReportEntry]
            # ReportEntry is dict[str, Reportable]
            model_entry = report.get(model.name)
            if model_entry is None:
                continue

            score_value = model_entry.get(metric.name)
            if score_value is None:
                continue

            # Convert Reportable to float (it should be a numeric value)
            if isinstance(score_value, (int, float)):
                score = float(score_value)
            else:
                # Try to convert via string representation
                try:
                    score = float(str(score_value))
                except (ValueError, TypeError):
                    logger.warning(
                        "Could not convert metric value to float: %s",
                        score_value,
                    )
                    continue

            results[float(threshold)] = score

            # Track best threshold
            if score > best_score:
                best_score = score
                best_threshold = float(threshold)

            logger.debug(
                "Threshold %.3f: %s = %.4f",
                threshold,
                metric.name,
                score,
            )

    finally:
        # Restore original threshold
        model.threshold = original_threshold

    logger.info(
        "Best threshold: %.3f with %s = %.4f",
        best_threshold,
        metric.name,
        best_score,
    )

    return best_threshold, results
