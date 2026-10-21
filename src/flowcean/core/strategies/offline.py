import math
from collections.abc import Iterable, Sequence

import polars as pl

from flowcean.core.environment.offline import OfflineEnvironment
from flowcean.core.learner import SupervisedLearner
from flowcean.core.metric import MetricCallable
from flowcean.core.model import ClassifierModel, Model, Predictor
from flowcean.core.report import Report, Reportable, ReportEntry
from flowcean.core.transform import Identity, Transform, inverse


def learn_offline[LearnedModel: Model[pl.LazyFrame, pl.LazyFrame]](
    environment: OfflineEnvironment[pl.LazyFrame],
    learner: SupervisedLearner[pl.LazyFrame, pl.LazyFrame, LearnedModel],
    inputs: list[str],
    outputs: list[str],
    *,
    input_transform: Transform[pl.LazyFrame, pl.LazyFrame] | None = None,
    output_transform: Transform[pl.LazyFrame, pl.LazyFrame] | None = None,
) -> LearnedModel:
    """Fit transforms and a mutable tabular model, retaining its concrete type.

    The output inverse is resolved before learning, so unsupported transforms
    or noninvertible fitted state cannot mutate the learner.
    """
    if input_transform is None:
        input_transform = Identity()
    if output_transform is None:
        output_transform = Identity()

    data = environment.observe()
    input_features = data.select(inputs)
    output_features = data.select(outputs)
    input_transform.fit(input_features)
    output_transform.fit(output_features)
    output_inverse = inverse(output_transform)
    model = learner.learn(
        input_transform(input_features),
        output_transform(output_features),
    )
    # New input preprocessing runs before any existing model preprocessing.
    model.pre_transform = input_transform | model.pre_transform
    model.post_transform = model.post_transform | output_inverse
    return model


def evaluate_offline(
    models: Predictor[pl.LazyFrame, pl.LazyFrame]
    | Iterable[Predictor[pl.LazyFrame, pl.LazyFrame]],
    environment: OfflineEnvironment[pl.LazyFrame],
    inputs: Sequence[str],
    outputs: Sequence[str],
    metrics: Sequence[
        MetricCallable[pl.LazyFrame, Reportable | dict[str, Reportable]]
    ],
) -> Report:
    """Evaluate tabular predictions against selected target columns."""
    model_list = list(models) if isinstance(models, Iterable) else [models]
    data = environment.observe()
    input_features = data.select(inputs)
    output_features = data.select(outputs)
    entries: dict[str, ReportEntry] = {}
    for model in model_list:
        predictions = model.predict(input_features)
        entries[model.name] = ReportEntry(
            {
                metric.name: metric(output_features, predictions)
                for metric in metrics
            }
        )
    return Report(entries)


def tune_threshold(
    model: ClassifierModel[pl.LazyFrame, pl.LazyFrame],
    environment: OfflineEnvironment[pl.LazyFrame],
    inputs: Sequence[str],
    outputs: Sequence[str],
    metric: MetricCallable[pl.LazyFrame, float],
    *,
    thresholds: Sequence[float] | None = None,
    num_thresholds: int = 19,
) -> tuple[float, dict[float, float]]:
    """Maximize a scalar metric over explicit binary decision thresholds.

    The original threshold, including None (native classification), is restored
    even when prediction or metric computation fails.
    """
    if thresholds is None:
        if num_thresholds < 1:
            message = "num_thresholds must be positive"
            raise ValueError(message)
        thresholds = [
            0.05 + 0.9 * index / max(num_thresholds - 1, 1)
            for index in range(num_thresholds)
        ]
    if not thresholds:
        message = "thresholds must not be empty"
        raise ValueError(message)
    if any(
        not math.isfinite(value) or not 0 <= value <= 1 for value in thresholds
    ):
        message = "thresholds must be finite values in [0, 1]"
        raise ValueError(message)

    data = environment.observe()
    input_features = data.select(inputs)
    output_features = data.select(outputs)
    original_threshold = model.threshold
    results: dict[float, float] = {}
    try:
        for threshold in thresholds:
            model.threshold = float(threshold)
            score = metric(output_features, model.predict(input_features))
            if not math.isfinite(score):
                message = "threshold metric must return a finite scalar score"
                raise ValueError(message)
            results[float(threshold)] = score
    finally:
        model.threshold = original_threshold
    best_threshold = max(results, key=lambda threshold: results[threshold])
    return best_threshold, results
