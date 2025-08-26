import logging
from typing import Any

from tabulate import tabulate

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
    if (
        isinstance(model, ModelWithTransform)
        and model.output_transform is not None
    ):
        output_features = model.output_transform.apply(output_features)
    report: dict[str, Any] = {}
    multi_output_report: dict[str, Any] = {}
    for metric in metrics:
        if hasattr(metric, "multi_output") and metric.multi_output:
            value = metric(output_features, predictions)
            multi_output_report[metric.name] = value
        else:
            for output_name in outputs:
                logger.info("Evaluating output: %s", output_name)
                single_output_true = output_features.select([output_name])
                single_output_pred = predictions.select([output_name])
                if output_name not in report:
                    report[output_name] = {}
                report[output_name][metric.name] = metric(
                    single_output_true,
                    single_output_pred,
                )
    if multi_output_report:
        report["multi_output"] = multi_output_report
    return Report(report)


def select_best_model(
    reports: dict[str, Report],
    metric_name: str,
    output_name: str,
) -> str | None:
    """Select the best model based on a given metric.

    Args:
        reports: A dictionary of model names to their evaluation reports.
        metric_name: The name of the metric to use for model selection.
        output_name: The name of the output to consider for the metric.
            Choose "multi_output" for multi-output metrics.

    Returns:
        The name of the best model.
    """
    best_model_name = None
    best_metric_value = float("inf")
    for model_name, report in reports.items():
        if report[output_name][metric_name] < best_metric_value:
            best_metric_value = report[output_name][metric_name]
            best_model_name = model_name

    return best_model_name


def print_report_table(report: Report) -> None:
    """Print the report as a table using tabulate."""
    table: list[list[str | float]] = []
    for output_name in report.entries:
        for metric_name, metric_value in report.entries[output_name].items():
            table.append([output_name, metric_name, metric_value])
    print(
        tabulate(
            table,
            headers=["Output", "Metric", "Value"],
            tablefmt="rounded_grid",
        ),
    )
