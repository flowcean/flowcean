from typing import Any

from flowcean.core import Metric, Model, OfflineEnvironment


class Report:
    def __init__(self, metrics: dict[str, Any]) -> None:
        self.metrics = metrics

    def __str__(self) -> str:
        return "\n".join(
            f"{name}: {value}" for name, value in self.metrics.items()
        )


def evaluate(
    model: Model,
    environment: OfflineEnvironment,
    inputs: list[str],
    outputs: list[str],
    metrics: list[Metric],
) -> Report:
    data = environment.get_data()
    input_features = data.select(inputs)
    output_features = data.select(outputs)
    predictions = model.predict(input_features)
    return Report(
        {
            metric.name: metric(output_features, predictions)
            for metric in metrics
        },
    )
