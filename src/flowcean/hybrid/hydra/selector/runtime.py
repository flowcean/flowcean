from collections import deque
from collections.abc import Mapping, Sequence
from typing import Any

import polars as pl

from .model import (
    FlowPredictionResult,
    HybridDecisionTreeModel,
)


class StatefulHybridDecisionTreeSelector:
    def __init__(
        self,
        model: HybridDecisionTreeModel,
        seed_flows: Sequence[int] = (),
    ) -> None:
        self.model = model
        self.config = model.feature_config
        raw_history = max(
            self.config.state_history,
            self.config.input_history,
            self.config.derivative_history,
        )
        self._raw_samples: deque[dict[str, Any]] = deque(
            maxlen=max(raw_history + 1, 1),
        )
        self._flow_history: deque[int] | None = None
        if self.config.flow_history:
            self._flow_history = deque(
                (int(flow_id) for flow_id in seed_flows),
                maxlen=self.config.flow_history,
            )
        self._samples_seen = 0

    def predict(self, sample: Mapping[str, Any]) -> FlowPredictionResult:
        missing_columns = set(self.config.required_columns()) - set(sample)
        if missing_columns:
            message = "missing required selector columns"
            msg = f"{message}: {sorted(missing_columns)}"
            raise ValueError(msg)

        raw_sample = {
            column: sample[column] for column in self.config.required_columns()
        }
        self._raw_samples.append(raw_sample)
        self._samples_seen += 1
        if not self._is_ready():
            return FlowPredictionResult(ready=False, flow_id=None)

        row = self._engineered_row()
        result = self.model.predict_details(pl.DataFrame([row]))[0]
        if self._flow_history is not None and result.flow_id is not None:
            self._flow_history.append(result.flow_id)
        return result

    def _is_ready(self) -> bool:
        raw_history = max(
            self.config.state_history,
            self.config.input_history,
            self.config.derivative_history,
        )
        if len(self._raw_samples) <= raw_history:
            return False
        if self._flow_history is None:
            return True
        return len(self._flow_history) >= self.config.flow_history

    def _engineered_row(self) -> dict[str, Any]:
        current_sample = self._raw_samples[-1]
        row = {
            column: current_sample[column]
            for column in (
                *self.config.state_features,
                *self.config.input_features,
                *self.config.derivative_features,
            )
        }

        for step in range(1, self.config.state_history + 1):
            previous_sample = self._raw_samples[-(step + 1)]
            for column in self.config.state_features:
                row[f"{column}_t_minus_{step}"] = previous_sample[column]

        for step in range(1, self.config.input_history + 1):
            previous_sample = self._raw_samples[-(step + 1)]
            for column in self.config.input_features:
                row[f"{column}_t_minus_{step}"] = previous_sample[column]

        for step in range(1, self.config.derivative_history + 1):
            previous_sample = self._raw_samples[-(step + 1)]
            for column in self.config.derivative_features:
                row[f"{column}_t_minus_{step}"] = previous_sample[column]

        if self._flow_history is not None:
            history = list(self._flow_history)
            for step in range(1, self.config.flow_history + 1):
                row[f"flow_t_minus_{step}"] = history[-step]

        return row
