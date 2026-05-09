from collections import deque
from collections.abc import Mapping, Sequence
from typing import Any

import polars as pl

from flowcean.hydra.selector.model import (
    HybridDecisionTreeModel,
    ModePredictionResult,
)


class StatefulHybridDecisionTreeSelector:
    def __init__(
        self,
        model: HybridDecisionTreeModel,
        seed_modes: Sequence[int] = (),
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
        self._mode_history: deque[int] | None = None
        if self.config.mode_history:
            self._mode_history = deque(
                (int(mode_id) for mode_id in seed_modes),
                maxlen=self.config.mode_history,
            )
        self._samples_seen = 0

    def predict(self, sample: Mapping[str, Any]) -> ModePredictionResult:
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
            return ModePredictionResult(ready=False, mode_id=None)

        row = self._engineered_row()
        result = self.model.predict_details(pl.DataFrame([row]))[0]
        if self._mode_history is not None and result.mode_id is not None:
            self._mode_history.append(result.mode_id)
        return result

    def _is_ready(self) -> bool:
        raw_history = max(
            self.config.state_history,
            self.config.input_history,
            self.config.derivative_history,
        )
        if len(self._raw_samples) <= raw_history:
            return False
        if self._mode_history is None:
            return True
        return len(self._mode_history) >= self.config.mode_history

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

        if self._mode_history is not None:
            history = list(self._mode_history)
            for step in range(1, self.config.mode_history + 1):
                row[f"mode_t_minus_{step}"] = history[-step]

        return row
