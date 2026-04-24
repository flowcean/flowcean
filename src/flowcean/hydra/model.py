from dataclasses import dataclass

import polars as pl
from typing_extensions import override

from flowcean.core.model import Model
from flowcean.hydra.selector import (
    HybridDecisionTreeModel,
    ModePredictionResult,
)
from flowcean.hydra.selector.features import build_selector_inference_frame


@dataclass(frozen=True)
class HyDRABatchPrediction:
    outputs: pl.DataFrame
    row_indices: list[int]
    selector_results: list[ModePredictionResult]


class HyDRAModel(Model):
    def __init__(
        self,
        modes: list[Model],
        *,
        input_features: list[str],
        output_features: list[str],
        selector: HybridDecisionTreeModel | None = None,
    ) -> None:
        super().__init__()
        self.modes = modes
        self.input_features = input_features
        self.output_features = output_features
        self.selector = selector

    @override
    def _predict(
        self,
        input_features: pl.DataFrame | pl.LazyFrame,
    ) -> pl.LazyFrame:
        prediction = self.predict_with_diagnostics(input_features)
        frame = (
            input_features.collect()
            if isinstance(input_features, pl.LazyFrame)
            else input_features
        )
        if len(prediction.row_indices) != frame.height:
            message = (
                "HyDRAModel batch prediction requires the stateful "
                "selector runtime "
                "when selector warmup omits rows."
            )
            raise NotImplementedError(message)
        return prediction.outputs.lazy()

    def predict_with_diagnostics(
        self,
        input_features: pl.DataFrame | pl.LazyFrame,
    ) -> HyDRABatchPrediction:
        if not self.modes:
            message = "HyDRAModel contains no learned modes."
            raise ValueError(message)
        frame = (
            input_features.collect()
            if isinstance(input_features, pl.LazyFrame)
            else input_features
        )
        if len(self.modes) == 1:
            return HyDRABatchPrediction(
                outputs=self.modes[0]
                .predict(frame.select(self.input_features))
                .collect()
                .select(self.output_features),
                row_indices=list(range(frame.height)),
                selector_results=[
                    ModePredictionResult(
                        ready=True,
                        mode_id=0,
                        flow_model=self.modes[0],
                    )
                    for _ in range(frame.height)
                ],
            )
        if self.selector is None:
            message = (
                "HyDRAModel prediction requires a mode selector when multiple "
                "modes were learned."
            )
            raise NotImplementedError(message)

        selector_frame = build_selector_inference_frame(
            frame,
            self.selector.feature_config,
        )
        selector_results = self.selector.predict_details(
            selector_frame.features,
        )
        row_indices = selector_frame.row_metadata["row_index"].to_list()

        routed_outputs: list[pl.DataFrame] = []
        predicted_mode_ids = list(
            dict.fromkeys(
                result.mode_id
                for result in selector_results
                if result.mode_id is not None
            ),
        )
        for mode_id in predicted_mode_ids:
            if mode_id < 0 or mode_id >= len(self.modes):
                message = f"selector predicted unknown mode ID {mode_id}"
                raise ValueError(message)

            routed_row_indices = [
                row_index
                for row_index, result in zip(
                    row_indices,
                    selector_results,
                    strict=True,
                )
                if result.mode_id == mode_id
            ]
            if not routed_row_indices:
                continue

            mode_inputs = frame[routed_row_indices].select(self.input_features)
            mode_outputs = (
                self.modes[mode_id]
                .predict(mode_inputs)
                .collect()
                .select(self.output_features)
                .with_columns(pl.Series("__row_index", routed_row_indices))
            )
            routed_outputs.append(mode_outputs)

        outputs = (
            pl.concat(routed_outputs, how="vertical")
            .sort("__row_index")
            .drop("__row_index")
            .select(self.output_features)
            if routed_outputs
            else pl.DataFrame({name: [] for name in self.output_features})
        )

        return HyDRABatchPrediction(
            outputs=outputs,
            row_indices=row_indices,
            selector_results=selector_results,
        )
