import polars as pl
from typing_extensions import override
from xgboost import XGBClassifier, XGBRegressor

from flowcean.core.model import Model


class XGBoostClassifierModel(Model):
    """Wrapper for an XGBoost classifier model."""

    classifier: XGBClassifier

    input_features: list[str]
    output_features: list[str]

    def __init__(
        self,
        classifier: XGBClassifier,
        *,
        input_features: list[str],
        output_features: list[str],
    ) -> None:
        super().__init__()
        self.classifier = classifier
        self.input_features = input_features
        self.output_features = output_features

    def __getstate__(self) -> dict:
        """Remove callbacks when pickling (they contain unpickleable locks)."""
        state = self.__dict__.copy()
        # Remove callbacks from the classifier before pickling
        if "classifier" in state:
            classifier = state["classifier"]
            params = classifier.get_params()
            if "callbacks" in params:
                # Create a new classifier without callbacks
                params_without_callbacks = {
                    k: v for k, v in params.items() if k != "callbacks"
                }
                state["classifier"] = XGBClassifier(**params_without_callbacks)
                # Copy trained model (need _Booster for pickling)
                state["classifier"]._Booster = classifier._Booster  # noqa: SLF001
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore state after unpickling."""
        self.__dict__.update(state)

    @override
    def _predict(
        self,
        input_features: pl.LazyFrame,
    ) -> pl.LazyFrame:
        return pl.from_numpy(
            self.classifier.predict(
                input_features.select(self.input_features)
                .collect()
                .to_numpy(),
            ),
            self.output_features,
        ).lazy()


class XGBoostRegressorModel(Model):
    """Wrapper for an XGBoost regressor model."""

    regressor: XGBRegressor

    input_features: list[str]
    output_features: list[str]

    def __init__(
        self,
        regressor: XGBRegressor,
        *,
        input_features: list[str],
        output_features: list[str],
    ) -> None:
        super().__init__()
        self.regressor = regressor
        self.input_features = input_features
        self.output_features = output_features

    def __getstate__(self) -> dict:
        """Remove callbacks when pickling (they contain unpickleable locks)."""
        state = self.__dict__.copy()
        # Remove callbacks from the regressor before pickling
        if "regressor" in state:
            regressor = state["regressor"]
            params = regressor.get_params()
            if "callbacks" in params:
                # Create a new regressor without callbacks
                params_without_callbacks = {
                    k: v for k, v in params.items() if k != "callbacks"
                }
                state["regressor"] = XGBRegressor(**params_without_callbacks)
                # Copy trained model (need _Booster for pickling)
                state["regressor"]._Booster = regressor._Booster  # noqa: SLF001
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore state after unpickling."""
        self.__dict__.update(state)

    @override
    def _predict(
        self,
        input_features: pl.LazyFrame,
    ) -> pl.LazyFrame:
        return pl.from_numpy(
            self.regressor.predict(
                input_features.select(self.input_features)
                .collect()
                .to_numpy(),
            ),
            self.output_features,
        ).lazy()
