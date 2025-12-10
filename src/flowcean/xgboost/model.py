import polars as pl
from typing_extensions import override
from xgboost import XGBClassifier, XGBRegressor

from flowcean.core.model import Model
from flowcean.core.transform import Identity


class XGBoostClassifierModel(Model):
    """Wrapper for an XGBoost classifier model with threshold support."""

    classifier: XGBClassifier

    input_features: list[str]
    output_features: list[str]

    def __init__(
        self,
        classifier: XGBClassifier,
        *,
        input_features: list[str],
        output_features: list[str],
        threshold: float = 0.5,
    ) -> None:
        self.classifier = classifier
        self.input_features = input_features
        self.output_features = output_features
        self.threshold = threshold
        # Initialize Protocol attributes
        self.pre_transform = Identity()
        self.post_transform = Identity()

    def _predict_proba(
        self,
        input_features: pl.LazyFrame,
    ) -> pl.LazyFrame:
        """Predict probability of positive class."""
        proba = self.classifier.predict_proba(
            input_features.select(self.input_features).collect().to_numpy(),
        )[:, 1]  # Get positive class probability

        return pl.from_numpy(
            proba,
            self.output_features,
        ).lazy()

    def predict_proba(self, input_features: pl.LazyFrame) -> pl.LazyFrame:
        """Predict class probabilities, applying preprocessing transforms.

        Args:
            input_features: The inputs for which to predict probabilities.

        Returns:
            The predicted probabilities for the positive class.
        """
        input_features = self.preprocess(input_features)
        return self._predict_proba(input_features)

    @override
    def _predict(
        self,
        input_features: pl.LazyFrame,
    ) -> pl.LazyFrame:
        """Predict class labels using threshold."""
        if self.threshold is not None:
            # Use threshold-based prediction
            probas = self._predict_proba(input_features).collect()
            predictions = {}
            for col in probas.columns:
                predictions[col] = (probas[col] >= self.threshold).cast(
                    pl.Int64,
                )
            return pl.LazyFrame(predictions)

        # Use default prediction
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
