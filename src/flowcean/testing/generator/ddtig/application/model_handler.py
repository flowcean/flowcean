#!/usr/bin/env python3

import logging

import polars as pl

from flowcean.core.model import Model
from flowcean.sklearn.model import SciKitModel
from flowcean.torch.model import PyTorchModel

logger = logging.getLogger(__name__)

class ModelHandler:
    """A class to load a Flowcean model and access its underlying machine learning model.

    Attributes:
    ----------
    model : flowcean.core.model.Model
        The loaded Flowcean model.


    Methods:
    -------
    get_ml_model()
        Returns the underlying machine learning model from the Flowcean model.

    get_model_prediction()
        Returns predictions from the Flowcean model as a LazyFrame.

    get_model_prediction_as_lst()
        Returns predictions from the Flowcean model as a Python list.
    """


    def __init__(
        self,
        model: Model,
    ) -> None:
        """Initializes the ModelHandler.

        Args:
            file : File containing the Flowcean model.
        """
        # Load the Flowcean model from the given file
        self.model = model


    def get_ml_model(self):
        """Extracts the underlying machine learning model from the Flowcean model.

        Returns:
            The machine learning model.
        """
        if (type(self.model) == SciKitModel):
            ml_model = self.model.estimator
        elif (type(self.model) == PyTorchModel):
            ml_model = self.model.module
        else:
            raise ValueError(f"Unsupported model type: {type(self.model)}")
        logger.info("Extracted the underlying ML model from the Flowcean model successfully.")
        return ml_model


    def get_model_prediction(self, input: pl.DataFrame) -> pl.LazyFrame:
        """Generates predictions using the Flowcean model.

        Args:
            input: A Polars DataFrame containing input features.

        Returns:
            A LazyFrame with predicted outputs.
        """
        return self.model.predict(input.lazy())


    def get_model_prediction_as_lst(self, input: pl.DataFrame) -> list:
        """Generates predictions using the Flowcean model and returns them as a list.

        Args:
            input: A Polars DataFrame containing input features.

        Returns:
            A list of predicted output values.
        """
        pred_df = self.model.predict(input.lazy()).collect()
        target_name = pred_df.columns[-1]
        return pred_df[target_name].to_list()
