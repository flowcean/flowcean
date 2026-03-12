from __future__ import annotations

import numpy as np
import polars as pl
from polars import DataFrame
from sklearn.neighbors import KernelDensity

from flowcean.testing.generator.ddtig.application import ModelHandler
from flowcean.testing.generator.ddtig.user_interface import SystemSpecsHandler


class DataModel:
    """A class used to generate synthetic samples based on the training data distribution.

    Attributes:
    ----------
    data : pl.DataFrame
        Original training data used in the Flowcean model.
    
    col_names : list
        Names of the columns in the training data.
    
    n_features : int
        Number of features in the dataset.
        
    model_handler : ModelHandler
        ModelHandler object used to produce predictions.
    
    int_features : list
        List of indices for features of type int.

    Methods:
    -------
    generate_dataset()
        Generate n random samples based on the data distribution or use original data.
    """


    def __init__(
        self,
        data: pl.DataFrame,
        seed: int,
        model_handler: ModelHandler,
        specs_handler: SystemSpecsHandler,
    ) -> None:
        """Initializes the DataModel.

        Args:
            data : Original training data used in the Flowcean model.
            seed : Random seed for reproducibility.
            model_handler : ModelHandler object used to produce predictions.
            specs_handler : SystemSpecsHandler object storing system specifications.
        """
        self.data = data
        self.seed = seed
        self.col_names = data.columns
        self.model_handler = model_handler

        self.n_features = specs_handler.get_n_features()
        self.int_features = specs_handler.get_int_features()

    def _compute_dist(self) -> KernelDensity:
        """Computes a kernel density estimate (KDE) of the training data.

        Kernel Density Estimation (KDE) is a nonparametric method 
        for estimating the probability density function of a dataset. 
        Kernel used is the Gaussian kernel, which assigns higher 
        weights to nearby points and lower weights to distant ones, 
        producing a smooth, bell-shaped density curve.

        Returns:
            Fitted KDE model.
        """
        data = self.data.to_numpy()
        return KernelDensity(bandwidth="silverman").fit(data)

    def _generate_samples(self,
                          n_samples: int,
                          int_features: list = []) -> DataFrame:
        """Generates n random input samples for all features using KDE.

        Args:
            n_samples : Number of samples to generate.
            int_features : List of indices for features of type int.

        Returns:
            DataFrame containing n synthetic samples.
            Example (n = 5):
            ┌────────┬──────────┬────────┬─────┐
            │ Length ┆ Diameter ┆ Height ┆ M   │
            │ ---    ┆ ---      ┆ ---    ┆ --- │
            │ f64    ┆ f64      ┆ f64    ┆ i64 │
            ╞════════╪══════════╪════════╪═════╡
            │ 0.6386 ┆ 0.437    ┆ 0.1337 ┆ 0   │
            │ 0.4278 ┆ 0.335    ┆ 0.0876 ┆ 0   │
            │ 0.7859 ┆ 0.649    ┆ 0.2223 ┆ 0   │
            │ 0.5587 ┆ 0.4507   ┆ 0.1767 ┆ 1   │
            │ 0.5265 ┆ 0.4673   ┆ 0.1601 ┆ 1   │
            └────────┴──────────┴────────┴─────┘
        """
        samples = pl.DataFrame()
        kde = self._compute_dist()
        samples_array = kde.sample(n_samples, random_state=self.seed)
        for i in range(self.n_features):
            # Round values for integer-type features
            if i in int_features:
                feature_samples = np.round(samples_array[:,i]).astype(int).tolist()
            else:
                feature_samples = samples_array[:,i].tolist()
            samples.insert_column(i, pl.Series(self.col_names[i], feature_samples))
        return samples

    def generate_dataset(self,
                         original_data: bool = False,
                         n_samples: int = 0) -> list:
        """Generates a dataset of inputs and corresponding model predictions.

        If original_data is True, uses the original training data.
        Otherwise, generates synthetic samples using KDE.

        Args:
            original_data : Whether to use original training data or generate synthetic samples.
            n_samples : Number of synthetic samples to generate.

        Returns:
            List of tuples containing input dictionaries and model outputs.
            Example (n_samples = 1):
            [({'Length': 0.5093, 'Diameter': 0.3886, 'Height': 0.1106, 'M': 0}, 8.6006)]
        """
        if original_data:
            training_inputs = self.data
        else:
            training_inputs = self._generate_samples(n_samples, self.int_features)
        training_outputs = self.model_handler.get_model_prediction(training_inputs).collect()
        samples_input_lst = training_inputs.to_dicts()
        samples_output_lst = pl.Series(training_outputs.select(training_outputs.columns[0])).to_list()
        samples = [(inputs, output) for inputs, output in zip(samples_input_lst, samples_output_lst, strict=False)]
        return samples
