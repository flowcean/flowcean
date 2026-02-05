#!/usr/bin/env python3
import polars as pl

class TestCompiler():
    """
    A class that transforms abstract test inputs into executable test inputs
    compatible with Flowcean models.

    Attributes
    ----------
    n_features : int
        Number of features in the dataset.
        
    abst_testinputs : list
        List of abstract test inputs.

    Methods
    -------
    compute_executable_testinputs()
        Converts abstract test inputs into a polars DataFrame for execution.
    """

    def __init__(
        self,
        n_features: int,
        testinputs: list
    ) -> None:
        """
        Initializes the TestCompiler.

        Args:
            n_features : Number of features in the dataset.
            testinputs : List of abstract test inputs.
        """
        self.n_features = n_features
        self.abst_testinputs = testinputs

    # Initializes a dictionary to store test inputs sorted by feature index.
    def _init_input_dict(self) -> pl.DataFrame:
        input_dict = dict()
        for feature in range(self.n_features):
            input_dict[str(feature)] = []
        return input_dict

    def compute_executable_testinputs(self, feature_names: dict) -> pl.DataFrame:
        """
        Converts abstract test inputs into a polars DataFrame for execution
        on Flowcean models.

        Args:
            feature_names : Dictionary mapping feature indices to feature names.

        Returns:
            DataFrame where each column represents a feature
            and each row represents a test input.
        """
        input_dict =  self._init_input_dict()

        # Populate input dictionary with values from abstract test inputs
        for ati in self.abst_testinputs:
            for feature, value in enumerate(ati):
                input_dict[str(feature)].append(value)
        input_dict = dict(zip(feature_names, list(input_dict.values())))

        # Convert to polars DataFrame (Flowcean-compatible format)
        exec_testinputs = pl.from_dict(input_dict, strict=False)
        return exec_testinputs
