from __future__ import annotations

import json
import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)
REQUIRED_FEATURE_KEYS = ["name", "min", "max", "type", "nominal"]


class SystemSpecsHandler:
    """Load and decode system specifications for test-input generation.

    Supports loading from a JSON file or inferring values from a dataset.

    Attributes:
    ----------
    specs: dict
        Dictionary storing system specifications.

    n_features: int
        Number of features.

    Methods:
    -------
    get_n_features()
        Returns the number of features.

    get_nominal_features()
        Returns the indices of all nominal features.

    get_numerical_features()
        Returns the indices of all numerical features.

    get_int_features()
        Returns the indices of all features with type 'int'.

    extract_minmax_values()
        Extracts min and max values from specifications.

    extract_input_types()
        Extracts value types from specifications.

    extract_feature_names()
        Extracts feature names.

    extract_feature_names_with_idx()
        Extract feature names along with their corresponding indices.

    extract_feature_names_with_idx_reversed()
        Extract feature indices along with their corresponding names.
    """

    def __init__(
        self,
        data: pl.DataFrame | None = None,
        specs_file: Path | None = None,
    ) -> None:
        """Loads specifications from a JSON file or infers them from a dataset.

        For details on defining specifications in JSON, refer to README.md.

        Example JSON structure:
        {
            "features": [
                {
                    "name": "feature_0_name",
                    "min": 0,
                    "max": 100,
                    "type": "float" or "int",
                    "nominal": true or false
                },
                ...
            ]
        }

        Args:
            data: Dataset used to infer specifications if specs_file
                is not provided.
            specs_file: JSON file containing system specifications.
        """
        if data is not None:
            self._load_from_data(data)
        elif specs_file is not None:
            self._load_from_file(specs_file)
        else:
            msg = (
                "Either data or specs_file must be provided "
                "to load specifications."
            )
            raise ValueError(msg)

    def _load_from_data(self, data: pl.DataFrame) -> None:
        target_col = data.columns[-1]
        data = data.drop(target_col)

        features = []
        column_names = data.columns
        maxs = data.max().row(0)
        mins = data.min().row(0)
        dtypes = data.dtypes
        self.n_features = len(column_names)

        for i in range(len(column_names)):
            feature = {
                "name": column_names[i],
                "min": mins[i],
                "max": maxs[i],
                "type": "float" if dtypes[i].is_float() else "int",
            }
            unique_vals = data.select(
                pl.col(column_names[i]).unique(),
            ).to_series()
            feature["nominal"] = set(unique_vals) <= {0, 1}
            features.append(feature)

        self.specs = {"features": features}
        logger.info("Specifications successfully extracted from dataset.")

    def _load_from_file(self, specs_file: Path) -> None:
        with Path.open(specs_file) as f:
            try:
                self.specs = json.load(f)
            except json.JSONDecodeError as e:
                msg = f"Invalid JSON in specification file: {e}"
                raise ValueError(msg) from e
            except Exception as e:
                msg = f"Failed to load specification file: {e}"
                raise RuntimeError(msg) from e

        features = self.specs.get("features")
        if features is None:
            msg = "Invalid JSON structure. Refer to README.md."
            raise LookupError(msg)

        self.n_features = len(features)
        for feature in features:
            self._validate_feature(feature)
        logger.info("Specifications successfully extracted from file.")

    def _validate_feature(self, feature: dict) -> None:
        has_required_keys = all(
            key in feature for key in REQUIRED_FEATURE_KEYS
        )
        has_expected_size = len(feature) == len(REQUIRED_FEATURE_KEYS)
        if not has_required_keys or not has_expected_size:
            msg = "Invalid JSON structure. Refer to README.md."
            raise LookupError(msg)

        if not isinstance(feature["name"], str):
            msg = "'name' must be a string."
            raise TypeError(msg)
        if feature["type"] not in ["int", "float"]:
            msg = "'type' must be either 'int' or 'float'."
            raise TypeError(msg)
        if not isinstance(feature["min"], (int, float)) or not isinstance(
            feature["max"],
            (int, float),
        ):
            msg = "'min' and 'max' must be int or float."
            raise TypeError(msg)
        if feature["type"] == "int" and not (
            isinstance(feature["min"], int) and isinstance(feature["max"], int)
        ):
            msg = "'min' and 'max' must be int for type 'int'."
            raise TypeError(msg)
        if feature["type"] == "float" and not all(
            isinstance(v, (int, float))
            for v in [feature["min"], feature["max"]]
        ):
            msg = "'min' and 'max' must be numeric for type 'float'."
            raise TypeError(msg)
        if not isinstance(feature["nominal"], bool):
            msg = "'nominal' must be a boolean."
            raise TypeError(msg)
        if feature["nominal"] and feature["type"] != "int":
            msg = "Nominal features must be of type 'int'."
            raise TypeError(msg)
        if feature["min"] > feature["max"]:
            msg = "'min' must be smaller than or equal to 'max'."
            raise TypeError(msg)

    def get_n_features(self) -> int:
        """Returns the number of features defined in the specifications.

        Returns:
            Number of features.
        """
        return len(self.specs["features"])

    def get_nominal_features(self) -> list:
        """Returns the indices of nominal features.

        Returns:
            A list containing indices of nominal features.
        """
        return [
            feature
            for feature in range(self.n_features)
            if self.specs["features"][feature]["nominal"]
        ]

    def get_numerical_features(self) -> list:
        """Returns the indices of numerical features.

        Returns:
            A list containing indices of numerical features.
        """
        return [
            feature
            for feature in range(self.n_features)
            if not self.specs["features"][feature]["nominal"]
        ]

    def get_int_features(self) -> list:
        """Returns the indices of features with type 'int'.

        Returns:
            A list containing indices of integer-type features.
        """
        return [
            feature
            for feature in range(self.n_features)
            if self.specs["features"][feature]["type"] == "int"
        ]

    def extract_minmax_values(self) -> dict:
        """Extracts the min and max values for each feature.

        Returns:
            Dictionary mapping feature index to its min and max values.

        Example:
            {
                0: {'min': 0.5, 'max': 5.3},
                1: {'min': 0, 'max': 500},
                2: {'min': 0.0, 'max': 10.0}
            }
        """
        minmax_dict = {}
        for feature in range(self.n_features):
            min_value = self.specs["features"][feature]["min"]
            max_value = self.specs["features"][feature]["max"]
            feature_range = {"min": min_value, "max": max_value}
            minmax_dict[feature] = feature_range
        return minmax_dict

    def extract_input_types(self) -> dict:
        """Extract the input type of each feature from the specifications.

        Returns:
            A dictionary mapping feature index to its type.

        Example:
                {
                    0: {'type': 'float'},
                    1: {'type': 'int'},
                    2: {'type': 'float'}
                }
        """
        type_dict = {}
        for feature in range(self.n_features):
            input_type = self.specs["features"][feature]["type"]
            feature_type = {"type": input_type}
            type_dict.update({feature: feature_type})
        return type_dict

    def extract_feature_names(self) -> list:
        """Extract the name of each feature from the specifications.

        Returns:
            A list of feature names.

        Example:
                ["pH", "temperature", "humidity"]
        """
        return [
            self.specs["features"][feature]["name"]
            for feature in range(self.n_features)
        ]

    def extract_feature_names_with_idx(self) -> dict:
        """Extract feature names along with their corresponding indices.

        Returns:
            A dictionary mapping feature names to their indices.

        Example:
                {
                    "pH": 0,
                    "temperature": 1,
                    "humidity": 2
                }
        """
        return {
            self.specs["features"][feature]["name"]: feature
            for feature in range(self.n_features)
        }

    def extract_feature_names_with_idx_reversed(self) -> dict:
        """Extract feature indices along with their corresponding names.

        Returns:
            A dictionary mapping feature indices to their names.

        Example:
                {
                    0: "pH",
                    1: "temperature",
                    2: "humidity"
                }
        """
        return {
            feature: self.specs["features"][feature]["name"]
            for feature in range(self.n_features)
        }
