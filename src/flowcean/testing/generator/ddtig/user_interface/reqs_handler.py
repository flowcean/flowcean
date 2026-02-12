from __future__ import annotations
import json
from pathlib import Path
from typing import TextIO
from contextlib import nullcontext
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class RequirementsHandler():
    """
    A class that loads and validates test requirements from a JSON file,
    extracting information necessary for generating test inputs.

    Attributes
    ----------
    requirements : dict
        Dictionary storing the test requirements.
    """

    # Parameters expected to be of type int
    int_params = ["n_testinputs", "sample_limit", "n_predictions", "max_depth"]

    # Parameters expected to be of type float
    float_params = ["epsilon", "performance_threshold"]

    # Parameters expected to be of type str with restricted values
    str_params_values = [["bva", "dtc"]]
    str_params        = ["test_coverage_criterium"]

    # Parameters expected to be of type bool
    bool_params = ["inverse_alloc"]

    # Parameters that must be present in the requirements file
    must_params = ["test_coverage_criterium", "n_testinputs"]


    def __init__(
        self,
        reqs_file: Path | str | TextIO,
        ) -> None:
        """
        Loads and validates test requirements from a JSON file.
        For details on defining requirements, refer to README.md.

        Example:
        {
            "test_coverage_criterium": "bva", 
            "n_testinputs": 2000,
            ...
        }

        Args:
            reqs_file : JSON file containing test requirements.
        """

        # Load JSON file
        file_ctx = open(reqs_file) if isinstance(reqs_file, (Path, str)) else nullcontext(reqs_file)
        with file_ctx as f:
            try:
                self.requirements = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in requirements file: {e}") from e
            except Exception as e:
                raise RuntimeError(f"Failed to load requirements file: {e}") from e

        # Validate presence of mandatory parameters
        for must_param in self.must_params:
            if must_param not in self.requirements:
                raise KeyError(f"Missing required parameter: '{must_param}'") 

        # Validate integer parameters
        for int_param in self.int_params:
            if int_param in self.requirements:
                value = self.requirements[int_param]
                if not isinstance(value, int):
                    raise TypeError(f"Expected type 'int', but got '{type(value).__name__}' instead.")

        # Validate float parameters
        for float_param in self.float_params:
            if float_param in self.requirements:
                value = self.requirements[float_param]
                if not (isinstance(value, float) or (isinstance(value, int) and not isinstance(value, bool))):
                    raise TypeError(f"Expected type 'float', but got '{type(value).__name__}' instead.")
        
         # Validate string parameters and their allowed values
        for str_param, str_params_value in zip(self.str_params, self.str_params_values):
            if str_param in self.requirements:
                value = self.requirements[str_param]
                if not isinstance(value, str):
                    raise TypeError(f"Expected type 'str', but got '{type(value).__name__}' instead.")
                if value not in str_params_value:
                    raise ValueError(f"Invalid value: '{value}'. Allowed values are: {str_params_value}")
        
        # Validate boolean parameters
        for bool_param in self.bool_params:
            if bool_param in self.requirements:
                value = self.requirements[bool_param]
                if not isinstance(value, bool):
                    raise TypeError(f"Expected type 'bool', but got '{type(value).__name__}' instead.")
