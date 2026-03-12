import logging
import pickle
from pathlib import Path
from typing import Any

import polars as pl
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from tabulate import tabulate

from flowcean.core import Model
from flowcean.testing.generator.ddtig.application import ModelHandler
from flowcean.testing.generator.ddtig.domain import (
    EquivalenceClassesHandler,
    HoeffdingTree,
    TestCompiler,
    TestGenerator,
    TestTree,
)
from flowcean.testing.generator.ddtig.user_interface import SystemSpecsHandler

logger = logging.getLogger(__name__)

class TestPipeline:
    """A class that defines the workflow for test input generation using the Flowcean framework.

    Attributes:
    ----------
    model_handler : ModelHandler
        Handles the Flowcean model and its predictions.
    
    model : Decision Tree | Black-box Model
        The underlying machine learning model extracted from the Flowcean model.
    
    dataset : pl.DataFrame
        The original training dataset.
    
    specs_handler : SystemSpecsHandler
        Extracts system specifications and feature information.
    
    requirements : dict
        Test requirements provided by the user.
    
    classification : bool
        Indicates whether the task is classification.
    
    eqclasses : list
        List of all equivalence classes.
    
    testplans : list
        List of all test plans (intervals used to sample test inputs).
    
    testinputs : list
        List of all generated test inputs.
    
    n_testinputs_lst : list
        Number of test inputs to generate per equivalence class.
    
    testinputs_df : pl.DataFrame
        Executable test inputs formatted for Flowcean.
    
    feature_names : list
        Names of all input features.
    
    hoeffding_tree : HoeffdingTreeRegressor
        Hoeffding tree used to approximate complex black-box models.
    
    Methods:
    -------
    execute()
        Executes the full test input generation workflow.

    save_hoeffding_tree()
        Saves the generated Hoeffding tree to a file.
    
    save_test_overview()
        Saves all intermediate results and outputs from the test input generation process.
    """

    eqclass_prio = []


    def __init__(
        self,
        model: Model,

        n_testinputs: int,
        test_coverage_criterium: str,
        dataset: pl.DataFrame | None = None,
        specs_file: Path | None = None,
        classification: bool = False,
        inverse_alloc: bool = False,
        epsilon: float = 0.5,
        seed: int = 42,

        performance_threshold: float = 0.3,
        sample_limit: int = 50000,
        n_predictions: int = 50,
        max_depth: int = 5,
        hoeffding_tree_extra_params: dict[str, Any] | None = None,
    ) -> None:
        """Initializes the TestPipeline.

        Args:
            model: The trained Flowcean model.
            n_testinputs: Total number of test inputs to generate.
            test_coverage_criterium: Strategy for test coverage ("bva" or "dtc").
            dataset (optional) : Original training dataset. Required if specs_file is not provided.
            specs_file (optional) : File containing system specifications. Required if dataset is not provided.
            classification (optional) : Whether the task is classification.
            inverse_alloc (optional) : If true, use inverse test allocation strategy.
            epsilon (optional) : Size of interval around boundaries for BVA testing.

            For Surrogate model training (only applicable for black-box models):
                performance_threshold (optional) : Minimum performance required to export the Hoeffding Tree (only applicable for black-box models).
                sample_limit (optional) : Maximum number of samples used to train the Hoeffding Tree (only applicable for black-box models).
                n_predictions (optional) : Number of correct predictions required before exporting the Hoeffding Tree (only applicable for black-box models).
                max_depth (optional) : Maximum depth of the Hoeffding tree.
                hoeffding_tree_extra_params (optional) : Additional parameters for training the Hoeffding Tree (only applicable for black-box models).
        """
        self.model_handler = ModelHandler(model)
        self.model = self.model_handler.get_ml_model()
        if test_coverage_criterium not in ["bva", "dtc"]:
            raise ValueError("Invalid test coverage criterium. Expected 'bva' or 'dtc'.")

        if (type(self.model) != DecisionTreeRegressor and
            type(self.model) != DecisionTreeClassifier and
            dataset is None):
            raise ValueError("Missing required parameter: 'dataset'")
        if dataset is None and specs_file is None:
            raise ValueError("Missing required parameter: 'dataset' or 'specs_file'")
        self.n_testinputs = n_testinputs
        self.test_coverage_criterium = test_coverage_criterium
        self.dataset = dataset
        self.specs_handler = SystemSpecsHandler(data = dataset, specs_file = specs_file)
        self.feature_names = self.specs_handler.extract_feature_names()
        self.hoeffding_tree = None
        self.classification = classification
        self.inverse_alloc = inverse_alloc
        self.seed = seed
        self.epsilon = epsilon
        self.performance_threshold = performance_threshold
        self.sample_limit = sample_limit
        self.n_predictions = n_predictions
        self.max_depth = max_depth
        self.hoeffding_tree_extra_params = hoeffding_tree_extra_params if hoeffding_tree_extra_params is not None else {}



    def _execute(self,
                test_coverage_criterium: str,
                n_testinputs: int,
                inverse_alloc: bool = False,
                epsilon: float = 0.5,
                performance_threshold: float = 0.3,
                sample_limit: int = 50000,
                n_predictions: int = 50,
                max_depth: int = 5,
                **kwargs) -> pl.DataFrame:
        """Executes the full test input generation workflow:

            1. If model is black-box: train Hoeffding Tree.
            2. Convert decision tree into framework-conformant structure.
            3. Compute equivalence classes.
            4. Generate test inputs.
            5. Compile test inputs into executable format.

        Args:
            test_coverage_criterium: 
                Strategy for test coverage (e.g., Boundary Value Analysis or Decision Tree Coverage).
            n_testinputs: 
                Total number of test inputs to generate.
            inverse_alloc (optional): 
                If True, generate more test inputs for less important equivalence classes.
            epsilon (optional): 
                Size of interval around boundaries for BVA testing.

            If self.model is a black-box model:
                performance_threshold (optional): 
                    Minimum performance required to export the Hoeffding Tree.
                sample_limit (optional): 
                    Maximum number of samples used to train the Hoeffding Tree.
                n_predictions (optional): 
                    Number of correct predictions required before exporting the model.
                **kwargs: 
                    Additional parameters for training the Hoeffding Tree.

        Returns:
            Executable test inputs formatted for Flowcean.
        """
        logger.debug(f"epsilon: {epsilon}")
        logger.debug(f"n_testinputs: {n_testinputs}")
        if not isinstance(self.model, (DecisionTreeRegressor, DecisionTreeClassifier)):
            # Generate Hoeffding tree only if it hasn't been created yet
            logger.info("Training Hoeffding Tree surrogate model for black-box model...")
            if self.hoeffding_tree is None:
                if self.dataset is None:
                    raise ValueError("Dataset is required to train the Hoeffding Tree surrogate model for black-box models.")
                htree_obj = HoeffdingTree(self.dataset, self.seed, self.model_handler, self.specs_handler)
                dtree = htree_obj.train_tree(performance_threshold=performance_threshold,
                                            sample_limit=sample_limit,
                                            n_predictions=n_predictions,
                                            max_depth=max_depth,
                                            classification=self.classification,
                                            **kwargs)
                self.hoeffding_tree = dtree
            else:
                dtree = self.hoeffding_tree
        else:
            # Use the decision tree directly if the model is a tree-based one
            logger.info("Using existing Decision Tree model for test input generation...")
            dtree = self.model.tree_

        # Extract specification details for equivalence class computation
        minmax_specs = self.specs_handler.extract_minmax_values()
        type_specs = self.specs_handler.extract_input_types()
        feature_names = self.specs_handler.extract_feature_names()
        n_features = self.specs_handler.get_n_features()

        # Compute equivalence classes from the decision tree
        test_tree = TestTree(dtree, self.specs_handler)
        eqclassobj = EquivalenceClassesHandler(test_tree, minmax_specs, n_features)
        self.eqclasses = eqclassobj.get_equivalence_classes()

        # Generate test inputs based on equivalence classes and coverage criteria
        testgenobj = TestGenerator(self.eqclasses, self.seed, type_specs)
        self.testinputs = testgenobj.generate_testinputs(test_coverage_criterium,
                                                       eqclassobj.eqclass_prio,
                                                       n_testinputs,
                                                       inverse_alloc=inverse_alloc,
                                                       epsilon=epsilon)
        self.testplans = testgenobj.testplans
        self.n_testinputs_lst = testgenobj.n_testinputs_lst

        # Compile test inputs into executable format for Flowcean
        testcompobj = TestCompiler(n_features, self.testinputs)
        self.testinputs_df = testcompobj.compute_executable_testinputs(feature_names)

        return self.testinputs_df


    def execute(self) -> pl.DataFrame:
        """Wrapper around `_execute()` to run the test input generation workflow with the parameters specified during initialization.

        Returns:
            Executable test inputs formatted for Flowcean.
        """
        return self._execute(
            test_coverage_criterium=self.test_coverage_criterium,
            n_testinputs=self.n_testinputs,
            inverse_alloc=self.inverse_alloc,
            epsilon=self.epsilon,
            performance_threshold=self.performance_threshold,
            sample_limit=self.sample_limit,
            n_predictions=self.n_predictions,
            max_depth=self.max_depth,
            **self.hoeffding_tree_extra_params,
        )


    # Print equivalence classes with test input counts to a text file
    def _print_eqclasses(self) -> None:
        columns = list(self.feature_names)
        columns.append("Number of test inputs")
        stringified_eqclasses = [[str(interval) for interval in eqclass] for eqclass in self.eqclasses]
        updated_eqclasses = [row + [self.n_testinputs_lst[i]] for i, row in enumerate(stringified_eqclasses)]
        eqclasses_table = tabulate(updated_eqclasses, headers=columns, tablefmt="grid")
        with open("equivalence_classes.txt", "w", encoding="utf-8") as f:
            f.write(eqclasses_table)


    # Print test plans to a text file
    def _print_testplans(self) -> None:
        testplans_table = tabulate(self.testplans, headers=self.feature_names, tablefmt="grid")
        with open("testplans.txt", "w", encoding="utf-8") as f:
            f.write(testplans_table)


    # Print executable test inputs to a text file
    def _print_testinputs(self) -> None:
        rows = self.testinputs_df.rows()
        testinputs_table = tabulate(rows, headers=self.feature_names, tablefmt="grid")
        with open("testinputs.txt", "w", encoding="utf-8") as f:
            f.write(testinputs_table)




    # Render and save the Hoeffding tree as a PNG image
    def _print_hoeffding_tree(self) -> None:
        if self.hoeffding_tree is not None:
            tree_img = self.hoeffding_tree.draw()
            tree_img.render("hoeffding_tree", format="png", cleanup=True)
        else:
            print("No Hoeffding Tree to print.")


    # Save the Hoeffding tree as a pickle file
    def save_hoeffding_tree(self, path) -> None:
        if self.hoeffding_tree is not None:
            with open(path+".pkl", "wb") as f:
                pickle.dump(self.hoeffding_tree, f)
        else:
            print("No Hoeffding Tree to save.")


    def save_test_overview(self, print_option: list = [1, 2, 3, 4]) -> None:
        """Generates and prints multiple report files containing:
            1. Equivalence classes + Number of test inputs
            2. Test plans
            3. Test inputs 
            4. Hoeffding Tree (if exists)

        Args:
            print_option : 
                List specifying which report files to print.
                - 1 → Equivalence classes + Number of test inputs
                - 2 → Test plans
                - 3 → Test inputs
                - 4 → Hoeffding tree
                Default: [1, 2, 3, 4]
        """
        logger.info(f"Printing: {print_option}")
        if 1 in print_option:
            self._print_eqclasses()
        if 2 in print_option:
            self._print_testplans()
        if 3 in print_option:
            self._print_testinputs()
        if 4 in print_option:
            self._print_hoeffding_tree()
