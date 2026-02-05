import polars as pl
from pathlib import Path
from typing import TextIO
from typing import BinaryIO
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from tabulate import tabulate
import yaml
import pickle
from flowcean.testing.generator.ddtig.user_interface import SystemSpecsHandler, RequirementsHandler
from flowcean.testing.generator.ddtig.application import ModelHandler
from flowcean.testing.generator.ddtig.domain import TestTree, TestGenerator, TestCompiler, EquivalenceClassesHandler, HoeffdingTree
from flowcean.testing.generator.ddtig.infrastructure import TestLogger

class TestPipeline():
    """
    A class that defines the workflow for test input generation using the Flowcean framework.

    Attributes
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
    
    logger : TestLogger
        Logger for tracking the test input generation process.
    
    hoeffding_tree : HoeffdingTreeRegressor
        Hoeffding tree used to approximate complex black-box models.
    
    Methods
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
        model_file: Path | BinaryIO,
        reqs_file: Path | TextIO,
        dataset: pl.DataFrame | None = None,
        specs_file: Path | TextIO | None = None,
        classification: bool = False,
        log: bool = False,
        log_file: Path | None = None,
    ) -> None:
        """
        Initializes the TestPipeline.

        Args:
            model_file : File containing the Flowcean model.
            reqs_file : File containing the test requirements.
            dataset (optional) : Original training dataset. Required if specs_file is not provided.
            specs_file (optional) : File containing system specifications. Required if dataset is not provided.
            classification (optional) : Whether the task is classification.
            log (optional) : Whether to enable logging.
            log_file (optional): Path to store the log file.
        """
        if log:
            if log_file is not None:
                self.logger = TestLogger(str(log_file))
            else:
                self.logger = TestLogger()
        else:
            self.logger = None
        self.model_handler = ModelHandler(model_file, self.logger)
        self.model = self.model_handler.get_ml_model()
        if (type(self.model) != DecisionTreeRegressor and 
            type(self.model) != DecisionTreeClassifier and 
            dataset is None):
            self.logger.log_error("Missing required parameter in TestPipeline.__init__(): 'dataset'.")
            raise ValueError("Missing required parameter: 'dataset'")
        if dataset is None and specs_file is None:
            self.logger.log_error("Missing required parameter in TestPipeline.__init__(): 'dataset' or 'specs_file'.")
            raise ValueError("Missing required parameter: 'dataset' or 'specs_file'")
        self.dataset = dataset 
        self.specs_handler = SystemSpecsHandler(data = dataset, specs_file = specs_file, logger = self.logger)
        reqs_handler = RequirementsHandler(reqs_file, logger = self.logger)
        self.requirements = reqs_handler.requirements
        self.feature_names = self.specs_handler.extract_feature_names()
        self.hoeffding_tree = None
        self.classification = classification


    def modify_reqs(self, reqs_file: Path | TextIO) -> None:
        """
        Updates the test requirements from a new requirements file.

        Args:
            reqs_file : File containing the updated test requirements.
        """
        reqs_handler = RequirementsHandler(reqs_file, logger = self.logger)
        self.requirements = reqs_handler.requirements
        self.logger.log_debug("Test requirements modified.")


    def _execute(self,
                test_coverage_criterium: str, 
                n_testinputs: int, 
                inverse_alloc: bool = False, 
                epsilon: float = 0.5,
                performance_threshold: float = 0.3,
                sample_limit: int = 50000,
                n_predictions: int = 50,
                **kwargs) -> pl.DataFrame:
        """
        Executes the full test input generation workflow:

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
        if not isinstance(self.model, (DecisionTreeRegressor, DecisionTreeClassifier)):
            # Generate Hoeffding tree only if it hasn't been created yet
            if self.hoeffding_tree is None:
                htree_obj = HoeffdingTree(self.dataset, self.model_handler, self.specs_handler, logger = self.logger)
                dtree = htree_obj.train_tree(performance_threshold=performance_threshold, 
                                            sample_limit=sample_limit, 
                                            n_predictions=n_predictions,
                                            classification=self.classification, 
                                            **kwargs)
                self.hoeffding_tree = dtree
            else:
                dtree = self.hoeffding_tree
        else:
            # Use the decision tree directly if the model is a tree-based one
            dtree = self.model.tree_

        # Extract specification details for equivalence class computation
        minmax_specs = self.specs_handler.extract_minmax_values()
        type_specs = self.specs_handler.extract_input_types()
        feature_names = self.specs_handler.extract_feature_names()
        n_features = self.specs_handler.get_n_features()
        
        # Compute equivalence classes from the decision tree
        test_tree = TestTree(dtree, self.specs_handler, logger = self.logger)
        eqclassobj = EquivalenceClassesHandler(test_tree, minmax_specs, n_features, logger = self.logger)
        self.eqclasses = eqclassobj.get_equivalence_classes()
        
        # Generate test inputs based on equivalence classes and coverage criteria
        testgenobj = TestGenerator(self.eqclasses, type_specs, logger = self.logger)
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
        """
        Wrapper around `_execute()` that uses the current test requirements.

        Returns:
            Executable test inputs formatted for Flowcean.
        """
        return self._execute(**self.requirements)
    

    # Print equivalence classes with test input counts to a text file
    def _print_eqclasses(self) -> None:
        columns = list(self.feature_names)
        columns.append("Number of test inputs")
        stringified_eqclasses = [[str(interval) for interval in eqclass] for eqclass in self.eqclasses]
        updated_eqclasses = [row + [self.n_testinputs_lst[i]] for i, row in enumerate(stringified_eqclasses)]
        eqclasses_table = tabulate(updated_eqclasses, headers=columns, tablefmt='grid')
        with open('equivalence_classes.txt', 'w', encoding='utf-8') as f:
            f.write(eqclasses_table)
    

    # Print test plans to a text file
    def _print_testplans(self) -> None:
        testplans_table = tabulate(self.testplans, headers=self.feature_names, tablefmt='grid')
        with open('testplans.txt', 'w', encoding='utf-8') as f:
            f.write(testplans_table)
    

    # Print executable test inputs to a text file
    def _print_testinputs(self) -> None:
        rows = self.testinputs_df.rows()
        testinputs_table = tabulate(rows, headers=self.feature_names, tablefmt='grid')
        with open('testinputs.txt', 'w', encoding='utf-8') as f:
            f.write(testinputs_table)
    

    # Print test requirements and hyperparameters to a YAML file
    def _print_test_requirements(self) -> None:
        # Requirements for test input generation with their default values
        reqs_default  = {"n_testinputs": 0, 
                         "test_coverage_criterium": "",
                         "epsilon": 0.5, 
                         "inverse_alloc": False}
        # Hyperparameters for training a Hoeffding Tree (+ default values)
        train_reqs_default  = {"performance_threshold": 0.3, "evaluation_metric": "MAE", 
                               "n_predictions": 50, "sample_limit": 50000, "max_depth": 5}
        reqs_to_print = {}
        test_reqs_to_print = {}

        # Get values for each parameters from user-specified requirements
        # or use default values if not specified
        for var, default in reqs_default.items():
            if var in self.requirements:
                test_reqs_to_print[var] = self.requirements[var]
            else:
                test_reqs_to_print[var] = default
        if self.requirements["test_coverage_criterium"] == "dtc":
            del test_reqs_to_print["epsilon"]
        reqs_to_print["Test Input Generation Parameters"] = test_reqs_to_print

        # Fill in Hoeffding tree parameters if applicable
        if not isinstance(self.model, (DecisionTreeRegressor, DecisionTreeClassifier)):
            train_reqs = {}
            for var, default in train_reqs_default.items():
                if var == "evaluation_metric" and self.classification:
                    train_reqs[var] = 'F1 Score'
                    continue
                if var in self.requirements:
                    train_reqs[var] = self.requirements[var]
                else:
                    train_reqs[var] = default
            reqs_to_print["Hoeffding Tree Hyperparameters"] = train_reqs

        # Write all requirements to a YAML file
        with open('test_requirements.yaml', 'w', encoding='utf-8') as file:
            yaml.dump(reqs_to_print, file, default_flow_style=False, sort_keys=False)


    # Render and save the Hoeffding tree as a PNG image
    def _print_hoeffding_tree(self) -> None:
        if self.hoeffding_tree is not None:
            tree_img = self.hoeffding_tree.draw()
            tree_img.render('hoeffding_tree', format='png', cleanup=True)
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
        """
        Generates and prints multiple report files containing:
            1. Equivalence classes + Number of test inputs
            2. Test plans
            3. Test inputs 
            4. Test requirements
            5. Hoeffding Tree (if exists)

        Args:
            print_option : 
                List specifying which report files to print.
                - 1 → Equivalence classes + Number of test inputs
                - 2 → Test plans
                - 3 → Test inputs
                - 4 → Test requirements
                - 5 → Hoeffding tree
                Default: [1, 2, 3, 4]
        """
        if self.logger:
            self.logger.log_debug(f"Printing: {print_option}")
        if 1 in print_option:
            self._print_eqclasses()
        if 2 in print_option:
            self._print_testplans()
        if 3 in print_option:
            self._print_testinputs()
        if 4 in print_option:
            self._print_test_requirements()
        if 5 in print_option:
            self._print_hoeffding_tree()