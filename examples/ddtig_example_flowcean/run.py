
import logging
from pathlib import Path

import polars as pl

from flowcean.core import Model
from flowcean.core.strategies.offline import learn_offline
from flowcean.polars import DataFrame
from flowcean.testing import run_model_tests
from flowcean.testing.generator import DDTIGenerator
from flowcean.testing.predicates import PolarsPredicate
from flowcean.torch import LightningLearner, MultilayerPerceptron
from flowcean.utils import initialize_random

dirpath = Path(Path(__file__).resolve()).parent

# TODO (optional): Modify the CSV file name of the dataset used to generate a Flowcean model.
#                  Example datasets are located in the "examples/dataset" directory.
dataset = "dataset/regression/BodyFat.csv"

logging.basicConfig(level=logging.DEBUG)

def construct_data_driven_model() -> None:
    """Generates a data-driven MUT (Model Under Test) using Flowcean.

    Returns:
        A trained Flowcean model.
    """
    csv_file = dirpath / dataset
    df = pl.read_csv(csv_file)

    # Convert dataset into the format required to train the MUT
    data = DataFrame(df)
    inputs = df.columns[:-1]
    outputs = [df.columns[-1]]

    # Create a regression tree using Flowcean
    # TODO (optional): Adjust "max_depth" to control the maximum depth of the decision tree
    #learner = RegressionTree(max_depth=7)

    # TODO (optional): To use a neural network instead, uncomment the block below,
    #                  and comment out other model definitions.
    #                  Modify hyperparameters such as "learning_rate", "hidden_dimensions",
    #                  and "max_epochs" as needed.
    #                  Refer to the Flowcean documentation for details.
    learner = LightningLearner(
        module = MultilayerPerceptron(
            learning_rate=1e-3,
            output_size=len(outputs),
            hidden_dimensions=[10, 10],
        ),
        max_epochs=5,
    )

    # Train the model
    model = learn_offline(
        data,
        learner,
        inputs,
        outputs,
    )

    # Save model
    model_file = dirpath / "model.fml"
    with Path.open(model_file, "wb") as f:
        model.save(f)


def generate_test_inputs() -> None:
    """Generates test inputs from a trained Flowcean model.

    Returns:
        A Polars DataFrame containing test inputs.
    """
    model_file = dirpath / "model.fml"
    csv_file = dirpath / dataset
    df = pl.read_csv(csv_file)

    model = Model.load(model_file)

    # Initialize the test pipeline and generate test inputs based on the test requirements
    # TODO (optional): Set log=True to enable logging
    test_generator = DDTIGenerator(
        model,
        n_testinputs=1000,
        test_coverage_criterium="dtc",
        dataset=df,
        epsilon=1.0,
        max_depth=7,
    )
    test_generator.save_csv("test_inputs.csv")
    test_generator.reset()

    predicate = PolarsPredicate(
        (pl.col("BodyFat") < 40) & (pl.col("BodyFat") > 1),
    )
    #prediction = model.predict(test_generator.df.lazy()).collect()
    #print(prediction)
    run_model_tests(
        model,
        test_generator,
        predicate,
        show_progress=True,
        stop_after=40,
        path="test_failures.txt",
    )

    # TODO (optional): Uncomment to save all intermediate results and outputs to files
    # testpipeline.save_test_overview()
    # TODO (optional): Uncomment to save the Hoeffding tree to a file
    #                  Useful if the MUT is not a decision tree
    #test_generator.test_pipeline.save_hoeffding_tree("HoeffingTree")




def main() -> None:

    # TODO (optional): Comment if model is already created
    initialize_random(544382)
    construct_data_driven_model()
    generate_test_inputs()



if __name__ == "__main__":
    main()
