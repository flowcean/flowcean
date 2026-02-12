
from flowcean.core import Model
from flowcean.core.strategies.offline import learn_offline
from flowcean.polars import DataFrame
from flowcean.sklearn import RegressionTree
import polars as pl
from flowcean.torch import LightningLearner, MultilayerPerceptron
from flowcean.testing.generator.ddtig import TestPipeline, ModelHandler
import os

dirpath = os.path.dirname(os.path.abspath(__file__))

# TODO (optional): Modify the CSV file name of the dataset used to generate a Flowcean model.
#                  Example datasets are located in the "examples/dataset" directory.
dataset = "dataset/regression/BodyFat.csv"

# TODO (optional): Adjust the parameters in this file to define the expected test requirements.
test_reqs_file = "test_reqs.json"

def construct_data_driven_model():
    '''
    Generates a data-driven MUT (Model Under Test) using Flowcean.

    Returns:
        A trained Flowcean model.
    '''
    
    csv_file = os.path.join(dirpath, dataset)
    df = pl.read_csv(csv_file)

    # Convert dataset into the format required to train the MUT
    data = DataFrame(df)
    inputs = df.columns[:-1]
    outputs = [df.columns[-1]]

    # Create a regression tree using Flowcean
    # TODO (optional): Adjust "max_depth" to control the maximum depth of the decision tree
    learner = RegressionTree(max_depth=7)

    # TODO (optional): To use a neural network instead, uncomment the block below,
    #                  and comment out other model definitions.
    #                  Modify hyperparameters such as "learning_rate", "hidden_dimensions", 
    #                  and "max_epochs" as needed.
    #                  Refer to the Flowcean documentation for details.
    # learner = LightningLearner(
    #     module = MultilayerPerceptron(
    #         learning_rate=1e-3,
    #         input_size=len(inputs),
    #         output_size=len(outputs),
    #         hidden_dimensions=[10, 10],
    #     ),
    #     max_epochs=5,
    # )

    # Train the model
    model = learn_offline(
        data,
        learner,
        inputs,
        outputs,
    )

    # Save model
    model_file = os.path.join(dirpath, "model.fml")
    with open(model_file, "wb") as f:
        model.save(f)


def generate_test_inputs() -> pl.DataFrame:
    '''
    Generates test inputs from a trained Flowcean model.

    Returns:
        A Polars DataFrame containing test inputs.
    '''
    model_file = os.path.join(dirpath, "model.fml")
    csv_file = os.path.join(dirpath, dataset)
    df = pl.read_csv(csv_file)

    test_reqs = os.path.join(dirpath, test_reqs_file)

    # Initialize the test pipeline and generate test inputs based on the test requirements
    # TODO (optional): Set log=True to enable logging
    testpipeline = TestPipeline(model_file, test_reqs, dataset=df)
    test_inputs = testpipeline.execute()

    # TODO (optional): Uncomment to save all intermediate results and outputs to files
    # testpipeline.save_test_overview()
    # TODO (optional): Uncomment to save the Hoeffding tree to a file
    #                  Useful if the MUT is not a decision tree
    # testpipeline.save_hoeffding_tree("PATH_TO_SAVE/FILE_NAME")
    
    return test_inputs


def execute_test_set(test_set: pl.DataFrame) -> pl.DataFrame:
    '''
    Executes a test set on a Flowcean model.

    Args:
        test_set: The test set to execute as Polars DataFrame

    Returns:
        A Polars DataFrame containing test inputs and their predicted outputs.
    '''
    model_file = os.path.join(dirpath, "model.fml")

    # Predict outputs using the Flowcean model
    y_preds = ModelHandler(model_file).get_model_prediction_as_lst(test_set)

    # Append predictions as the last column
    test_results = test_set.with_columns([
        pl.Series("preds", y_preds)
    ])
    
    # TODO (optional): Uncomment to save the test results to a CSV file
    test_results.write_csv(os.path.join(dirpath, "test_results.csv"))
    
    return test_results


def main() -> None:

    # TODO (optional): Comment if model is already created
    construct_data_driven_model()
    
    test_set = generate_test_inputs()
    results = execute_test_set(test_set)
    print(results)

if __name__ == "__main__":
    main()

        