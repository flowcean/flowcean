# XOR Function

In this example, we demonstrate how to learn the XOR (exclusive OR) function using a regression tree model.
The resulting model is then used to predict the output of the XOR function for given binary inputs and is checked by automatic generated test cases.

## The XOR Function

The XOR function is a fundamental binary operation that outputs true only when the inputs differ and false otherwise.
Its truth table is given by

| Input A | Input B | Output (A XOR B) |
|---------|---------|------------------|
|    0    |    0    |        0         |
|    0    |    1    |        1         |
|    1    |    0    |        1         |
|    1    |    1    |        0         |

While simple, the XOR function is not linearly separable, making it a classic example for testing machine learning algorithms and a good candidate to demonstrate flowcean's prediction and testing capabilities.

## Learning a Regression Tree Model

There are many suitable algorithms to learn the XOR function.
For this example, we use a regression tree model, as it is simple, fast to train, and easy to interpret, while still being able to capture the non-linear relationships between the inputs and the output.

The required code to learn the model can be found in the following listing.

```python
# Load the truth table from a CSV file
data = DataFrame.from_csv("./data/data.csv")
inputs = [
    "x",
    "y",
]
outputs = ["z"]

# Setup a regression tree learner and learn the model
learner = RegressionTree()
model = learn_offline(
    data,
    learner,
    inputs,
    outputs,
)

# Save the learned model to a file
model.save(Path("xor_model.fml"))
```

First, the reference truth table is loaded from a CSV file and the input and output feature names are defined.
Then a regression tree learner is instantiated and the model is learned using the `learn_offline` function.
Finally, the learned model is saved to a file for later use.

As we have only 4 data points, the model is learned almost instantly and without any errors.
We therefore skip validating the model with a test set and metrics.

## Using the Model for Prediction

The learned -- and saved -- model can now be used to predict the output of the XOR function for given binary inputs.
Flowcean offers a simple way to connect a model with a cyber-physical system (CPS) in the loop through so-called `Adapters`.
`Adapters` are components that establish a connection between the model and the CPS.
This allows the model to receive live data for prediction purposes and send the results back to the CPS.

For this example, a simple mockup adapter is used to simulate the CPS, which reads samples from a flowcean environment and writes the predictions back to another file.
The code to set up the adapter and use the model for prediction can be found in the following listing.

```python
model = Model.load(Path("xor_model.fml"))
# Setup an adapter to read from a DataFrame environment and write results to a CSV file
adapter = DataFrameAdapter(
    DataFrame.from_csv("data.csv"),
    input_features=["x", "y"],
    result_path="result.csv",
)
# Run the prediction loop with the model and adapter
start_prediction_loop(
    model,
    adapter,
)
```

The `prediction_loop` runs until the adapter signals that there is no more data to process -- in this case, when all samples from the environment have been read.

After running the prediction loop and examining the `result.csv` file, we can see that the model correctly predicts the output of the XOR function for all input combinations.

## Testing the Model

Besides using the model for prediction, we can also automatically generate test cases to verify that the model -- and therefore the CPS -- behaves as expected.
Testing a model with flowcean is a two-step process.
First, a `TestcaseGenerator` is used to create a corpus of test cases based on the possible values of the input features.
Second, the generated test cases are executed and the results are checked against the expected output.
The expected outputs can either be concrete values or expressions that should evaluate to true for all test cases.

The behavior of the learned XOR function can be tested with the following code.

```python
# Setup a test case generator for all combinations of the binary inputs
test_generator = CombinationGenerator(
    Discrete("x", [0, 1]),
    Discrete("y", [0, 1]),
)

test_generator.save_excel("xor_test_cases.xlsx")
test_generator.reset()

# Define a predicate to check with the data
predicate = PolarsPredicate(
    pl.col("x").xor(pl.col("y")) == pl.col("z"),
)

# Test the model with the test data and predicate
test_model(
    model,
    test_generator,
    predicate,
    show_progress=True,
    stop_after=0,
)
```

First, a `CombinationGenerator` is used to create all possible combinations of the binary input features `x` and `y`.
Then, the generated test cases are saved to an Excel file for documentation purposes, and the generator is reset to start generating test cases from the beginning.
Next, a predicate is defined, which should hold true for all test cases.
In this case, the predicate checks that the output `z` is equal to the result of the -- logical and not learned -- XOR operation on the inputs `x` and `y`.
For other use cases, a more complex predicate could be used, such as ensuring that a certain output is within a specified range at all times.
Finally, the tests are executed using flowcean's `test_model` function.
Depending on the configuration, the testing can either stop at the first failed test case or continue until all test cases have been executed.

When running the tests for our learned XOR model, all test cases pass successfully, confirming that the model behaves as expected for all input combinations.
