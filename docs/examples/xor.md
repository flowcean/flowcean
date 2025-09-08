# XOR

In this example, we demonstrate how to learn the XOR function using a regression tree model.
The resulting model is then used to predict the output of the XOR function for given binary inputs and is checked by automatic generated test cases.

## The XOR Function

The XOR (exclusive OR) function is a fundamental binary operation that outputs true only when the inputs differ.
It's truth table is

| Input A | Input B | Output (A XOR B) |
|---------|---------|------------------|
|    0    |    0    |        0         |
|    0    |    1    |        1         |
|    1    |    0    |        1         |
|    1    |    1    |        0         |

While simple, the XOR function is not linearly separable, making it a classic example for testing machine learning algorithms and a good candidate to demonstrate flowceans prediction and testing capabilities.

## Learning a Regression Tree Model

There are many suitable algorithms to learn the XOR function.
For this example, we use a regression tree model as it is simple, fast to train and easy to interpret while still being able to capture the non-linear relationships of the inputs and output.

The required code to learn the model can be found in the following listing.

```python
data = DataFrame.from_csv("./data/data.csv")
inputs = [
    "x",
    "y",
]
outputs = ["z"]

learner = RegressionTree()
model = learn_offline(
    data,
    learner,
    inputs,
    outputs,
)

model.save(Path("xor_model.fml"))
```

First the reference truth table is loaded from a CSV file and the input and output feature names are defined.
Than a regression tree learner is instantiated and the model is learned using the `learn_offline` function.
Finally, the learned model is saved to a file for later use.

As we have only 4 data points, the model is learned almost instantly and without any errors.
We therefore skip validating the model with a test set and metrics.

## Using the Model for Prediction

The learned -- and saved -- model can now be used to predict the output of the XOR function for given binary inputs.

!todo: Hier was zum Thema Adapter und Anbindung von CPS oder so

## Testing the Model

Besides using the model for prediction, we can also automatically generate test cases to verify that the model -- and therefore the CPS -- behaves as expected.
