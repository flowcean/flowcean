---
icon: lucide/brain-circuit
---

# Model

Models and learners are separated in Flowcean.

A **learner** trains a model.
The learner is a process that learns patterns from data.
A learner can be based on different machine learning algorithms with various configurations.
During the learning process the learner updates its internal parameters or structure in order to optimize its task.
Which learners are implemented in Flowcean can be found [here](../reference/index.md#learners-and-models).

Once the learner is done training, it outputs a **model**.
The model represents the learned patterns that the learner has extracted from the data.
The model can be used to make predictions on new, unseen data.
Models can be saved and loaded.
For more information take a look at the [API](../reference/core.md#flowcean.core.Model).

## Implementing a Model

Subclass `Model[Input, Output]` and implement `_predict(input_features: Input) -> Output`.
The inherited `predict` and call operator preserve these types while applying preprocessing and postprocessing.
Attached transforms must preserve the corresponding input or output type.
Learning strategies attach transforms to the returned model, so these models must allow mutation.

For tabular models, subclass `flowcean.polars.PolarsModel` and implement `_predict` for lazy frames.
Its public prediction methods accept both eager and lazy frames, normalizing inputs before preprocessing, and return a lazy frame.
Consumers that only need prediction can accept the structural `Predictor[Input, Output]` contract instead of requiring model inheritance.

Load trusted saved models through the expected concrete class, such as `SciKitModel.load(path)`, to retain its public types.
Loading checks the class, not a dataframe schema or generic type arguments.
Pickle files can execute code: never load an untrusted model.

## Classifier Predictions

Scikit-learn and XGBoost classifier wrappers use the estimator's native predictions by default (`threshold=None`).
This preserves native class labels and supports multiclass prediction within the estimator's own restrictions.

An explicit `threshold` is supported only for a single binary target.
It must be finite and between zero and one.
A probability greater than or equal to the threshold selects `classes_[1]`; otherwise prediction selects `classes_[0]`.
The wrappers do not replace those labels with zero and one.

Flowcean's `predict_proba` has a narrower contract than the estimator method: it returns one target-named column containing the probability of `classes_[1]` for a binary target.
It rejects multiclass and multioutput probability requests.
Preprocessing is applied, but label postprocessing is not.

`tune_threshold` returns the best threshold and its scores, restoring the model's previous threshold even if evaluation fails.
Assign the returned threshold explicitly when you want to use it for subsequent predictions.
