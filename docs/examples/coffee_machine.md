# Coffee Machine with AALpy

This example learns a Mealy machine from Coffee Machine traces using
[AALpy's passive RPNI algorithm](https://github.com/DES-Lab/AALpy/wiki/RPNI---Passive-Deterministic-Automata-Learning).
Training and prediction run locally in Python.

## Run the example

From the repository root, retrieve the DVC-managed CSV traces, then run:

```sh
uv run dvc pull --recursive examples/coffee_machine
just examples-coffee_machine
```

Access to the configured DVC remote is required. The example keeps each CSV as
one trace, uses 80% of the traces for training, and reports the fraction of test
traces whose entire output word is predicted correctly.

## Trace contract

`flowcean.aalpy.RPNIMealyLearner` works with `learn_offline` and returns a separate
`RPNIMealyModel`. Select exactly one input column and one output column. Each row
contains a full trace, represented as either:

- a list of scalar symbols (strings, booleans, integers, or finite floats); or
- a Flowcean time series: a list of structs with exactly numeric `time` and
  scalar `value` fields, as produced by `ToTimeSeries("t")` in this example.

Timestamped sequences are sorted independently by time before pairing inputs
and outputs by position. Equal timestamps retain their original order. Timing
is not learned. Null traces, timestamps, and symbols are rejected.

Mealy traces require equal-length input and output words. For Moore machines,
use `RPNIMooreLearner`: the output word must have one additional symbol at the
start, representing the initial state's output. Contradictory labels for the
same input prefix are rejected.

Both model types return one list of output symbols per input trace, retaining
the training output column's name and symbol dtype. Moore predictions include
the initial output, even for an empty input. Predictions do not fabricate
output timestamps; the example's metric compares symbol words directly, without
a model post-transform.

Undefined transitions raise `ValueError`. The learners leave AALpy's automata
input-incomplete rather than adding synthetic completion symbols that could
change the output dtype. Predictions beyond the observed traces may generalize
differently from other passive learners.

The read-only `model.automaton` property exposes the underlying AALpy automaton
for inspection. Treat its graph as read-only while predicting. Prediction starts
at the initial state for each trace and does not change AALpy's `current_state`.
Models support Flowcean's `model.save(...)` and `Model.load(...)` persistence;
only load trusted pickle files.
