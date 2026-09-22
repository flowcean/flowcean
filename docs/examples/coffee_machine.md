# Coffee Machine with AALpy

This example learns a Mealy machine from recorded Coffee Machine traces using
[AALpy's passive RPNI algorithm](https://github.com/DES-Lab/AALpy/wiki/RPNI---Passive-Deterministic-Automata-Learning).
Training and prediction run locally in Python.

## Run the example

From the repository root, retrieve the DVC-managed CSV traces and run the
example:

```sh
uv run dvc pull --recursive examples/coffee_machine
just examples-coffee_machine
```

Each CSV is one synchronized trace. The example sorts its rows by time once and
collects the aligned `input` and `output` columns into symbol words. It uses 80%
of the traces for training and reports the fraction of test traces whose entire
output word is predicted correctly.
