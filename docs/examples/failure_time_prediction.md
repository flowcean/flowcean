# Failure Time Prediction

This example trains a model to predict the failure time of a system.
The failure time allows to schedule maintenance to repair or replace system before its downtime.

The example is based on the research of Sampaio.
The respective paper is published at MDPI: [Prediction of Motor Failure Time Using An Artificial Neural Network](https://www.mdpi.com/1424-8220/19/19/4342).
The Accelerometer Dataset (00611) is found at the UCI Machine Learning Repository <https://archive.ics.uci.edu/ml/datasets/Accelerometer>.

## Downloaded Data

Download the Dataset and save the respective csv-file at `data/download.csv` of the example's root directory.
The tabular dataset consists of time series data in the following format:

| wconfid | pctid | x      | y      | z      |
| ------- | ----- | ------ | ------ | ------ |
| 1       | 20    | 21.004 | 0.090  | -0.12  |
| 1       | 20    | 1.004  | -0.043 | -0.125 |
| ...     | ...   | ...    | ...    | ...    |

## Preprocess to Generate Labels

The labels of an estimated failure time are generated using formulas presented in the paper.
The `preprocessing.ipynb` notebook is used to reproduce the steps and generate a labeled dataset in the following format:

| x-Amplitude | x-Frequency | y-Amplitude | y-Frequency | z-Amplitude | z-Frequency | Growth-rate | Estimated-Failure-Time |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ---------------------- |
| 0.27        | 0.28        | 0.03        | 0.28        | 0.06        | 0.28        | 0.05        | 1093.65                |
| ...         | ...         | ...         | ...         | ...         | ...         | ...         | ...                    |

The resulting file is stored at `data/processed_data.csv`.

## Run this example

To run this example first make sure you followed the [installation instructions](../getting_started/preparation.md) to setup python and git.
Afterwards you can either use `hatch` or run the examples from source.

### Hatch

The easiest way to run this example is using `hatch`.
Follow the [installation guide](../getting_started/installation.md) to clone flowcean but stop before installing it or any of its dependencies.
Now you can run the example using

```sh
hatch run examples:failure_time_prediction
```

This command will take care of installing any required dependencies in a separate environment.
After a short moment you should see the learning results and the achieved metric values.

### From source

Follow the [installation guide](../getting_started/installation.md) to install flowcean and it's dependencies from source.
Afterwards you can navigate to the `examples` folder and run the examples.

```sh
cd examples/failure_time_prediction
python run.py
```