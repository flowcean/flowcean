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

## Learning a Model

Data used in the AGenC framework needs to be supplied with additional metadata.
The metadata of this example is located in `data/metadata.yaml`.
The experiment is configured in `experiment.yaml`.
This configuration specifies the respective steps in the framework, e.g. transformations and learning algorithms applied to train a model.
Additionally, it references the `metadata.yaml` to supply the experiment with a dataset.
Use the AGenC commandline interface to run the experiment.

```bash
agenc --experiment experiment.yaml
```

```{spelling:word-list}
Sampaio
wconfid
pctid
```
