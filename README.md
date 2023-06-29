# ToyExample_PassiveDataLearning

## Installation

The _one-script-to-run-everything_-file is the `run.py`.
To be able to use it, you should first setup a virtual python environment:

```bash
python -m venv venv
# On Linux:
. ./venv/bin/activate
# Or for windows/powershell:
.\venv\Scripts\activate.ps1
```

Afterwards, you can install the package with

```bash
pip install -e .
```

and finally start the script with

```bash
python run.py
```

Currently, it should run the training and exit with an output of metrics evaluated on the test data.

## Downloaded Data

The Accelerometer Dataset (00611) is found at the UCI Machine Learning Repository <https://archive.ics.uci.edu/ml/datasets/Accelerometer>.
The new UCI Repository seems to miss the dataset.
It can still be found at [web-archive](http://web.archive.org/web/20230130144205/https://archive.ics.uci.edu/ml/machine-learning-databases/00611/accelerometer.csv).
The respective paper is published at MDPI: [Prediction of Motor Failure Time Using An Artificial Neural Network](https://www.mdpi.com/1424-8220/19/19/4342).

Download the Dataset and save the respective csv-file at `data/accelerometer/download.csv`.

The tabular dataset consists of time series data in the following format:

| wconfid | pctid | x      | y      | z      |
| ------- | ----- | ------ | ------ | ------ |
| 1       | 20    | 21.004 | 0.090  | -0.12  |
| 1       | 20    | 1.004  | -0.043 | -0.125 |
| ...     | ...   | ...    | ...    | ...    |

## Preprocess to Generate Labels

The labels of an estimated failure time are generated using formulas presented in the paper.
The `accelerometer_preprocessing.ipynb` notebook can be used to reproduce the steps and generate a labeled dataset in the following format:

| x-Amplitude | x-Frequency | y-Amplitude | y-Frequency | z-Amplitude | z-Frequency | Growth-rate | Estimated-Failure-Time |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ---------------------- |
| 0.27        | 0.28        | 0.03        | 0.28        | 0.06        | 0.28        | 0.05        | 1093.65                |
| ...         | ...         | ...         | ...         | ...         | ...         | ...         | ...                    |

The resulting file is stored at `data/accelerometer/processed_data.csv`.

## Create a Train and Test Split

Run the the script to generate a train and test split.

```sh
./split_train_test.py data/accelerometer/processed_data.csv
```

The dataset is split in 80%/20% fashion and the respective files are placed at `data/accelerometer/train.csv` and `data/accelerometer/test.csv`.
