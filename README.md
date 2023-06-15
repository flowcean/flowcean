# ToyExample_PassiveDataLearning

## Downloaded Data

The Accelerometer Dataset (00611) is found at the UCI Machine Learning Repository <https://archive.ics.uci.edu/ml/datasets/Accelerometer> ([web-archive](http://web.archive.org/web/20230130144205/https://archive.ics.uci.edu/ml/machine-learning-databases/00611/accelerometer.csv)]
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

The dataset is split in 80%/20% fashioln and the respective files are placed at `data/accelerometer/train.csv` and `data/accelerometer/test.csv`.

### Start the Training Process

Start the training process of the [pytorch-lightning](https://lightning.ai/) neural network learner to train a neural network.

```sh
python learner.py
```

The learner expects the `training.csv` and `test.csv` data to reside in `data/accelerometer/`.
