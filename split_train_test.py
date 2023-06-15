import pandas as pd

def split(csv_file):
    data = pd.read_csv(csv_file)

    # split data into training and test sets
    train = data.sample(frac=0.8, random_state=0)
    test = data.drop(train.index)

    train.to_csv("train.csv")
    test.to_csv("test.csv")

split("processed_data.csv")