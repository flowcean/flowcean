import polars as pl
import random
import os
import numpy as np
from flowcean.testing.generator.ddtig import SystemSpecsHandler

# Generate 5000 test inputs based on each baseline
n_tests         = 5000
n_tests_half    = 2500
n_tests_quarter = 1250


# Generate test inputs based on baseline method "Random Sampling"
def baseline_random(data: pl.DataFrame) -> pl.DataFrame:
    specs_handler = SystemSpecsHandler(data=data, specs_file=None, logger=None)
    samples = []
    for feature in specs_handler.specs['features']:
        lower = feature['min']
        upper = feature['max']
        if feature['type'] == 'float':
            samples.append([random.uniform(lower, upper) for _ in range(n_tests)])
        if feature['type'] == 'int':
            samples.append([random.randint(lower, upper) for _ in range(n_tests)])
    tests_tuples = list(zip(*samples))
    tests = pl.DataFrame(tests_tuples, schema=data.columns[:-1], orient='row')
    return tests

# Generate test inputs based on baseline method "Boundary Sampling"
def baseline_boundary(data: pl.DataFrame) -> pl.DataFrame:
    specs_handler = SystemSpecsHandler(data=data, specs_file=None)
    samples = []
    epsilon = 1.0
    for feature in specs_handler.specs['features']:
        lower = feature['min']
        upper = feature['max']
        if feature['type'] == 'float':
            samples_min = [random.uniform(lower-epsilon, lower+epsilon) for _ in range(n_tests_half)]
            samples_max = [random.uniform(upper-epsilon, upper+epsilon) for _ in range(n_tests_half)]
        if feature['type'] == 'int':
            samples_min = [random.randint(lower-epsilon, lower+epsilon) for _ in range(n_tests_half)]
            samples_max = [random.randint(upper-epsilon, upper+epsilon) for _ in range(n_tests_half)]
        samples_total = samples_min + samples_max
        random.shuffle(samples_total)
        samples.append(samples_total)
    tests_tuples = list(zip(*samples))
    tests = pl.DataFrame(tests_tuples, schema=data.columns[:-1], orient='row')
    return tests

# Generate test inputs based on baseline method "Quantile-Based Sampling"
def baseline_quantile(data: pl.DataFrame) -> pl.DataFrame:
    data = data.drop(data.columns[-1])
    df = data.to_pandas()
    sampled_dict = {}

    for col in df.columns:
        col_series = df[col].dropna()
        quantiles=4
        samples_per_bin=n_tests_quarter
        # Get quantile edges
        quantile_edges = col_series.quantile(q=np.linspace(0, 1, quantiles + 1)).values

        samples = []
        for i in range(len(quantile_edges) - 1):
            q_min = quantile_edges[i]
            q_max = quantile_edges[i + 1]

            # Sample uniformly within the quantile range
            if q_min == q_max:
                if data.schema[col] in [pl.Int8, pl.Int16, pl.Int32, pl.Int64]:
                    sampled = np.full(samples_per_bin, q_min).astype(int)
                else:
                    sampled = np.full(samples_per_bin, q_min).astype(float)
            else:
                if data.schema[col] in [pl.Int8, pl.Int16, pl.Int32, pl.Int64]:
                    sampled = np.random.randint(low=q_min, high=q_max, size=samples_per_bin)
                else:
                    sampled = np.random.uniform(low=q_min, high=q_max, size=samples_per_bin)

            samples.extend(sampled)

        # Convert to Polars Series
        sampled_dict[col] = pl.Series(name=col, values=samples)

    return pl.DataFrame(sampled_dict)

# Generate test inputs based on baseline method "Example-Based Sampling"
def baseline_original(data: pl.DataFrame) -> pl.DataFrame:
    data = data.drop(data.columns[-1])
    n_rows = data.height
    replace = n_rows < n_tests
    tests = data.sample(n=n_tests, with_replacement=replace, seed=42)
    return tests


def main() -> None:

    # TODO (optional): Modify the CSV file name of the dataset for test input generation.
    #                  Example datasets are located in the "examples/dataset" directory.
    dataset = "dataset/regression/BodyFat.csv"
    dirpath = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(dirpath, dataset)
    data = pl.read_csv(csv_file)

    tests_RS  = baseline_random(data)
    tests_BS  = baseline_boundary(data)
    tests_QBS = baseline_quantile(data)
    tests_EBS = baseline_original(data)

    # TODO (optional): Uncomment to store generated test inputs in a file
    filename = dataset.replace("dataset/regression/", "").replace(".csv", "")
    # tests_RS.write_csv(filename +"_RS.csv")
    # tests_BS.write_csv(filename + "_BS.csv")
    # tests_QBS.write_csv(filename + "_QBS.csv")
    # tests_EBS.write_csv(filename + "_EBS.csv")

if __name__ == "__main__":
    main()