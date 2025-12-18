# Automatic Lashing Platform - Container Weight Estimation Example

The automatic lashing platform (ALP) is a harbour system which automates the handling of twist locks on the land side.
Twist locks are used to secure container stacks during sea transport and need to be removed when containers are unloaded at the port.
During unloading, containers are placed by the Ship-To-Shore crane (STS) on the ALP, which uses hydraulic actuatores to remove the twist locks automatically and stores them in an internal magazine.
When loading containers onto a ship, the ALP retrieves the twist locks from the magazine and installs them on the containers before the STS picks them up.

The ALP does not rely on an external energy source; instead, it uses the potential energy of the container when it is placed on the platform.
Four hydraulic cylinders convert this potential energy into hydraulic pressure, which is then stored in an internal accumulator.
This pressure powers the hydraulic actuators and a generator, which provides electrical energy for the ALP's control system.
A simplified schematic of the ALP's energy harvesting system is shown below.

![Schematical display of the ALP's hydraulic system](./images/ALP_Hydraulic_Schema.svg)

The main valves enable or disable individual cylinders for energy harvesting, depending on the weight of the container and the desired cycle time.
More energy can be generated if more cylinders are active, but the required container weight to fully compress the cylinders also increases.
This has the potential to stall the operation if the container is too light.
Conversely, if fewer cylinders are active, the required weight decreases, but so does the energy obtained.

## Problem

For a variety of reasons and purposes, it is important to know the weight of containers being loaded onto or unloaded from a ship.
One such reason is to calculate the centre of gravity of the ship to ensure stability.
If the centre of gravity is incorrect, it can negatively affect the ship's hydrostatic and dynamic behaviour, potentially leading to dangerous situations.
Another example is illegal activities such as smuggling, where the actual load and weight of a container differ from the values declared on the load sheet.
The weight is also of interest to the ALP itself.
Depending on the weight of the container, the energy harvesting system can be adjusted to maximise energy output in the shortest possible time without disrupting the loading or unloading process.

This makes independently obtaining and verifying the weight of a container an important step to ensure a safe, secure and efficient operation.
The goal of this example is, to use the data obtained from the ALP's sensors to estimate the weight of a single container.

## Data

Multiple sensors are installed on the ALP to monitor its state during container handling operations.
Pressure sensors are placed at the outlet of each of the four hydraulic cylinders (`p_cylinder1` to `p_cylinder4`) and at the accumulator (`p_accumulator`).
These sensors measure the hydraulic pressure in whole millibar.
A distance sensor measures the height of the topframe (`frame_z`) in whole millimetres, where zero corresponds to a completely lowered topframe.
All time-dependent sensor readings are sampled at a frequency of 100 Hz over a time span of 15 seconds and are stored in the Flowcean time series format.

In addition to the time-dependent sensor readings, a set of static parameters are available for each container handling operation.
These include the ambient temperature (`T`) in whole millidegrees Celsius, the state of each individual valve (`valve_state_0` to `valve_state_3`) where `0` means the corresponding cylinder does not participate in energy harvesting and `1` means it does and the total number of active valves for energy harvesting (`active_valve_count`).
The actual container weight `container_weight` in whole kilograms is also included as the ground truth.

There are four different datasets available for this example.
The primary dataset contains approximately 160,000 simulated container handling operations, involving containers with weights ranging from 2,500 kg to 60,000 kg.
The other three datasets are smaller, containing between 3,100 and 4,000 samples each.
In these datasets, only one of the three parameters — container weight (`container_weight`), ambient temperature (`T`), and starting pressure in the accumulator (`p_accumulator[0]`) — is varied, while all other parameters are held constant.
The datasets used for this example can be found on [Zenodo](https://zenodo.org/records/17812362).

## Learning Models

Different machine learning techniques can be applied to estimate the container weight based on the available sensor data.
One potential approach is to use a gradient boosted regression model, to predict the container weight based on the accumulator pressure (`p_accumulator`), the ambient temperature `T` and the number of active valves (`active_valve_count`).

First the data is loaded and preprocessed to extract the relevant features and targets.

```python
from flowcean.polars import (DataFrame, Filter, Flatten, Lambda, Resample, Select, TimeWindow)

data = DataFrame.from_parquet("path/to/dataset.parquet")
  | Lambda(lambda df: df.limit(50_000)) # Use a subset of the data for faster training
  | Select([
      "p_accumulator",
      "T",
      "active_valve_count",
      "container_weight"
  ])                                    # Select the relevant source and target features
  | Filter("active_valve_count > 0")    # Filter out samples where no valves are active
  | Resample(0.25)                      # Downsample time series features to 4 Hz to reduce dimensionality
  | TimeWindow(
    time_start=0,
    time_end=6,
  )                                     # Only use the first 6 seconds of each sample
  | Flatten()                           # Flatten the time series features into individual columns

```

Then split the data into a test and a training set.

```python
from flowcean.polars import TrainTestSplit

train_env, test_env = TrainTestSplit(
    ratio=0.8,
    shuffle=True,
).split(data)
```

Next, setup a gradient boosted regression learner and define the input and target features.

```python
from flowcean.xgboost import XGBoostRegressorLearner

learner = XGBoostRegressorLearner(
    n_estimators=10,              # Use 10 boosting rounds
    max_depth=20,                 # Allow for deep trees to capture complex relationships
    objective="reg:squarederror", # Aim for minimizing the mean squared error
)

inputs = [
    "^p_accumulator_.*$",  # All time series features of the accumulator pressure
    "active_valve_count",  # Number of active valves
    "T",                   # Ambient temperature
]

outputs = [
    "container_weight",    # Target feature: container weight
]
```

Finally, train the model using the learner we've just defined and the `learn_offline` strategy.

```python
from flowcean.core import learn_offline

model = learn_offline(
    train_env,
    learner,
    inputs,
    outputs,
)

```

Once the model is trained, it can be evaluated on the test set to assess its performance.

```python
from flowcean.core import evaluate_offline
from flowcean.sklearn import MeanAbsoluteError, MeanSquaredError

report = evaluate_offline(
    model,
    test_env,
    inputs,
    outputs,
    [MeanAbsoluteError(), MeanSquaredError()],
)

print(report)
```
