# One-Tank Incremental Learning Example

This example demonstrates how to train an incremental learning model to predict the time-series behavior of a single water tank with a height-dependent outflow and a time-varying inflow. It is inspired by the one-tank Simulink model used for control design [^1]. A schematic drawing of the system is shown in the following figure.

![The one-tank system](./images/one_tank.svg)

The complete source code for this example can be found [in the repository](https://github.com/flowcean/flowcean/blob/main/examples/one_tank/run_incremental.py). See also [Run this example](#run-this-example) on how to run this example locally.

The dynamics of the system follow an ordinary differential equation (ODE). ODEs describe the derivative of a variable (e.g., its change over time) as a function of the variable itself.

For this example, the system is described by the equation

$$ \dot{x} = \frac{b V(t) - a \sqrt{x}}{A} $$

where $x$ is the water level in the tank, $\dot{x}$ is the change of the water level over time, $V(t)$ is the time-dependent inflow, $A$ is the tank area, and $a$ and $b$ are scaling constants for the equation. The solution of an ODE is not a single value, but a function (here $x(t)$) or a series of its values for different times $t$. As solving this ODE analytically is quite complicated, a numerical solver is used which computes solution points starting from an initial value. In this example, the initial value is the initial level of the liquid $x(0) = x_0$ in the tank.

The free parameters from the above equation are set to

| $A$ | $b$ | $a$   |
| --- | --- | ----- |
| $5$ | $2$ | $0.5$ |

The inflow is given by $V(t) = \mathrm{max}\left(0, \sin\left( 2 \pi \frac{1}{10} t \right)\right)$ and the initial condition is $x_0 = 1$. Using a suitable numerical solution algorithm, the equation can be solved for the level $x_n$. Since the solution is not continuous, the level is not a function of time, but a discrete function of the sample number $n$. The corresponding time can be calculated by multiplying the sample number $n$ by the step size $T$ between two samples. The graph below shows the development of the water level $x$ from zero to ten seconds.

![Differential equation solution plotted over time](./images/one_tank_graph.svg)

## Incremental Learning Model

After setting up the simulation, we want to use an incremental learner to predict the level of the tank $x[n]$ given the current input $V[n]$ and the level and input in the previous two time steps. The unknown function we are looking for and that we want to learn is

$$ x*n = f\left(V_n, x*{n-1}, V*{n-1}, x*{n-2}, V\_{n-2}\right). $$

To do this, we first need data to learn the function's representation in Flowcean. Normally this data would be recorded from a real CPS and imported into the framework as a CSV, ROS bag, or something similar. However, since we know the differential equation describing the system behavior, we can also use this equation to generate data. We can do this by using an [`ODEEnvironment`](../reference/flowcean/ode/index.md#flowcean.ode.OdeEnvironment) to model the ODE as an [`IncrementalEnvironment`](../reference/flowcean/core/index.md#flowcean.core.IncrementalEnvironment) within the framework.

To do so, a special `OneTank` class is created which inherits from the general Flowcean `OdeSystem` class.

```python
class OneTank(OdeSystem[TankState]):
    def __init__(
        self,
        *,
        area: float,
        outflow_rate: float,
        inflow_rate: float,
        initial_t: float = 0,
        initial_state: TankState,
    ) -> None:
        super().__init__(
            initial_t,
            initial_state,
        )
        self.area = area
        self.outflow_rate = outflow_rate
        self.inflow_rate = inflow_rate

    @override
    def flow(
        self,
        t: float,
        state: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        pump_voltage = np.max([0, np.sin(2 * np.pi * 1 / 10 * t)])
        tank = TankState.from_numpy(state)
        d_level = (
            self.inflow_rate * pump_voltage
            - self.outflow_rate * np.sqrt(tank.water_level)
        ) / self.area
        return np.array([d_level])
```

The `OneTank` class describes the differential equation in the `flow` method as well as all parameters needed to evaluate it. The type parameter `TankState` is used to map the general numpy array holding the current state of the simulation to a more tangible representation. The state of an `ODESystem` can also be used for systems with multiple states, where the behavior might change when certain conditions are met. For this example with only a single state, the `TankState` class simply maps the water level in the tank to the first entry in the state vector.

```python
class TankState(State):
    water_level: float

    @override
    def as_numpy(self) -> NDArray[np.float64]:
        return np.array([self.water_level])

    @classmethod
    @override
    def from_numpy(cls, state: NDArray[np.float64]) -> Self:
        return cls(state[0])
```

The `ODESystem` can now be constructed by creating an instance of the `OneTank` class with the parameters given above.

```python
    system = OneTank(
        area=5,
        outflow_rate=0.5,
        inflow_rate=2,
        initial_state=TankState(water_level=1),
    )
```

From the system, an `OdeEnvironment` can be constructed.

```python
    data_incremental = OdeEnvironment(
        system,
        dt=0.1,
        map_to_dataframe=lambda ts, xs: pl.DataFrame(
            {
                "t": ts,
                "h": [x.water_level for x in xs],
            },
        ),
    )
```

Beside the ODE, the time resolution `dt` and the mapping function `map_to_dataframe` are passed to the constructor. The mapping function describes how the generated solutions for different points in time can be mapped into a data frame for further processing within Flowcean.

The generated output of the `OdeEnvironment` environment has the form

| $x$     | $V$     |
| ------- | ------- |
| $x[0]$  | $V[0]$  |
| $x[1]$  | $V[1]$  |
| $\dots$ | $\dots$ |
| $x[N]$  | $V[N]$  |

Until now, the data is in a time series format with each row representing a sampled value at the step $n$. However, for our prediction of the current fill level $x[n]$, as described by the equation above, we need the current input $V[n]$ and the values of the two previous time steps as a single sample. To achieve this, we use a [`SlidingWindow`](../reference/flowcean/polars/index.md#flowcean.polars.SlidingWindow) transform. See the linked documentation for a more detailed explanation of how the transform works.

```python
window_transform = SlidingWindow(window_size=3)
```

Now that the data is in the correct format, it can be split into a test set with 80% of the samples and a training set with the remaining 20%. This is done by using a [`TrainTestSplit`] operation and helps with evaluating the learned model's performance after training. To make the learning less biased, the samples are shuffled before splitting.

```python
train, test = TrainTestSplit(ratio=0.8, shuffle=False).split(data_incremental.collect(250).with_transform(window_transform))
```

With the training data generated, fully transformed, and split, it's time to use a learning algorithm to learn the prediction function from the beginning of this section. We use a linear regression model for incremental learning.

```python
learner = LinearRegression(
    input_size=2,
    output_size=1,
    learning_rate=0.01,
)
```

The `inputs` and `outputs` variables contain the names of the input and output fields in the `train` dataframe.

```python
inputs = ["h_0", "h_1"]
outputs = ["h_2"]
```

The incremental learning process is started by calling the `learn_incremental` method.

```python
t_start = datetime.now(tz=UTC)
model = learn_incremental(
    train.as_stream(batch_size=1),
    learner,
    inputs,
    outputs,
)
delta_t = datetime.now(tz=UTC) - t_start
print(f"Learning took {np.round(delta_t.microseconds / 1000, 1)} ms")
```

For this example, the resulting metrics are about

| Learner type           | Runtime              | Mean Absolute Error | Mean Squared Error |
| ---------------------- | -------------------- | ------------------- | ------------------ |
| Linear Regression      | $15.5\: \mathrm{ms}$ | $0.0919$            | $0.01215$           |

Depending on the size of the dataset, the way the train and test set are split and shuffled, the learner's configuration, and other random factors, these values may vary. However, it is clear that the learner produced a model with relatively small errors ($\sim 2\%$) which could be used for tasks such as prediction.

## Run this example

To run this example, first make sure you followed the [installation instructions](../getting_started/prerequisites.md) to set up Python and `just`. Afterwards, you can run the examples from source.

### From source

Follow the [installation guide](../getting_started/installation.md) to install Flowcean and its dependencies from source. Afterwards, you can navigate to the `examples` folder and run the examples.

```sh
cd examples/one_tank
python run_incremental.py
```

[^1]: <https://de.mathworks.com/help/slcontrol/ug/watertank-simulink-model.html>.
