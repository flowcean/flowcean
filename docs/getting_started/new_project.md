# Your First Project with Flowcean

### Analyze your current problem

Flowcean can be used for different types and strategies of learning with different sources of data.
First step when using Flowcean is to think about your problem at hand and specify:

- What type of data can be provided?
- Is it a dataset, a dataset, a simulation or something else?
- Are inputs to the data required (is it interactive)?
- What kind of learning process is required?

According to these questions flowcean offers different learning environments (offline, incremental and active).
To decide what kind approach you need see (link).

In this example we a simulation that requires a floating point value as input and returns a value as output (randomly chosen in this case).
The objective of the learning process is to predict the output based on the simulation's input value.

The best fitting flowcean environment for this simulation is an ActiveOnlineEnvironment.
This type of environment requires a learning algorithm that is able to perform active learning e.g.
returning an input to the siimulation.
With flowcean the algorithm is stored and can be used in the model of the learner.
In this getting started we show how to use Flowcean to implement your own learning pipeline.

### The main function

The first step in your code is to activate the flowcean CLI utilities, which mainly includes logging.
The CLI utilities allows you to specify log messages (e.g.
output).

```python
import flowcean.cli
flowcean.cli.initialize()
```

For active learning, the next step is to set an environment representing your dataset/datastream/simulation:

```python
environment = MyEnvironment(
    initial_state=0.0,
    max_value=10.0,
    max_num_iterations=1_000,
)
```

The class `MyEnvironment` does not exist yet, we will create it later.

Next we will need a learner:

```python
learner = MyLearner()
```

Like the environment, the learner is not yet defined.
We will come to this.

The last piece we need is a learning strategy.
For this example we chose the active learning strategy as we have a simulation as our data source.
The strategy is implemented as a model.

```python
from flowcean.strategies.active import learn_active

model = learn_active(
    environment,
    learner,
)
```

This will build a model based on the chosen strategy on the given environment and with the given learner.

The full code will look like this:

```python
import flowcean.cli
from flowcean.strategies.active import learn_active

def main() -> None:
    flowcean.cli.initialize()

    environment = MyEnvironment(
        initial_state=0.0,
        max_value=10.0,
        max_num_iterations=1_000,
    )

    learner = MyLearner()

    model = learn_active(
        environment,
        learner,
    )
    print(model)  # Show which model was build
    print(learner.rewards)  # Show the reward over time


if __name__ == "__main__":
    main()
```

### Writing an environment for your data/simulation

To implement the described simulation use case, we first need to define the objects that are used to exchange data between environment and learner:

```python
from dataclasses import dataclass

Action = float

@dataclass
class ReinforcementObservation:
    reward: float
    sensor: float
```

With this, we define the environment class `MyEnvironment`.

```python
from flowcean.core import ActiveOnlineEnvironment
from flowcean.strategies.active import StopLearning

class MyEnvironment(ActiveOnlineEnvironment[Action, ReinforcementObservation]):
    state: float
    max_value: float
    last_action: Action | None
    max_num_iterations: int

```

The *state* contains the value that is determined by the environment in each simulation step, which is the value that should be predicted by the input value.

The *max_value* specifies the data range in which the state can be, e.g, providing a *max_value* of 3 will produce states in [0, 3).

The *last_action* will store the current input of the simulation and is used to evaluate the objective, which is to predict the internal *state* of the environment.

*max_num_iterations* is simply used to define how many steps the simulation should perform.

Note that those definitions are only used for the example environment described earlier.

Your actual environment object might need more variables.
Flowcean allows you to specify your environment according to the characteristics of your data source.

Next, we need to define the function to provide inputs for the simulation.

```python
    @override
    def act(self, action: Action) -> None:
        self.last_action = action
```

Since we don't actually have a simulation running in the background, we will simply store the value of the action in *last_action*.

To step the simulation, we need to define a function as well:

```python
    @override
    def step(self) -> None:
        self.state = random.random() * self.max_value
        self.max_num_iterations -= 1
        if self.max_num_iterations < 0:
            raise StopLearning
```

Here, we would call the actual simulation object to perform calculations based on the inputs provided earlier.
In this example, we will simply get a new *state* value and decrease the simulation counter.
In case we reached the maximum number of steps, we will raise a `StopLearning` exception to stop the learning process.

The last crucial function to implement is the `observe` function.

It is important that this functions does not do any changes of the environment object, i.e., calling it multiple times provides the same results.

```python

    @override
    def observe(self) -> ReinforcementObservation:
        return ReinforcementObservation(
            reward=self._calculate_reward(),
            sensor=self.state,
        )

    def _calculate_reward(self) -> float:
        if self.last_action is None:
            return nan
        return self.max_value - abs(self.state - self.last_action)
```

The full code for the environment definition is shown below:

```python
from dataclasses import dataclass
from flowcean.core import ActiveOnlineEnvironment

Action = float

@dataclass
class ReinforcementObservation:
    reward: float
    sensor: float

class MyEnvironment(ActiveOnlineEnvironment[Action, ReinforcementObservation]):
    state: float
    max_value: float
    last_action: Action | None
    max_num_iterations: int

    def __init__(
        self,
        initial_state: float,
        max_value: float,
        max_num_iterations: int,
    ) -> None:
        self.state = initial_state
        self.max_value = max_value
        self.last_action = None
        self.max_num_iterations = max_num_iterations

    @override
    def load(self) -> Self:
        return self

    @override
    def act(self, action: Action) -> None:
        self.last_action = action

    @override
    def step(self) -> None:
        self.state = random.random() * self.max_value
        self.max_num_iterations -= 1
        if self.max_num_iterations < 0:
            raise StopLearning

    @override
    def observe(self) -> ReinforcementObservation:
        return ReinforcementObservation(
            reward=self._calculate_reward(),
            sensor=self.state,
        )

    def _calculate_reward(self) -> float:
        if self.last_action is None:
            return nan
        return self.max_value - abs(self.state - self.last_action)
```


### Writing your own learning algorithm

In Flowcean, the learning approach is based on mainly two classes — *learner* and *model*.
While the actual learning is implemented in the *learner* class, the model can be seen as the result of the learning exp.
a function to predict output values based a given input.
The model itself can be stored and loaded for later applications and use cases.

Since we have an ActiveOnlineEnvironment, the learner must be an active learner.
We can define a learner class based on the Action and ReinforcementObservation objects we defined in the environment section.


```python
from flowcean.core import ActiveLearner

class MyLearner(ActiveLearner[Action, ReinforcementObservation]):
    model: MyModel
    rewards: list[float]
```
The *model* contains a model object of the class MyModel, which can predict based on given observations.
The learner will hold a model object with the actual training algorithm.

*Rewards* contains values to compare the proposed actions with observations in the environment.

The function `learn_active` provides a model based on the previously proposed action and the given observation of the environment.
In this example the learner itself does not learn something.
Therefore, returned models are based on random numbers.

```python
@override
    def learn_active(
        self,
        action: Action,
        observation: ReinforcementObservation,
    ) -> Model:
        _ = observation.sensor
        self.model = MyModel(best_action=random.random())
        self.rewards.append(observation.reward)
        return self.model
```
And each time, learn_active is called, the value-to-be-returned will be determined, to "simulate" a learning process.
This specific model will return a random floating point value.

The `propose_action` function returns a suggested action of the model based on a given observation of the environment for a example the sensor.

```python
@override
    def propose_action(self, observation: ReinforcementObservation) -> Action:
        sensor = observation.sensor
        action = self.model.predict(pl.DataFrame({"sensor": [sensor]}))
        return action["action"][0]
```
The full code of the learner class is defined below.

```python
class MyLearner(ActiveLearner[Action, ReinforcementObservation]):
    model: MyModel
    rewards: list[float]

    def __init__(self) -> None:
        self.model = MyModel(best_action=nan)
        self.rewards = []

    @override
    def learn_active(
        self,
        action: Action,
        observation: ReinforcementObservation,
    ) -> Model:
        _ = observation.sensor
        self.model = MyModel(best_action=random.random())
        self.rewards.append(observation.reward)
        return self.model

    @override
    def propose_action(self, observation: ReinforcementObservation) -> Action:
        sensor = observation.sensor
        action = self.model.predict(pl.DataFrame({"sensor": [sensor]}))
        return action["action"][0]
```

The model class is implemented to create model objects based on the learner and the observations of the environment.
The model itself does not contain the learning algorithm.
It contains the prediction function given by the learner to predict output values based on a given input.

```python
from flowcean.strategies.active import StopLearning, learn_active

class MyModel(Model):
    best_action: float
```

*best_action* ...

```python
@override
    def predict(self, input_features: pl.DataFrame) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "action": [
                    self.best_action for _ in range(len(input_features))
                ],
            },
        )
```

This is the actual application of the model.
The `predict` function returns a data frame based on the underlying function, learnend by the learner.


```python
@override
    def save(self, path: Path) -> None:
        raise NotImplementedError
```

For later usage models can be saved with the `save` function.

```python
@override
    def load(self, path: Path) -> None:
        raise NotImplementedError
```

To reuse models in a given environment they can be loaded with the `load` function.

The full code of the learner class is defined below.

```python
class MyModel(Model):
    best_action: float

    def __init__(self, best_action: float) -> None:
        self.best_action = best_action

    @override
    def predict(self, input_features: pl.DataFrame) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "action": [
                    self.best_action for _ in range(len(input_features))
                ],
            },
        )

    @override
    def save(self, path: Path) -> None:
        raise NotImplementedError

    @override
    def load(self, path: Path) -> None:
        raise NotImplementedError
```
