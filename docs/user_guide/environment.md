# Environment

Generally, there are three possible ways to retrieve data from a Cyber-Physical System:

- a pre-recorded data set,
- a data stream from a simulation,
- a data stream from a real CPS.

Environments in Flowcean are a way to describe the possible data sources for the learning and evaluation procedure.
They are implemented via classes.
A high-level class diagram of the environment classes is provided below.

- **Offline Environment**: An offline environment has the ability to load a data set.
Data is pre-recorded (offline) and saved in a file or in a database.
This data is all at once provided to the learning pipeline.
This data is fixed and cannot be (meaningfully) changed.
The offline environment provides the interface to a data set but is not the data set itself.
- **Incremental Environment**: An incremental environment has the ability to provide a stream of single data samples or batches of data.
This could be provided by iterating over a data set, by an interface to a simulation, or by a live interface to the real Cyber-Physical System.
This is often referred to as *passive online learning*.
- **Active Environment**: An active environment has the ability to be influenced by the learning algorithm.
This is done by receiving an *Action*.
After an action is received the environment will evolve based on the action and the previous state.
After a set time interval or a discrete step inside a simulation, the environment can be observed again.
Examples are a controlled simulation or a controlled live experiment. 

``` mermaid
---
title: A high-level class diagram of the environment classes
--- 
classDiagram
  Environment <|-- OfflineEnvironment
  Environment <|-- IncrementalEnvironment
  Environment <|-- ActiveEnvironment
  
  class Environment{
    Provides data to a learner.
    + load() -> Self
    + with_transform(transform) -> TransformedEnvironment[Self]
  }
  class IncrementalEnvironment{
    Loads data in an iterative way. 
    + collect(n: int) -> DataFrame
  }
  class OfflineEnvironment{
    Loads data only once and in an non-interactive way. 
    + get_data() -> DataFrame
  }
  class ActiveEnvironment{
    Loads data in an interactive way
    allowing to act on the environment. 
    + act(Action)
    + step()
    + observe() -> Observation
  }
```



It is possible to apply [Transforms](https://www3.tuhh.de/agenc/user_guide/transforms/) to an environment.
This is done by applying a transformation (e.g. resampling or normalization) to the *DataFrame* that the environment provides.
As can be seen in the class diagram, the parent class `Environment` has a method `with_transform()` which allows to specify the transforms that are applied to an environment.

Depending on the environment class, different [Learning Strategies](https://www3.tuhh.de/agenc/user_guide/learning_strategies/) can be applied.
An active learning strategy, for example, can only be applied to an `ActiveEnvironment`.

For more information on the available classes and how environments are implemented in Flowcean, check out the [API](https://www3.tuhh.de/agenc/reference/flowcean/).