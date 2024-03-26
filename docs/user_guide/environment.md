# Environment

Generally, there are three possible ways data from Cyber Physical Systems (CPS) can be available: 

- a pre-recorded data set, 
- a live stream from a simulation,
- a live stream from a real-world CPS.

Environments in Flowcean are a way to describe the possible data sources for the learning and evaluation procedure.
They are implemented via classes.
The structure and ability of the Environments are largely influenced by the learning strategies.

#### Offline Environment:

A Data set.
Data is pre-recorded (offline) and saved in a file or in a data base.
This data is all at once provided to the learning pipeline.
This data is already fixed and cannot be (meaningfully) changed.

#### Incremental Environment:

An incremental environment can provide a stream of single data points or batches of data.
This could be provided by iterating over a data set, by an interface to a simulation or an live-interface to the real CPS.
This often referred to as *passive online learning*.

#### Active Environment:

An active environment is able to be influenced by the learning algorithm.
This is done by receiving an *Action*.
After an action is received the environment will evolve based on the action and the previous states.
After some time, the environment can be observed.
This could be a controlled simulation, or a controlled live experiment.

![Environment Classes](../assets/environment_classes.svg)

!!! todo

    - Class for the possible data sources
    - Various options (see Stephan's notes and also [Learning Strategies](https://collaborating.tuhh.de/w-6/forschung/agenc/learning-strategies/-/jobs/artifacts/main/raw/document.pdf?job=build))
        - Simulation
        - Data set
        - Data stream
            - Live
            - Iteration over data set
    - Reference to API
    - Clarify that not all learning strategies are applicable to all environments -> possible transition to the "Learning Strategies" section
