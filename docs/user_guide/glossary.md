# Glossary

This glossary provides definitions for key terms and concepts used throughout the Flowcean documentation.

## Core Framework Components

- **Adapter**: Component that connects Flowcean with real cyber-physical systems during deployment, handling the interface between the learned models and the physical system's sensors and actuators.
- **Environment**: Abstraction that describes the possible data sources for the learning and evaluation procedure. Environments can be offline (pre-recorded datasets), incremental (streaming data), or active (interactive systems where the learner can influence the environment).
- **Experiment**: Structure definition that orchestrates the entire machine learning pipeline, including data loading, preprocessing, model training, and evaluation. It serves as the main entry point for defining and executing learning tasks.
- **Learner**: A process that learns patterns from data. The learner takes an environment and produces a trained model by applying a specific learning algorithm to the data.
- **Metric**: Quantitative measure used to evaluate model performance during training and testing (e.g., Mean Absolute Error, Mean Squared Error, accuracy).
- **Model**: Represents the learned patterns that the learner has extracted from the data. A model can make predictions or decisions based on new input data.
- **Transform**: Generalization of data processing operations including preprocessing, feature engineering, and data augmentation. Transforms are applied to data before it is fed to the learner or used for evaluation.

## Learning Strategies

- **Active Learning**: Learning strategy where the learning algorithm actively influences the environment by selecting actions and receiving observations and rewards in return. The learner explores the state-action space to optimize its behavior.
- **Incremental Learning**: Learning strategy that processes data as a stream of single samples or small batches, updating the model continuously. Also known as passive online learning, where the learner adapts to new data without influencing how data is generated.
- **Offline Learning**: Learning strategy where a fixed batch of data is collected first and then processed at once to train a model. This is the traditional supervised learning approach with separate training and testing phases.
- **Strategy / Learning Strategy**: General term for the approach used to train a model, encompassing how data is accessed and processed (offline, incremental, or active).

## Active Learning and Reinforcement Learning

- **Action**: Input provided to an active environment by the learning agent. Actions influence the environment's state and the subsequent observations received.
- **Observation**: Output from an environment that reflects its current state. In active learning, observations are received after taking actions and are used to update the model.
- **Reward**: Scalar signal provided by an environment in reinforcement learning that indicates the quality or desirability of the current state or action. The learner aims to maximize cumulative reward over time.

## Data and Processing

- **Data Augmentation**: Technique of creating additional training data by applying transformations to existing data (e.g., adding noise, scaling, rotation) to improve model generalization.
- **Data Stream**: Continuous flow of data samples arriving over time, as opposed to a pre-recorded fixed dataset. Common in incremental and active learning scenarios.
- **Feature Engineering**: Process of creating, selecting, or transforming input variables (features) to improve model performance and interpretability.
- **Preprocessing**: Initial data processing steps applied before model training, such as normalization, scaling, handling missing values, or filtering.
- **Time Series**: Sequence of data points ordered in time, representing the temporal evolution of one or more variables. Common in cyber-physical systems monitoring and prediction tasks.

## System Properties and Analysis

- **Analysis / Diagnosis**: Refers to the process of systematically examining data or system states to identify causes of observed behavior, deviations, or errors. While analysis encompasses understanding and evaluating system behavior, diagnosis specifically targets the localization and explanation of disturbances or anomalies.
- **Resilience**: Ability of a system to recover or adapt after unexpected disturbances.
- **Robustness**: Ability of a system or model to deliver stable and reliable results under changed or erroneous input conditions.
- **Self-healing**: Ability of a system to automatically stabilize or repair itself through appropriate measures when errors or disturbances occur.
- **System Behavior**: The course or development of the system state over time, describing how the system evolves and responds to inputs or disturbances.
- **System State**: The totality of relevant variables that characterize the current condition of a system at a specific point in time.

## Cyber-Physical Systems

- **Actuator**: Physical component that performs actions on the system based on control signals, such as motors, valves, or switches.
- **Cyber-physical System (CPS)**: System that integrates physical processes with digital computation and networking. CPSs are complex systems combining physical components (sensors, actuators) with digital controllers and software.
- **Sensor**: Device that measures physical quantities (e.g., temperature, pressure, position) and provides observations of the system state.
- **Simulation**: Computational model that mimics the behavior of a real physical system, used for testing, training, or analysis without requiring access to the physical system.

## Testing and Evaluation

- **Test Case Generation**: Process of creating specific scenarios or inputs to evaluate the performance and behavior of a system or model systematically.
