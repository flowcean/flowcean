# Learning Strategies

Within Flowcean there are different learning strategies that can be used to train a model.
The learning strategies are implemented as modules and can be combined to create a custom learning pipeline.
Flowcean provides standard learning strategies implementing so-called learning processes for each environment.
Learning strategies in the context of learning models for Cyber-Physical Systems can be categorized into *offline*, *incremental*, and *active* approaches.

Offline learning strategies involve processing a fixed
batch of data at once, updating model parameters in one go.
This approach is commonly
used by methods such as supervised learning, where the learner gains knowledge from a pre-
collected dataset.
In the ML community this is also referred to as batch learning.
Within Flowcean this is called **Offline Learning**.

On the other hand, online learning takes place incrementally, with the learner updating the
model continuously.
In each incremental step, the learner processes small subsets of available
data, even down to a single sample.
Online learning can further be classified into active and
passive strategies.

**Active learning** involve the learner actively by letting the learner provide actions to the environment.
The environment, in turn, responds with samples.
This interactive process is notably found in approaches like Reinforcement Learning, where the learner’s actions influence the simulation state or dataset selection.

In contrast, (passive) **Incremental Learning** lacks the learner’s ability to influence the environment.
The environment independently determines which sample to present to the learner, making it
a non-interactive process.
Learning from streaming data is an example of passive online learning.
This type of learning is often called online learning in the ML community.

The three learning strategies that are already implemented are visualized and explained in detail below.

![learning_strategies](../assets/learning_strategies.svg)

In general, custom learning strategies can be constructed in a similar manner.
All learning strategies have in common that the output is a [model](https://www3.tuhh.de/agenc/user_guide/model/).
The model can be saved and compared to other models that might be trained in the future.
Furthermore, the model can be used to predict states of Cyber-Physical System or detect faulty behavior.
For more information on what can be done with a model, check out the documentation on [Tools](https://www3.tuhh.de/agenc/user_guide/tools/).
Technically, strategies are represented as modular functions inside Flowcean.
For more information on the implementation of the learning strategies take a look at the [API](https://www3.tuhh.de/agenc/reference/flowcean/strategies/).

## Offline Learning

The first step of an offline learning strategy is to get the dataset from an environment.
This environment is typically an [OfflineEnvironment](https://www3.tuhh.de/agenc/reference/flowcean/core/environment/offline/).
Along with the environment the learner requires the names of the inputs and outputs.
Transforms can be applied to the input features of the data set and the outputs.
The last step is to learn the model using a learning algorithm.
For this, the learning algorithm receives the entire transformed dataset at once.

## Incremental Learning

For the incremental learning strategy, a learner is connected to an [IncrementalEnvironment](https://www3.tuhh.de/agenc/reference/flowcean/core/environment/passive_online/).
It iteratively receives data - either in single packets or small batches.
Along with the environment the learner requires the names of the inputs and outputs.
Transforms can be applied to the input features of the data set and the outputs.
The next step is to incrementally learn the model using a learning algorithm.
This means the model is updated in the process.
The learning process stops when the environment ends, i.e. when the data stream is stopped or the data set is empty.

## Active Learning

For the active learning strategy, a learner is connected to an [ActiveEnvironment](https://www3.tuhh.de/agenc/reference/flowcean/core/environment/passive_online/).
First, the environment is observed.
Next, the learner proposes an action which should be applied to the environment.
This is called *acting on the environment*.
Subsequently, the environment advances or performs a *step*.
Again, the environment gets observed.
Based on the observations and the applied action, the model is updated.
The learning process stops when the environment ends or when the learner requests to stop.
