# Flowcean Modules

One of Flowcean's core features is to allow you to combine preprocessing, learning, evaluation, and postprocessing in whichever way you like.
This is achieved by organizing functionalities in modules as shown in th figure below.

![Flowcean Modules](../assets/flowcean_modules.svg)

Note, that modules described here do not coincide with Python modules on the implementation side.
If you want to know more about a specific Python module of Flowcean please refer to the [API Reference](https://flowcean.me/reference/flowcean/).
In the following, the term _module_ always refers to modules for the conceptual idea of Flowcean.
Anyways, some of the Flowcean modules form Python modules as well which we discuss at the end of this page.

In the figure, a hierarchical arrangement of modules is shown.
Every use-case with Flowcean defines a Machine Learning _Experiment_ for CPS.
An experiment consists of

- a **learning phase** defined by a learning _Strategy_ and
- an **evaluation phase** using a learned _Model_, a _Metric_, and an _Environment_.

An environment is a source of data.
Thus, three major types of environments exist, namely, _Data Sets_, _Simulations_, and _Data Streams_.
More information about the concepts and differentiation between these environments can be found in [the next section](https://flowcean.me/user_guide/environment/).

Of course, a data source and, thus, an environment is as well needed for the learning phase, i.e., inside the strategy.
In addition to that, a strategy involves a learner, representing the actual learning algorithm and, by that, the type of model learned.
The [model section](https://flowcean.me/user_guide/model/) provides more insights into the separation of learner and model modules.
The third element of a strategy are _Transforms_.
Transforms are all kind of preliminary operations on data sets which might be preprocessing, abstraction, augmentation, feature engineering, etc.

## Reference to Python API

On the implementation side, abstract base classes of the _Model_, _Metric_, _Learner_, and _Transform_ are included in the Python module [_core_](https://flowcean.me/reference/flowcean/core/).
Furthermore, the core includes the three main environments.

Concrete implementations of [models](https://flowcean.me/reference/flowcean/models/), [metrics](https://flowcean.me/reference/flowcean/metrics/), [learners](https://flowcean.me/reference/flowcean/learners/), and [transforms](https://flowcean.me/reference/flowcean/transforms/) form individual Python modules.
A special case is the learner module which has an additional sub-module for a gRPC-learner.
This learner implements a universal interface to external, e.g., non-Python learners via a gRPC connection.
Learning strategies form a further Python module.
As strategies are Python-functions, no abstract base class exists.
Three main strategies based on the three types of environments are explained in the section [Learning Strategies](https://flowcean.me/user_guide/learning_strategies/).

The super-module _Experiment_ is not a Python module, as this is the individual combination of Flowcean modules for a specific CPS application.
Learn more about experiments in the section [Experiment](https://flowcean.me/user_guide/experiment/).
