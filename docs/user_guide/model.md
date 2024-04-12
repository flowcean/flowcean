# Model

Models and learners are separated in Flowcean.

A **learner** trains a model.
The learner is a process that learns patterns from data.
A learner can be based on different machine learning algorithms with various configurations.
During the learning process the learner updates its internal parameters or structure in order to optimize its task.
Which learners are implemented in Flowcean can be found [here](https://www3.tuhh.de/agenc/reference/flowcean/learners/).

Once the learner is done training, it outputs a **model**.
The model represents the learned patterns that the learner has extracted from the data.
The model can be used to make predictions on new, unseen data.
Models can be saved and loaded.
For more information take a look at the [API](https://www3.tuhh.de/agenc/reference/flowcean/core/model/).
