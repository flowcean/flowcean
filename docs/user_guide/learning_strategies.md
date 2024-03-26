# Learning Strategies

Within Flowcean there are different learning strategies that can be used to train a model.
The learning strategies are implemented as modules and can be combined to create a custom learning pipeline.
Flowcean provides standard learning strategies implementing nominal learning processes for each environment.
These learning strategies are visualized and introduced below.
In general, custom learning strategies can be constructed in a similar manner.
Technically, stratgies are represented as modular functions inside Flowcean.

!!! todo

    - Check with [Learning Strategies](https://collaborating.tuhh.de/w-6/forschung/agenc/learning-strategies/-/jobs/artifacts/main/raw/document.pdf?job=build)
    - Describe the terminology online/offline, incremental, active and why we have arrived at these definitions/categories
    - Clarify the difference between Learning Algorithm and Learning Strategy
    - Rough textual description of the graphic
    - The product is a model! -> Explain what is possible with the model

#### Offline Learning

Learn from an offline environment by learning from the input-output pairs.

!!! todo

    explain the path in detail

#### Incremental Learning

Learn from a Incremental environment by incrementally learning from the input-output pairs.
The learning process stops when the environment ends.

!!! todo

    explain the path in detail

#### Active Learning

Learn from an active environment by interacting with it and learning from the observations.
The learning process stops when the environment ends or when the learner requests to stop.

!!! todo

    explain the path in detail

