---
icon: lucide/lightbulb
---

# Why Flowcean?

In Flowcean, we favor simple, interpretable abstractions of cyber-physical systems. Our goal is to capture the behavior relevant to a task while keeping the model understandable, rather than pursuing fidelity alone.

A compact equation, a decision tree, or a state-based model can make system behavior easier to inspect and explain. Flowcean supports exploring these representations and evaluating whether they are sufficient for prediction, monitoring, or testing.

## Spend less effort connecting tools

Preparing measurements, training models, and evaluating results often requires experiment-specific glue code. Flowcean provides reusable components for these steps.

Separating data preparation, learning, and evaluation lets you compare compatible modeling approaches without rebuilding the entire experiment. You can work with recorded datasets, incremental streams, or interactive environments, depending on how data becomes available.

## Choose the abstraction that fits

Flowcean does not prescribe a single model family. Explicit hybrid models combine continuous dynamics with discrete transitions; other tasks may call for learned equations, automata, or predictive models. Not every supported learner produces an interpretable model, but alternatives let you assess what additional complexity actually buys you.

## How Flowcean relates to other tools

Flowcean builds on existing learning libraries. The choice is often whether to use those libraries directly or through a shared CPS modeling workflow.

### Why not just scikit-learn?

[scikit-learn](https://scikit-learn.org/stable/) already provides interpretable models, preprocessing pipelines, and evaluation tools. If those cover your experiment, using it directly is a good choice.

Flowcean adds structure around the learning algorithm: environments for obtaining data, learning strategies, and evaluation across supported backends. Its [scikit-learn integration](../reference/sklearn.md) lets you retain familiar learners while exploring other modeling approaches.

### Why not just PyTorch?

[PyTorch](https://docs.pytorch.org/tutorials/beginner/basics/intro.html) is a natural choice when developing neural architectures and training procedures is the main task.

Flowcean addresses a different question: which model is suitable for the system behavior you want to capture? Its [PyTorch integration](../reference/torch.md) lets neural models participate in that investigation alongside simpler alternatives. It does not replace PyTorch's modeling capabilities or make neural models inherently interpretable.

### Why not write my own scripts?

For a small experiment, a standalone script may be the clearest solution. Flowcean experiments are Python scripts too; the difference is how much of the surrounding machinery you build yourself.

Flowcean becomes useful when data preparation, connections to environments, and evaluation need to be reused across experiments. The tradeoff is learning its interfaces and working within the capabilities of its integrations.
