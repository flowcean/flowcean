# Coffee Machine Example

This example shows how to train a model to predict the behavior of a coffee machine using an automaton. 
Inspired by the work of Steffen et al.[^1], we consider a simple coffee machine.
LearnLib[^2], a framework for automata learning written in Java, is used for the inference of the automaton.

!!! todo

    This explanation is outdated, we no longer use `experiment.yaml` but `run.py` to run experiments.
    Update the explanation accordingly.

## Run this example

There are two options to run this example.

### Docker Container

For this option, the *backend* for the learner is specified to *docker*. Example snippet from *experiment.yaml*:
```yaml
learners:
  - name: Automaton Learner
    class_path: agenc.learners.grpc.GrpcLearner
    arguments:
      backend: docker
      image_name: collaborating.tuhh.de:5005/w-6/forschung/agenc/agenc/java-automata-learner:latest
```

The external learner written in Java is containerized for this example. Thus, Docker must be installed to run it.
Follow the instructions on the [official Docker page](https://docs.docker.com/get-docker/) to install it.
The image for this example is stored in the TUHH image registry. Authentication is required for this.
Log in to the registry using `docker login collaborating.tuhh.de` and provide your credentials when asked.
Afterwards use the command line interface run the experiment

```bash
agenc --experiment experiment.yaml
```

The required images will be automatically retrieved, and a container will be started.

### Local Execution

For this option, the *backend* for the learner is specified to *none*. Example snippet from *experiment.yaml*:
```
learners:
  - name: Automaton Learner
    class_path: agenc.learners.grpc.GrpcLearner
    arguments:
      port: 8080
      ip: "localhost"
```

It has to be assured that the *ip* and *port* match those of the server configuration in Java.

**Before** running the example in Flowcean, the server-side has to be started manually by running the Java-project in *java/AutomataLearner*. A JDK and the Maven build automaton tool are required. More information on Maven projects using the LearnLib library can be found on [their website](https://learnlib.de/).

[^1]: Bernhard Steffen, Falk Howar, and Maik Merten. Introduction to Active Automata Learning from a Practical Perspective, pages 256–296. Springer Berlin Heidelberg, Berlin, Heidelberg, 2011. [doi:10.1007/978-3-642-21455-4_8](https://doi.org/10.1007/978-3-642-21455-4_8).
[^2]: Malte Isberner, Falk Howar, and Bernhard Steffen. The open-source learnlib - A framework for active automata learning. In Computer Aided Verification (CAV), volume 9206 of Lecture Notes in Computer Science, 487–495. Springer, 2015. [doi:10.1007/978-3-319-21690-4\_32](https://doi.org/10.1007/978-3-319-21690-4\_32).
