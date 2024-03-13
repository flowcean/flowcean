# Coffee Machine Example

This example shows how to train a model to predict the behavior of a coffee machine using an automaton. 
The example is taken from {footcite:t}`SHM2011`. LearnLib{footcite:p}`IHS2015`, a framework for automata learning written in Java is used for the inference of the automaton.

## Run this example

There are two options to run this example. The option can be specified in the *experiment.yaml* file of the example.

### 1. Docker Container

For this option, the *backend* for the learner is specified to *docker*. Example snippet from *experiment.yaml*:
```
learners:
  - name: Automaton Learner
    class_path: agenc.learners.grpc.GrpcLearner
    arguments:
      backend: docker
      image_name: collaborating.tuhh.de:5005/w-6/forschung/agenc/agenc/java-automata-learner:latest
```

The external learner written in Java is containerized for this example. Thus, Docker must be installed in order to run it.
Follow the instructions on the [official Docker page](https://docs.docker.com/get-docker/) to install it.
The image for this example is stored in the TUHH image registry. Authentication is required for this.
Log in to the registry using `docker login collaborating.tuhh.de` and provide your credentials when asked.
Afterwards use the command line interface run the experiment

```bash
agenc --experiment experiment.yaml
```

The required images will be automatically retrieved and a container will be started.

### 2. Local Execution

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

## References

```{footbibliography}
```