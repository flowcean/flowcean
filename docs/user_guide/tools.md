# Tools

After a model has been learned, it should do something useful.
It should be deployed on an actual Cyber-Physical System.

At the moment, it is up to the user what to do with the trained models.
However, the Flowcean developers have the vision of making the deployment of models in real-world scenarios easier using **Tools**.

!!! todo
    - tool implementations are currently not included
    - this section introduces the general idea of tools

It is planned to have a test case generation tool and a predictive monitoring tool.
These tools will briefly be described in the following.

## Test Case Generation

The test case generation tool lets the user select a model and requirements that the model should fulfil.
A test case generator creates samples for the model and applies the input samples to the model.
The output of the model is observed and compared with the previously defined requirements.
If a requirement is not fulfilled, the generated sample presents a counter-example.
At the end of the process, all counter-examples are shown to the user.
The list of counter-examples can then be used to test the CPS in potentially critical scenarios.

## Predictive Monitoring

The predictive monitoring tool is connected live to the CPS and receives input samples from the sensors of the system.
The input is applied to the model which predicts future state of the Cyber-Physical System.
If the state is outside the desired bounds, a warning message will be published allowing the user to take appropriate measures.
