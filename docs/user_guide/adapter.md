# Adapter

Adapters allow to connect Flowcean with real CPS.
While similar to [environments](https://flowcean.me/user_guide/environment/) they are not the same.
Adapters are used during the deployment phase of a model while environments are used during the training and evaluation process of a model.
Adapters allow Flowcean to make observation from a CPS as well as passing actions or predictions back to it.
The key difference between adapters and environments is the fact, that adapters only provide the input observations from a CPS without the necessary outputs for training and evaluation.
With the ability to make observations and pass data back to a CPS, adapters allow to create a closed loop between the CPS and Flowcean and are an important part in the tool loop.
<!-- The figure below shows how adapters are integrated in the tool loop and therefore the deployment phase of a model. -->

In general, adapters need to be adjusted to a specific CPS to ensure seamless communication between the two.
Some generic adapters are provided by Flowcean which can serve as a starting point for a CPS specific adapter.
