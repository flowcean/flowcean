# Use Case

The three main use cases in the AGenC research project will be described in the following sections.

## Data-Driven Monitoring for Autonomous Mobile Robots

Mobile robots and autonomous vehicles are key technological developments in logistics that can significantly improve the efficiency, cost reduction and flexibility of production and logistics processes.
The autonomous navigation of these vehicles depends largely on reliable localization systems.
Various environmental influences such as uneven ground, dust, direct sunlight or extreme weather conditions can disrupt localization systems and jeopardize process reliability.

Data-driven monitoring software is to be developed and evaluated using Flowcean in order to detect and monitor these faults.
The aim is to control the localization quality during operation, identify errors and make adjustments to ensure the safety and efficiency of the robots.
This includes the collection of relevant environmental data, the creation of models to estimate the uncertainty of the robot pose and the use of synthetic and real sensor data for modeling.
The models are validated experimentally to ensure their transferability to real application scenarios.

## Energy Optimization and Container Weight Estimation

Container handling using ships and seaport terminals plays a central role in global trade and the maintenance of logistics and supply chains.
To achieve this more efficiently, there is a global trend towards increased automation.
One of the few processes still often performed manually is twist-lock handling.
This involves manually removing connecting pieces when unloading containers or inserting them during loading to prevent containers from slipping and falling overboard.
The automatic lashing platform from KALP GmbH automates this step.
However, to select the optimal configuration for the platform, the weight of the containers is required.
In practice, this information is often unavailable or very imprecise, which limits the platform's potential.
The Flowcean framework is used to develop a prediction system that determines the container weight based on the pressure parameters measured internally within the platform.
An interface between the Flowcean framework and the industrial PLC of the ALP is developed, tested, and implemented.
This interface ensures smooth data exchange between the systems, allowing the ALP to optimally switch its valves and minimize process time.
The generated model undergoes testing and validation using both synthetic data from simulations and real data from the ALP in terminal operations.
This in-situ testing demonstrates the model's applicability and capabilities.
