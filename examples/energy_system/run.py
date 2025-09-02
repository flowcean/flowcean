import logging
import random
from copy import copy
from statistics import mean, median, stdev

import numpy as np
from typing_extensions import override

import flowcean.cli
from flowcean.core import (
    Action,
    ActiveInterface,
    ActiveLearner,
    Model,
    Observation,
    learn_active,
)
from flowcean.mosaik import EnergySystemActive

logger = logging.getLogger("energy_example")


def run_active() -> None:
    flowcean.cli.initialize()
    environment = EnergySystemActive(
        "midasmv_der",
        "my_results.csv",
        reward_func=calculate_reward,
    )

    actuator_ids = [
        "Pysimmods-0.Photovoltaic-0.q_set_mvar",
        "Pysimmods-0.Photovoltaic-1.q_set_mvar",
        "Pysimmods-0.Photovoltaic-2.q_set_mvar",
    ]
    sensor_ids = [
        "Powergrid-0.0-bus-1.vm_pu",
        "Powergrid-0.0-bus-1.in_service",
        "Powergrid-0.0-bus-2.vm_pu",
        "Powergrid-0.0-bus-3.vm_pu",
        "Powergrid-0.0-bus-4.vm_pu",
        "Powergrid-0.0-bus-5.vm_pu",
        "Powergrid-0.0-bus-6.vm_pu",
    ]

    learner = MyLearner(actuator_ids, sensor_ids)
    learner.load(environment.action, environment.observation)

    try:
        learn_active(environment, learner)
        observation = environment.observe()

        print({r.uid: r.value for r in observation.rewards})
    except Exception:
        logger.exception("Error during environment operation.")

    environment.shutdown()
    logger.info("Finished!")


class MyModel(Model):
    best_action: float
    action: Action
    observation: Observation

    def __init__(
        self,
        action: Action,
        observation: Observation,
    ) -> None:
        self.action = action
        self.observation = observation

    @override
    def _predict(self, input_features: Observation) -> Action:
        _ = input_features

        actuators = []
        for actuator in self.action.actuators:
            new_actuator = copy(actuator)
            if (
                new_actuator.value_min is not None
                and isinstance(
                    new_actuator.value_min,
                    int | float | np.ndarray,
                )
                and new_actuator.value_max is not None
                and isinstance(
                    new_actuator.value_max,
                    int | float | np.ndarray,
                )
                and not isinstance(actuator.value, str)
            ):
                new_actuator.value = (
                    new_actuator.value_min
                    + random.random()
                    * (new_actuator.value_max - new_actuator.value_min)
                )

            actuators.append(new_actuator)
        return Action(actuators=actuators)


class MyLearner(ActiveLearner):
    model: MyModel
    rewards: list[ActiveInterface]
    actuator_ids: list[str]
    sensor_ids: list[str]
    action: Action
    observation: Observation

    def __init__(
        self,
        actuator_ids: list[str],
        sensor_ids: list[str],
    ) -> None:
        self.actuator_ids = actuator_ids
        self.sensor_ids = sensor_ids
        self.rewards = []

    def load(self, action: Action, observation: Observation) -> None:
        self.action = filter_action(action, self.actuator_ids)
        self.observation = filter_observation(observation, self.sensor_ids)

        self.model = MyModel(self.action, self.observation)

    @override
    def learn_active(
        self,
        action: Action,
        observation: Observation,
    ) -> Model:
        _ = action

        self.rewards.append(observation.rewards[0])

        return self.model

    @override
    def propose_action(self, observation: Observation) -> Action:
        filtered = filter_observation(observation, self.sensor_ids)

        return self.model.predict(filtered)


def filter_action(action: Action, available_ids: list[str]) -> Action:
    return Action(
        actuators=[a for a in action.actuators if a.uid in available_ids],
    )


def filter_observation(
    observation: Observation,
    available_ids: list[str],
) -> Observation:
    return Observation(
        sensors=[s for s in observation.sensors if s.uid in available_ids],
        rewards=observation.rewards,
    )


def calculate_reward(sensors: list) -> list:
    voltages = sorted([s.value for s in sensors if "vm_pu" in s.uid])
    voltage_rewards = [
        ActiveInterface(
            value=voltages[0],
            uid="vm_pu-min",
            shape=(),
            dtype=np.float32,
            value_min=0.0,
            value_max=1.5,
        ),
        ActiveInterface(
            value=voltages[-1],
            uid="vm_pu-max",
            shape=(),
            dtype=np.float32,
            value_min=0.0,
            value_max=1.5,
        ),
        ActiveInterface(
            value=median(voltages),
            uid="vm_pu-median",
            shape=(),
            dtype=np.float32,
            value_min=0.0,
            value_max=1.5,
        ),
        ActiveInterface(
            value=mean(voltages),
            uid="vm_pu-mean",
            shape=(),
            dtype=np.float32,
            value_min=0.0,
            value_max=1.5,
        ),
        ActiveInterface(
            value=stdev(voltages),
            uid="vm_pu-std",
            shape=(),
            dtype=np.float32,
            value_min=0.0,
            value_max=1.5,
        ),
    ]

    lineloads = sorted(
        [s.value for s in sensors if ".loading_percent" in s.uid],
    )

    lineload_rewards = [
        ActiveInterface(
            value=lineloads[0],
            uid="lineload-min",
            shape=(),
            dtype=np.float32,
            value_min=0.0,
            value_max=200.0,
        ),
        ActiveInterface(
            value=lineloads[-1],
            uid="lineload-max",
            shape=(),
            dtype=np.float32,
            value_min=0.0,
            value_max=200.0,
        ),
        ActiveInterface(
            value=median(lineloads),
            uid="lineload-median",
            shape=(),
            dtype=np.float32,
            value_min=0.0,
            value_max=200.0,
        ),
        ActiveInterface(
            value=mean(lineloads),
            uid="lineload-mean",
            shape=(),
            dtype=np.float32,
            value_min=0.0,
            value_max=200.0,
        ),
        ActiveInterface(
            value=stdev(lineloads),
            uid="lineload-std",
            shape=(),
            dtype=np.float32,
            value_min=0.0,
            value_max=200.0,
        ),
    ]

    return voltage_rewards + lineload_rewards


if __name__ == "__main__":
    run_active()
