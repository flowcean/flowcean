import logging
import random
from copy import copy
from statistics import mean, median, stdev
from typing import Any

from typing_extensions import Self, override

import flowcean.cli
from flowcean.core import ActiveLearner, Model
from flowcean.core.strategies.active import Interface, learn_active
from flowcean.mosaik.energy_system import (
    Action,
    EnergySystemActive,
    Observation,
)

LOG = logging.getLogger("energy_example")


def run_active() -> None:
    flowcean.cli.initialize_logging(log_level="WARNING")
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
        LOG.exception("Error during environment operation.")

    environment.shutdown()
    LOG.info("Finished!")


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
        # self.best_action = best_action

    @override
    def predict(self, input_features: Observation) -> Action:
        _ = input_features

        actuators = []
        for actuator in self.action.actuators:
            new_actuator = copy(actuator)
            new_actuator.value = new_actuator.value_min + random.random() * (
                new_actuator.value_max - new_actuator.value_min
            )
            actuators.append(new_actuator)
        return Action(actuators=actuators)

    @override
    def save_state(self) -> dict[str, Any]:
        raise NotImplementedError

    @override
    @classmethod
    def load_from_state(cls, state: dict[str, Any]) -> Self:
        _ = state
        raise NotImplementedError


class MyLearner(ActiveLearner):
    model: MyModel
    rewards: list[float]
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
    vspace = "Box(low=0.0, high=1.5, shape=(), dtype=np.float32)"
    lspace = "Box(low=0.0, high=200.0, shape=(), dtype=np.float32)"

    voltages = sorted([s.value for s in sensors if "vm_pu" in s.uid])
    voltage_rewards = [
        Interface(
            value=voltages[0],
            uid="vm_pu-min",
            space=vspace,
            value_min=0.0,
            value_max=1.5,
        ),
        Interface(
            value=voltages[-1],
            uid="vm_pu-max",
            space=vspace,
            value_min=0.0,
            value_max=1.5,
        ),
        Interface(
            value=median(voltages),
            uid="vm_pu-median",
            space=vspace,
            value_min=0.0,
            value_max=1.5,
        ),
        Interface(
            value=mean(voltages),
            uid="vm_pu-mean",
            space=vspace,
            value_min=0.0,
            value_max=1.5,
        ),
        Interface(
            value=stdev(voltages),
            uid="vm_pu-std",
            space=vspace,
            value_min=0.0,
            value_max=1.5,
        ),
    ]

    lineloads = sorted(
        [s.value for s in sensors if ".loading_percent" in s.uid],
    )

    lineload_rewards = [
        Interface(
            value=lineloads[0],
            uid="lineload-min",
            space=lspace,
            value_min=0.0,
            value_max=200.0,
        ),
        Interface(
            value=lineloads[-1],
            uid="lineload-max",
            space=lspace,
            value_min=0.0,
            value_max=200.0,
        ),
        Interface(
            value=median(lineloads),
            uid="lineload-median",
            space=lspace,
            value_min=0.0,
            value_max=200.0,
        ),
        Interface(
            value=mean(lineloads),
            uid="lineload-mean",
            space=lspace,
            value_min=0.0,
            value_max=200.0,
        ),
        Interface(
            value=stdev(lineloads),
            uid="lineload-std",
            space=lspace,
            value_min=0.0,
            value_max=200.0,
        ),
    ]

    return voltage_rewards + lineload_rewards


if __name__ == "__main__":
    run_active()
