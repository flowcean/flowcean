import logging
from pathlib import Path
from statistics import mean, median, stdev

import numpy as np
from midas_palaestrai import ArlDefenderObjective

import flowcean.cli
from flowcean.core.strategies.active import ActiveInterface, learn_active
from flowcean.mosaik.energy_system import (
    EnergySystemActive,
)
from flowcean.palaestrai.sac_learner import SACLearner

logger = logging.getLogger("energy_example_sac")


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
    try:
        learner = SACLearner(actuator_ids, sensor_ids, ArlDefenderObjective())
        learner.setup(environment.action, environment.observation)
    except Exception:
        logger.exception("Failed to load learner")
        environment.shutdown()
        return

    try:
        learn_active(environment, learner)
        observation = environment.observe()

        print({r.uid: r.value for r in observation.rewards})
    except Exception:
        logger.exception("Error during environment operation.")
    except KeyboardInterrupt:
        print("Interrupted. Attempting to terminate environment.")

    environment.shutdown()
    learner.save(str(Path.cwd() / "_outputs"))
    learner.model.save(str(Path.cwd() / "_outputs" / "sac_model_only"))
    print(learner.objective_values)
    logger.info("Finished!")


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
