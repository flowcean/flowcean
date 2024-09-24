import logging
import queue
import sys
import threading
from collections.abc import Callable
from copy import copy
from dataclasses import dataclass
from datetime import datetime
from importlib import import_module
from multiprocessing import Process
from statistics import mean, median, stdev
from typing import Any, Self

import mosaik_api_v3
import numpy as np
from loguru import logger
from numpy.random import RandomState
from simulator import SyncSimulator

from flowcean.core import ActiveEnvironment
from flowcean.strategies.active import StopLearning

LOG = logging.getLogger("mosaik_environment")
DATE_FORMAT = "%Y-%m-%d %H:%M:%S%z"


@dataclass
class Sensor:
    value: int | float | None
    uid: str


@dataclass
class Reward:
    value: int | float | None
    uid: str


@dataclass
class Actuator:
    value: int | float | None
    uid: str


@dataclass
class Observations:
    sensors: list[Sensor]
    rewards: list[Reward]


@dataclass
class Actions:
    actuators: list[Actuator]


class MosaikEnvironment(ActiveEnvironment[Actions, Observations]):
    def __init__(
        self,
        start_date: str,
        end: int | str,
        seed: int | None,
        *,
        sync_freq: int = 900,
        module: str = "midas_palaestrai.descriptor:Descriptor",
        description_func: str = "describe",
        instance_func: str = "get_world",
        sync_host: str = "localhost",
        sync_port: int = 58976,
        silent: bool = False,
        params: dict[str, Any] | None = None,

    ) -> None:
        self.rng: RandomState = RandomState(seed)

        self.sensor_queue: queue.Queue
        self.actuator_queue: queue.Queue

        self._module = module
        self._description_func = description_func
        self._instance_func = instance_func

        self._sync_host = sync_host
        self._sync_port = sync_port

        self._mosaik_params = {} if params is None else params
        self._mosaik_params["meta_params"] = {
            "seed": self.rng.randint(2**32 - 1),
            "end": parse_end(end),
            "start_date": parse_start_date(start_date, self.rng),
            "sync_freq": sync_freq,
            "silent": silent,
        }
        self.sync_task: threading.Thread
        self.sim_proc: Process

        self._data_for_simulation: dict | None = None

        self.sensors = None
        self.rewards = None
        self._data_from_simulation = None

    def load(self) -> Self:
        self.sensor_queue = queue.Queue(1)
        self.actuator_queue = queue.Queue(1)

        description, instance = load_funcs(
            self._module, self._description_func, self._instance_func
        )
        sensor_description, actuator_description, world_state = description(
            self._mosaik_params
        )

        self.sync_task = threading.Thread(
            target=_start_simulator,
            args=[
                self._sync_host,
                self._sync_port,
                self.sensor_queue,
                self.actuator_queue,
            ],
        )
        self.sync_task.start()
        self.sim_proc = Process(
            target=_start_world,
            args=(
                instance,
                self._mosaik_params,
                sensor_description,
                actuator_description,
                self._sync_host,
                self._sync_port,
            ),
        )
        self.sim_proc.start()
        self.sensors, self.sen_map = create_sensors(sensor_description)
        self.actuators, self.act_map = create_actuators(actuator_description)
        return self

    def act(self, action: Actions) -> None:
        self._data_for_simulation = {}
        self.actuators = action.actuators
        if self.actuators is not None and self.actuators:
            for actuator in self.actuators:
                self._data_for_simulation[actuator.uid] = actuator.value
        else:
            LOG.warning("Simulation will step without setpoints.")

    def step(self) -> None:
        if self._data_for_simulation is None:
            raise StopLearning

        self.actuator_queue.put(self._data_for_simulation, block=True)

        done, self._data_from_simulation = self.sensor_queue.get(
            block=True, timeout=60
        )
        if done or self._data_for_simulation is None:
            raise StopLearning

        self.sensors = None
        self.rewards = None

    def observe(self) -> Observations:
        if self.sensors is not None and self.rewards is not None:
            return Observations(sensors=self.sensors, rewards=self.rewards)

        self.sensors = []
        for uid, value in self._data_from_simulation.items():
            if uid == "simtime_ticks":
                continue
            if uid == "simtime_timestamp":
                continue

            new_sensor = copy(self.sen_map[uid])
            new_sensor.value = value
            self.sensors.append(new_sensor)
        self.rewards = calculate_reward(self.sensors, self.actuators)

        return Observations(sensors=self.sensors, rewards=self.rewards)


def calculate_reward(sensors: list) -> list:
    voltages = sorted([s.value for s in sensors if "vm_pu" in s.uid])
    voltage_rewards = [
        Reward(value=voltages[0], uid="vm_pu-min"),
        Reward(value=voltages[-1], uid="vm_pu-max"),
        Reward(value=median(voltages), uid="vm_pu-median"),
        Reward(value=mean(voltages), uid="vm_pu-mean"),
        Reward(value=stdev(voltages), uid="vm_pu-std"),
    ]

    lineloads = sorted(
        [s.value for s in sensors if ".loading_percent" in s.uid]
    )

    lineload_rewards = [
        Reward(value=lineloads[0], uid="lineload-min"),
        Reward(value=lineloads[-1], uid="lineload-max"),
        Reward(value=median(lineloads), uid="lineload-median"),
        Reward(value=mean(lineloads), uid="lineload-mean"),
        Reward(value=stdev(lineloads), uid="lineload-std"),
    ]

    # in_service = np.sort(
    #     np.array([s() for s in state if "in_service" in s.uid]), axis=None
    # )
    # in_service_unique, in_service_counts = np.unique(
    #     in_service, return_counts=True
    # )
    # in_service_dict = dict(zip(in_service_unique, in_service_counts))

    # num_in_service = in_service_dict.get(1) if (1 in in_service_dict) else 1

    # num_out_of_service = (
    #     in_service_dict.get(0) if (0 in in_service_dict) else 0
    # )

    # in_service_rewards = [
    #     RewardInformation(
    #         np.array(num_in_service, dtype=np.int32),
    #         Discrete(len(in_service) + 1),
    #         reward_id="num_in_service",
    #     ),
    #     RewardInformation(
    #         np.array(num_out_of_service, dtype=np.int32),
    #         Discrete(len(in_service) + 1),
    #         reward_id="num_out_of_service",
    #     ),
    # ]

    return voltage_rewards + lineload_rewards


def parse_start_date(start_date: str, rng: RandomState) -> str:
    if start_date == "random":
        start_date = (
            f"2020-{rng.randint(1, 12):02d}-"
            f"{rng.randint(1, 28):02d} "
            f"{rng.randint(0, 23):02d}:00:00+0100"
        )
    try:
        datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S%z")
    except ValueError:
        msg = (
            f"Unable to parse start_date {start_date} "
            f"(format string: {DATE_FORMAT})"
        )
        LOG.exception(msg)
    return start_date


def parse_end(end: str | int) -> int:
    """Read the *end* value from the params dict.

    The *end* value is an integer, but sometimes it is provided
    as float, or as str like '15*60'. In the latter case, the
    str is evaluated (i.e., multiplied). In any case, *end* is
    returned as int.

    """
    if isinstance(end, str):
        parts = end.split("*")
        end = 1
        for part in parts:
            end *= float(part)
    return int(end)


def load_funcs(
    module_name: str, description_func: str, instance_func: str
) -> Any:
    """Load the description functions.

    Expects a dictionary containing the keys *"module"*,
    *"description_func"*, and "instance_func". *"module"* can
    either be a python module or a python class. The path segments
    for modules are separated by a dot "." and a class is separated
    by a colon ":", e.g., if *descriptor* is a module::

        {
            "module": "midas_palaestrai.descriptor",
            "description_func": "describe",
            "instance_func": "get_world",
        }

    or, if *Descriptor* is a class::

        {
            "module": "midas_palaestrai.descriptor:Descriptor",
            "description_func": "describe",
            "instance_func": "get_world",
        }


    Parameters
    ----------
    params : dict
        A *dict* containing the keys as described above.

    Returns:
    --------
    tuple
        A *tuple* of the description function and the instance
        function.

    """
    if ":" in module_name:
        module, clazz = module_name.split(":")
        module = import_module(module)
        obj = getattr(module, clazz)()
    else:
        obj = import_module(module_name)

    dscr_func = getattr(obj, description_func)
    inst_func = getattr(obj, instance_func)

    return dscr_func, inst_func


def _start_simulator(
    host: str, port: int | str, q1: queue.Queue, q2: queue.Queue
) -> None:
    argv_backup = sys.argv
    sys.argv = [
        argv_backup[0],
        "--remote",
        f"{host}:{port}",
        "--log-level",
        "error",
    ]
    mosaik_api_v3.start_simulation(SyncSimulator(q1, q2))
    sys.argv = argv_backup


def _start_world(
    get_world: Callable,
    params: dict,
    sensors: list,
    actuators: list,
    host: str,
    port: int | str,
) -> None:
    meta_params = params["meta_params"]

    world, entities = get_world(params)
    world.sim_config["ARLSyncSimulator"] = {"connect": f"{host}:{port}"}
    arlsim = world.start(
        "ARLSyncSimulator",
        step_size=meta_params["arl_sync_freq"],
        start_date=meta_params.get("start_date", None),
    )

    for sensor in sensors:
        sid, eid, attr = sensor["uid"].split(".")
        full_id = f"{sid}.{eid}"
        sensor_model = arlsim.ARL_Sensor(uid=sensor["uid"])
        world.connect(entities[full_id], sensor_model, (attr, "reading"))

    for actuator in actuators:
        sid, eid, attr = actuator["uid"].split(".")
        full_id = f"{sid}.{eid}"
        actuator_model = arlsim.ARL_Actuator(uid=actuator["uid"])
        world.connect(
            actuator_model,
            entities[full_id],
            ("setpoint", attr),
            time_shifted=True,
            initial_data={"setpoint": None},
        )

    logger.disable("mosaik")
    logger.disable("mosaik_api_v3")

    world.run(
        until=meta_params["end"], print_progress=not meta_params["silent"]
    )


def create_sensors(sensor_defs: list) -> list[Sensor]:
    """Create sensors from the sensor description.

    The description is provided during initialization.

    Returns:
    --------
    list
        The *list* containing the created sensor objects.

    """
    sensors = []
    sensor_map = {}
    for sensor in sensor_defs:
        uid = str(sensor.get("uid", sensor.get("sensor_id", "Unnamed Sensor")))
        try:
            value = sensor.get("value", None)
            sensors.append(Sensor(value=value, uid=uid))
        except RuntimeError:
            LOG.exception(sensor)
            raise
        sensor_map[uid] = copy(sensors[-1])

    return sensors, sensor_map


def create_actuators(actuator_defs: list) -> list[Actuator]:
    """Create actuators from the actuator description.

    The description is provided during initialization.

    Returns:
    -------
    list
        The *list* containing the created actuator objects.

    """
    actuators = []
    actuator_map = {}
    for actuator in actuator_defs:
        uid = str(
            actuator.get(
                "uid", actuator.get("actuator_id", "Unnamed Actuator")
            )
        )

        try:
            value = actuator.get(
                "value",
                actuator.get("setpoint", None),
            )
            actuators.append(Actuator(value=value, uid=uid))
        except RuntimeError:
            LOG.exception(actuator)
            raise
        actuator_map[uid] = copy(actuators[-1])
    return actuators, actuator_map
