import ast
import logging
import queue
import threading
from collections.abc import Callable, Sequence
from copy import copy
from typing import Any

import midas.api
import mosaik
import numpy as np
import polars as pl
from midas.scenario.scenario import Scenario
from midas_store.meta import META
from mosaik.exceptions import SimulationError
from numpy.random import RandomState
from numpy.typing import NDArray
from typing_extensions import override

from flowcean.core import (
    Action,
    ActiveEnvironment,
    ActiveInterface,
    Observation,
    StopLearning,
)
from flowcean.polars import DataFrame

logger = logging.getLogger("energysystem")


class EnergySystemActive(ActiveEnvironment):
    _data_for_simulation: dict[str, int | float | NDArray[Any] | None]
    _data_from_simulation: dict[str, int | float | NDArray[Any] | None]
    reward: Callable

    def __init__(
        self,
        scenario_name: str,
        data_file: str,
        *,
        scenario_file: str | None = None,
        end: int = 86_400,
        seed: int | None = None,
        reward_func: Callable | None = None,
    ) -> None:
        super().__init__()

        self.rng: RandomState = RandomState(seed)
        if reward_func is None:
            self.reward = default_reward
        else:
            self.reward = reward_func

        self.sensor_queue = queue.Queue(1)
        self.actuator_queue = queue.Queue(1)
        self.sync_finished = threading.Event()
        self.sync_terminate = threading.Event()
        self.sim_finished = threading.Event()

        params = {
            "end": end,
            "seed": seed,
            "with_arl": True,
            "store_params": {
                "path": ".",
                "filename": data_file,
            },
        }

        self.scenario: Scenario = midas.api.run(
            scenario_name,
            params,
            scenario_file,
            no_build=True,
            no_run=True,
        )

        self.sensors: dict[str, ActiveInterface] = create_interface(
            self.scenario.sensors,
        )
        self.actuators: dict[str, ActiveInterface] = create_interface(
            self.scenario.actuators,
        )

        self.task = threading.Thread(
            target=start_mosaik,
            args=(self.scenario,),
            kwargs={
                "sensors": list(self.sensors),
                "actuators": list(self.actuators),
                "sensor_queue": self.sensor_queue,
                "actuator_queue": self.actuator_queue,
                "sync_finished": self.sync_finished,
                "sync_terminate": self.sync_terminate,
                "sim_finished": self.sim_finished,
            },
        )
        self.task.start()

        self.action = Action(actuators=list(self.actuators.values()))
        self.observation = Observation(
            sensors=list(self.sensors.values()),
            rewards=[],
        )
        self._data_for_simulation = {}
        self._data_from_simulation = {}
        self._get_data_from_sensor_queue()

    @override
    def _observe(self) -> Observation:
        logger.info("Returning current observation ...")
        return self.observation

    @override
    def step(self) -> None:
        logger.info("Stepping environment with current settings ...")
        self.actuator_queue.put(self._data_for_simulation, block=True)
        logger.debug("Placed actuators. Now waiting for sensors ...")
        done = self._get_data_from_sensor_queue()

        self._data_for_simulation = {}

        if done:
            logger.info("Environment has finished. Terminating.")
            raise StopLearning
        logger.info("Step complete!")

    def _get_data_from_sensor_queue(self) -> bool:
        done, self._data_from_simulation = self.sensor_queue.get(
            block=True,
            timeout=60,
        )

        sensors = []
        for uid, value in self._data_from_simulation.items():
            if uid == "simtime_ticks":
                continue
            if uid == "simtime_timestamp":
                continue
            sensor = copy(self.sensors[uid])
            sensor.value = value
            sensors.append(sensor)

        rewards = self.reward(sensors)
        self.observation = Observation(sensors=sensors, rewards=rewards)

        return done

    @override
    def act(self, action: Action) -> None:
        logger.info("Preparing actions on the environment ...")
        self._data_for_simulation = {}
        if action.actuators is not None:
            for actuator in action.actuators:
                self._data_for_simulation[actuator.uid] = actuator.value
        else:
            logger.info("Simulation will step without setpoints")
        # self._data_received = False
        self.action = action

    def shutdown(self) -> None:
        logger.info("Initiating shutdown procedure ...")
        if not self.sync_finished.is_set():
            self.sync_terminate.set()
        if not self.sim_finished.is_set():
            self.sim_finished.wait(30)
        self.task.join()
        del self.sensor_queue
        del self.actuator_queue
        logger.info("Shutdown complete!")


def create_interface(
    defs: list[dict[str, int | float | str | None]],
) -> dict[str, ActiveInterface]:
    object_map = {}

    for interf in defs:
        uid = str(interf["uid"])
        space = str(interf["space"])
        value = interf.get("value", None)
        if not isinstance(value, int | float):
            value = None
        vmin, vmax, shape, dtype = read_min_and_max_from_space(space)

        object_map[uid] = ActiveInterface(
            value=value,
            uid=uid,
            shape=shape,
            value_min=vmin,
            value_max=vmax,
            dtype=dtype,
        )
    return object_map


def read_min_and_max_from_space(
    space: str,
) -> tuple[
    int | float,
    int | float,
    Sequence[int],
    type[np.floating[Any]] | type[np.integer[Any]],
]:
    parts = space.split(",")
    value_min = 0
    value_max = 0
    shape = ()
    dtype = np.float32

    for part in parts:
        if "low" in part:
            _, val = part.split("=")
            try:
                value_min = int(val)
                continue
            except (TypeError, ValueError):
                pass

            value_min = float(val)

        if "high" in part:
            _, val = part.split("=")
            try:
                value_max = int(val)
                continue
            except (TypeError, ValueError):
                pass
            value_max = float(val)
        if "shape" in part:
            _, val = part.split("=")
            shape = ast.literal_eval(val)
            continue
        if "dtype" in part:
            _, val = part.split("=")
            if val.endswith(")"):
                val = val.removesuffix(")")
            dtype = get_numpy_type(val)
    return value_min, value_max, shape, dtype


def start_mosaik(
    scenario: Scenario,
    *,
    sensors: list[str],
    actuators: list[str],
    sensor_queue: queue.Queue,
    actuator_queue: queue.Queue,
    sync_finished: threading.Event,
    sync_terminate: threading.Event,
    sim_finished: threading.Event,
) -> None:
    """Start the mosaik simulation process."""
    scenario.makelists.sim_config["SyncSimulator"] = {
        "python": "flowcean.mosaik.simulator:SyncSimulator",
    }
    scenario.build()
    world = scenario.world
    if not isinstance(world, mosaik.World):
        sim_finished.set()
        logger.error(
            "Malformed world object: %s (%s)",
            str(world),
            type(world),
        )
        return

    logger.debug("Starting SyncSimulator ...")
    sync_sim = world.start(
        "SyncSimulator",
        step_size=scenario.base.step_size,
        start_date=scenario.base.start_date,
        sensor_queue=sensor_queue,
        actuator_queue=actuator_queue,
        sync_finished=sync_finished,
        sync_terminate=sync_terminate,
        end=scenario.base.end,
    )

    logger.debug("Connecting sensor entities ...")
    for uid in sensors:
        sid, eid, attr = uid.split(".")
        full_id = f"{sid}.{eid}"
        sensor_model = sync_sim.Sensor(uid=uid)
        logger.debug("Connecting %s ...", full_id)
        world.connect(
            scenario.entities[full_id],
            sensor_model,
            (attr, "reading"),
        )

    logger.debug("Connecting actuator entities ...")
    for uid in actuators:
        sid, eid, attr = uid.split(".")
        full_id = f"{sid}.{eid}"
        actuator_model = sync_sim.Actuator(uid=uid)
        world.connect(
            actuator_model,
            scenario.entities[full_id],
            ("setpoint", attr),
            time_shifted=True,
            initial_data={"setpoint": None},
        )

    logger.info("Starting mosaik run ...")
    try:
        world.run(until=scenario.base.end)
    except SimulationError:
        logger.info("Simulation finished non-regular.")
        world.shutdown()
    else:
        logger.info("Simulation finished.")
    sim_finished.set()


def default_reward(sensors: list[ActiveInterface]) -> list[ActiveInterface]:
    _ = sensors
    return [
        ActiveInterface(
            value=0,
            uid="default",
            shape=(),
            dtype=np.int32,
            value_max=1,
            value_min=0,
        ),
    ]


class EnergySystemOffline(DataFrame):
    """Offline version of the energy system environment.

    Runs the simulation without interaction and stores the results in
    a csv file. The file is read into polars and can be used for
    offline learning.

    """

    def __init__(
        self,
        scenario_name: str,
        data_file: str,
        *,
        scenario_file: str | None = None,
        end: int = 0,
        seed: int | None = None,
    ) -> None:
        if "DatabaseCSV" not in META["models"]:
            msg = (
                "Wrong version of midas_store is used. "
                "At least required is 2.1.0a1"
            )
            raise ValueError(msg)
        params = {
            "seed": seed,
            "store_params": {
                "path": ".",
                "filename": data_file,
                # "buffer_size": 10,
            },
        }
        if end > 0:
            params["end"] = end

        midas.api.run(scenario_name, params, scenario_file)
        data = pl.scan_csv(data_file)
        super().__init__(data)


def get_numpy_type(
    type_str: str,
) -> type[np.floating[Any]] | type[np.integer[Any]]:
    return getattr(np, type_str.split(".")[1])
