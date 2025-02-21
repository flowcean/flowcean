"""This module contains the :class:`.ARLSyncSimulator`.

It is used by the :class:`.MosaikEnvironment` for synchronization.
"""

import logging
import queue
import threading
from datetime import datetime, timedelta
from typing import Any

import mosaik_api_v3
from mosaik.exceptions import SimulationError
from mosaik_api_v3.types import Meta

LOG = logging.getLogger("flowcean_mosaik.simulator")

META: dict[str, Any] = {
    "type": "time-based",
    "models": {
        "Sensor": {
            "public": True,
            "params": ["uid"],
            "attrs": ["reading"],
        },
        "Actuator": {
            "public": True,
            "params": ["uid"],
            "attrs": ["setpoint"],
        },
    },
}


class SyncSimulator(mosaik_api_v3.Simulator):
    """A simulator for the synchronization of palaestrAI and mosaik.

    Attributes:
    -----------
    sid : str
        The simulator id for this simulator given by mosaik
    step_size : int
        The step_size of this simulator
    models : dict
        A dictionary containing all models of this simulator.
        Currently, there is no reason why there should be more than one
        agent model.

    """

    def __init__(
        self,
        sensor_queue: queue.Queue,
        actuator_queue: queue.Queue,
        terminate: threading.Event,
        finished: threading.Event,
        end: int,
        timeout: int = 60,
    ) -> None:
        super().__init__(META)

        self.sensor_queue = sensor_queue
        self.actuator_queue = actuator_queue
        self.sync_terminate = terminate
        self.sync_finished = finished
        self.sid = None
        self.step_size: int = 0
        self.models = {}
        self.a_uid_dict = {}
        self.s_uid_dict = {}
        self.uid_dict = {}
        self.model_ctr = {"Sensor": 0, "Actuator": 0}
        self._env = None
        self._sim_time = 0
        self._now_dt = None
        self._timeout = timeout
        self._aq_timeout = 3
        self._sq_timeout = 5
        self._end = end

    def init(
        self,
        sid: str,
        time_resolution: float = 1,  # noqa: ARG002
        **sim_params: dict[str, Any],
    ) -> Meta:
        """Initialize this simulator.

        Called exactly ones after the simulator has been started.

        Parameters
        ----------
        sid : str
            Simulator id provided by mosaik.

        Returns:
        --------
        dict
            The meta description for this simulator as *dict*.

        """
        self.sid = sid
        step_size = sim_params["step_size"]
        if isinstance(step_size, int):
            self.step_size = step_size
        else:
            msg = "Step size is not of type int."
            raise TypeError(msg)

        if "start_date" in sim_params:
            try:
                self._now_dt = datetime.strptime(
                    str(sim_params["start_date"]),
                    "%Y-%m-%d %H:%M:%S%z",
                )
            except ValueError:
                print(
                    f"Unable to parse start date: {sim_params['start_date']}",
                )
                self._now_dt = None
        return self.meta

    def create(self, num: int, model: str, **model_params: dict) -> list:
        """Initialize the simulation model instance (entity).

        Parameters
        ----------
        num : int
            The number of models to create in one go.
        model : str
            The model to create. Needs to be present in the META.

        Returns:
        --------
        list
            A *list* of the entities created during this call.

        """
        if num > 1:
            msg = f"Only one model at a time but {num} were requested."
            raise ValueError(msg)

        uid = model_params["uid"]
        if model == "Sensor":
            if uid in self.s_uid_dict:
                msg = (
                    f"A Sensor model with uid '{uid}' was already created "
                    "but only one model per uid is allowed."
                )
                raise ValueError(msg)
        elif model == "Actuator":
            if uid in self.a_uid_dict:
                msg = (
                    f"An Actuator model with uid '{uid}' was already created "
                    "but only one model per uid is allowed.",
                )
                raise ValueError(msg)
        else:
            msg = f"Invalid model: '{model}'. Use ARLSensor or ARLActuator."
            raise ValueError(msg)
        num_models = self.model_ctr[model]
        self.model_ctr[model] += 1

        eid = f"{model}-{num_models}"
        self.models[eid] = {"uid": model_params["uid"], "value": None}
        self.uid_dict[model_params["uid"]] = eid

        if model == "Sensor":
            self.s_uid_dict[uid] = eid
        elif model == "Actuator":
            self.a_uid_dict[uid] = eid
        return [{"eid": eid, "type": model}]

    def step(self, time: int, inputs: dict, max_advance: int = 0) -> int:
        """Perform a simulation step.

        Parameters
        ----------
        time : int
            The current simulation time (the current step).
        inputs : dict
            A *dict* with inputs for the models.

        Returns:
        --------
        int
            The simulation time at which this simulator should
            perform its next step.
        """
        LOG.debug("Stepped SyncSim at step %d (advance %d)", time, max_advance)
        self._sim_time = time
        sensors: dict[str, Any] = {"simtime_ticks": self._sim_time}
        if self._now_dt is not None:
            self._now_dt += timedelta(seconds=self.step_size)
            sensors["simtime_timestamp"] = self._now_dt.strftime(
                "%Y-%m-%d %H:%M:%S%z",
            )
        if self.sync_terminate.is_set():
            msg = "Stop was requested. Terminating simulation."
            raise SimulationError(msg)

        for sensor_eid, readings in inputs.items():
            reading = readings["reading"]
            for value in reading.values():
                sensors[self.models[sensor_eid]["uid"]] = (
                    (1 if value else 0) if isinstance(value, bool) else value
                )

        if self._sim_time + self.step_size >= self._end:
            LOG.info("Repent, the end is nigh. Final readings are coming.")
            self._notified_done = True
        success = False
        while not success:
            try:
                self.sensor_queue.put(
                    (False, sensors),
                    block=True,
                    timeout=self._sq_timeout,
                )
                success = True
            except queue.Full:  # noqa: PERF203
                msg = "Failed to fill queue!"
                LOG.exception(msg)

        if self.sync_terminate.is_set():
            msg = "Stop was requested. Terminating simulation."
            raise SimulationError(msg)

        return time + self.step_size

    def get_data(self, outputs: dict) -> dict:
        """Return requested outputs (if feasible).

        Since this simulator does not generate output for its own, an
        empty dict is returned.

        Parameters
        ----------
        outputs : dict
            Requested outputs.

        Returns:
        --------
        dict
            An empty dictionary, since no output is generated.

        """
        if self.sync_terminate.is_set():
            msg = "Stop was requested. Terminating simulation."
            raise SimulationError(msg)
        data = {}
        success = False
        to_ctr = self._timeout
        actuator_data = {}
        while not success:
            try:
                actuator_data = self.actuator_queue.get(
                    block=True,
                    timeout=self._aq_timeout,
                )
                success = True
            except queue.Empty as exc:  # noqa: PERF203
                to_ctr -= self._aq_timeout
                timeout_msg = (
                    f"At step {self._sim_time}: Failed to get actuator "
                    "data from queue (queue is empty). Timeout in "
                    f"{to_ctr} s ..."
                )
                if to_ctr <= 0:
                    msg = (
                        f"No actuators after {to_ctr * 3:.1f} seconds. "
                        "Stopping mosaik."
                    )
                    raise SimulationError(msg) from exc
                log_timeout(to_ctr, self._timeout, timeout_msg)

        for uid, value in actuator_data.items():
            self.models[self.uid_dict[uid]]["value"] = value

        for eid in outputs:
            data[eid] = {"setpoint": self.models[eid]["value"]}

        if self.sync_terminate.is_set():
            msg = "Stop was requested. Terminating simulation."
            raise SimulationError(msg)
        return data

    def finalize(self) -> None:
        sensors: dict[str, Any] = {
            "simtime_ticks": self._sim_time + self.step_size,
        }
        if self._now_dt is not None:
            self._now_dt += timedelta(seconds=self.step_size)
            sensors["simtime_timestamp"] = self._now_dt.strftime(
                "%Y-%m-%d %H:%M:%S%z",
            )
        self.sensor_queue.put((True, sensors), block=True, timeout=3)


def log_timeout(ctr: int, timeout: int, msg: str) -> None:
    if ctr < timeout / 8:
        LOG.critical(msg)
    elif ctr < timeout / 4:
        LOG.error(msg)
    elif ctr < timeout / 2:
        LOG.warning(msg)
    else:
        LOG.info(msg)
