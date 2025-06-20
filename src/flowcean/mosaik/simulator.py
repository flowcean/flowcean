import logging
import queue
import threading
from typing import Any

import mosaik_api_v3
import numpy as np
from mosaik.exceptions import SimulationError
from mosaik_api_v3.types import Meta
from typing_extensions import override

logger = logging.getLogger("flowcean.mosaik.simulator")
META = {
    "type": "time-based",
    "models": {
        "Sensor": {"public": True, "params": ["uid"], "attrs": ["reading"]},
        "Actuator": {
            "public": True,
            "params": ["uid"],
            "attrs": ["setpoint"],
        },
    },
}


class SyncSimulator(mosaik_api_v3.Simulator):
    def __init__(self) -> None:
        super().__init__(META)

        self.sid: str = ""
        self.step_size: int = 0
        self.end: int = 0
        self.timeout: int = 20
        self.models = {}
        self.a_uid_dict: dict[str, str] = {}
        self.s_uid_dict: dict[str, str] = {}
        self.model_ctr: dict[str, int] = {"Sensor": 0, "Actuator": 0}
        self.sensors: dict[str, Any] = {}
        self._notified_done: bool = False

        self.sensor_queue: queue.Queue
        self.actuator_queue: queue.Queue
        self.sync_finished: threading.Event
        self.sync_terminate: threading.Event

    @override
    def init(
        self,
        sid: str,
        time_resolution: float = 1.0,
        *,
        step_size: int = 0,
        end: int = 0,
        sensor_queue: queue.Queue | None = None,
        actuator_queue: queue.Queue | None = None,
        sync_finished: threading.Event | None = None,
        sync_terminate: threading.Event | None = None,
        **sim_params: dict[str, Any],
    ) -> Meta:
        if sensor_queue is None:
            msg = "sensor_queue is None. Terminating!"
            raise ValueError(msg)
        if actuator_queue is None:
            msg = "actuator_queue is None. Terminating!"
            raise ValueError(msg)
        if sync_finished is None:
            msg = "sync_finished is None. Terminating!"
            raise ValueError(msg)
        if sync_terminate is None:
            msg = "sync_terminate is None. Terminating!"
            raise ValueError(msg)

        self.sid = sid
        self.step_size = step_size
        self.end = end
        self.sensor_queue = sensor_queue
        self.actuator_queue = actuator_queue
        self.sync_finished = sync_finished
        self.sync_terminate = sync_terminate

        return self.meta

    @override
    def create(
        self,
        num: int,
        model: str,
        **model_params: dict[str, Any],
    ) -> list:
        if num > 1:
            msg = (
                f"Only one model per sensor/actuator allowed but {num} were "
                "requested"
            )
            raise ValueError(msg)

        uid = (
            model_params["uid"] if isinstance(model_params["uid"], str) else ""
        )
        if not uid:
            msg = f"UID of the {model} model is empty"
            raise ValueError(msg)

        num_models = self.model_ctr[model]
        self.model_ctr[model] += 1
        eid = f"{model}-{num_models}"
        self.models[eid] = {"uid": uid, "value": None}

        if model == "Sensor":
            self.s_uid_dict[uid] = eid
        elif model == "Actuator":
            self.a_uid_dict[uid] = eid

        return [{"eid": eid, "type": model}]

    @override
    def step(self, time: int, inputs: dict, max_advance: int = 0) -> int:
        if self.sync_terminate.is_set():
            msg = "Stop was requested (step). Terminating simulation."
            raise SimulationError(msg)

        for sensor_eid, readings in inputs.items():
            reading = readings["reading"]
            for value in reading.values():
                if isinstance(value, bool):
                    value = 1 if value else 0  # noqa: PLW2901
                self.sensors[self.models[sensor_eid]["uid"]] = value

        ctr = 0
        while True:
            try:
                logger.debug("Trying to fill sensors ... (ctr=%d)", ctr)
                self.sensor_queue.put(
                    (False, self.sensors),
                    block=True,
                    timeout=3,
                )
                break
            except queue.Full:
                ctr += 1
            if ctr > self.timeout:
                msg = "Could not fill sensor queue. Terminating!"
                raise SimulationError(msg)

            if self.sync_terminate.is_set():
                msg = "Stop was requested (sensorq). Terminating simulation."
                raise SimulationError(msg)
        return time + self.step_size

    @override
    def get_data(self, outputs: dict) -> dict:
        data = {}
        ctr = 0
        while True:
            try:
                logger.debug("Trying to get actuators ... (ctr=%d)", ctr)
                actuator_data = self.actuator_queue.get(block=True, timeout=3)
                break
            except queue.Empty:
                ctr += 1
            if ctr > self.timeout:
                msg = "Actuator queue remains empty. Terminating!"
                raise SimulationError(msg)
            if self.sync_terminate.is_set():
                msg = "Stop was requested (actq). Terminating simulation."
                if not self._notified_done:
                    self.actuator_queue.task_done()
                    self._notified_done = True
                raise SimulationError(msg)
        for uid, value in actuator_data.items():
            if isinstance(value, np.ndarray) and value.shape == ():
                self.models[self.a_uid_dict[uid]]["value"] = value.item()
            else:
                self.models[self.a_uid_dict[uid]]["value"] = value

        for eid in outputs:
            data[eid] = {"setpoint": self.models[eid]["value"]}
        return data

    @override
    def finalize(self) -> None:
        try:
            logger.debug("Final attempt to fill sensor queue")
            self.sensor_queue.put(
                (True, self.sensors),
                block=True,
                timeout=3,
            )
        except queue.Full:
            msg = "Sensor queue is full. No final data available."
            logger.exception(msg)

        self.sync_finished.set()
