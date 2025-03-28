import logging
import queue
import threading
from typing import Any

import mosaik_api_v3
from mosaik.exceptions import SimulationError
from mosaik_api_v3.types import Meta
from typing_extensions import override

LOG = logging.getLogger("flowcean.mosaik.simulator")
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
    sid: str
    step_size: int
    end: int
    timeout: int
    a_uid_dict: dict[str, str]
    s_uid_dict: dict[str, str]
    sensor_queue: queue.Queue
    actuator_queue: queue.Queue
    sync_finished: threading.Event
    sync_terminate: threading.Event
    sensors: dict[str, Any]

    def __init__(self) -> None:
        super().__init__(META)

        # self.sid: str | None = None
        self.step_size = 0
        self.end = 0
        self.models = {}
        self.a_uid_dict = {}
        self.s_uid_dict = {}
        self.model_ctr = {"Sensor": 0, "Actuator": 0}
        # self.sensor_queue: queue.Queue | None = None
        # self.actuator_queue: queue.Queue | None = None
        # self.sync_finished: threading.Event | None = None
        # self.sync_terminate: threading.Event | None = None
        self.sensors = {}
        self.timeout = 20

    @override
    def init(
        self,
        sid: str,
        time_resolution: float = 1.0,
        **sim_params: dict[str, Any],
    ) -> Meta:
        self.sid = sid
        self.step_size = (
            sim_params["step_size"]
            if isinstance(sim_params["step_size"], int)
            else 0
        )
        self.sensor_queue = (
            sim_params["sensor_queue"]
            if isinstance(sim_params["sensor_queue"], queue.Queue)
            else queue.Queue()
        )
        self.actuator_queue = (
            sim_params["actuator_queue"]
            if isinstance(sim_params["actuator_queue"], queue.Queue)
            else queue.Queue()
        )
        self.sync_finished = (
            sim_params["sync_finished"]
            if isinstance(sim_params["sync_finished"], threading.Event)
            else threading.Event()
        )
        self.sync_terminate = (
            sim_params["sync_terminate"]
            if isinstance(sim_params["sync_terminate"], threading.Event)
            else threading.Event()
        )
        self.end = (
            sim_params["end"] if isinstance(sim_params["end"], int) else 0
        )
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
                LOG.debug("Trying to fill sensors ... (ctr=%d)", ctr)
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
                LOG.debug("Trying to get actuators ... (ctr=%d)", ctr)
                actuator_data = self.actuator_queue.get(block=True, timeout=3)
                break
            except queue.Empty:
                ctr += 1
            if ctr > self.timeout:
                msg = "Actuator queue remains empty. Terminating!"
                raise SimulationError(msg)
            if self.sync_terminate.is_set():
                msg = "Stop was requested (actq). Terminating simulation."
                self.actuator_queue.task_done()
                raise SimulationError(msg)
        for uid, value in actuator_data.items():
            self.models[self.a_uid_dict[uid]]["value"] = value
        for eid in outputs:
            data[eid] = {"setpoint": self.models[eid]["value"]}
        return data

    @override
    def finalize(self) -> None:
        try:
            LOG.debug("Final attempt to fill sensor queue")
            self.sensor_queue.put(
                (True, self.sensors),
                block=True,
                timeout=3,
            )
        except queue.Full:
            msg = "Sensor queue is full. No final data available."
            LOG.exception(msg)

        self.sync_finished.set()
