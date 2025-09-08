from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import docker
import grpc
import polars as pl
from docker import DockerClient
from docker.models.containers import Container
from typing_extensions import Self, override

from flowcean.core import Model, SupervisedLearner

from ._generated.learner_pb2 import (
    DataField,
    DataPackage,
    LogLevel,
    Message,
    Prediction,
    Status,
    TimeSample,
    TimeSeries,
)
from ._generated.learner_pb2_grpc import LearnerStub

if TYPE_CHECKING:
    from collections.abc import Iterable

    from flowcean.core.data import Data

MAX_MESSAGE_LENGTH = 1024 * 1024 * 1024

logger = logging.getLogger(__name__)


class _Backend(ABC):
    @property
    @abstractmethod
    def server_address(self) -> str:
        pass


class _AddressBackend(_Backend):
    def __init__(self, address: str) -> None:
        self._address = address

    @property
    @override
    def server_address(self) -> str:
        return self._address


class _DockerBackend(_Backend):
    _docker_client: DockerClient
    _docker_container: Container
    _server_address: str
    _internal_port: int

    def __init__(
        self,
        image_name: str,
        internal_port: int,
        *,
        pull: bool,
    ) -> None:
        logger.info("Connecting to docker daemon")
        self._docker_client = docker.from_env()
        if pull:
            logger.info("Pulling image '%s'", image_name)
            self._docker_client.images.pull(image_name)
        logger.info("Starting container")
        container = self._docker_client.containers.run(  # type: ignore[type]
            image_name,
            ports={f"{internal_port}/tcp": ("127.0.0.1", None)},  # type: ignore[type]
            detach=True,
            remove=False,
        )
        if isinstance(container, Container):
            self._docker_container = container
        else:
            message = "did not receive a container"
            raise TypeError(message)
        self._docker_container.reload()
        client = docker.APIClient()
        time.sleep(2)
        ports = client.inspect_container(self._docker_container.id)[  # type: ignore[type]
            "NetworkSettings"
        ]["Ports"]
        host_ip = ports[f"{internal_port}/tcp"][0]["HostIp"]
        host_port = ports[f"{internal_port}/tcp"][0]["HostPort"]
        self._server_address = f"{host_ip}:{host_port}"

    @property
    @override
    def server_address(self) -> str:
        return self._server_address

    def __del__(self) -> None:
        if hasattr(self, "_docker_container"):
            logger.info("Stopping container")
            self._docker_container.stop()
            logger.info("Removing container")
            self._docker_container.remove()
        if hasattr(self, "_docker_client"):
            self._docker_client.close()  # type: ignore[type]


class GrpcPassiveAutomataLearner(SupervisedLearner, Model):
    """Binding for an external learner using gRPC.

    This learner class connects to an external learner using gRPC.
    Processed data for training or inference is sent to the learner and results
    are received from it.
    """

    _backend: _Backend
    _stub: LearnerStub

    def __init__(
        self,
        backend: _Backend,
    ) -> None:
        """Create a GrpcLearner.

        Args:
            backend: The backend to use for the learner.
        """
        super().__init__()
        self._backend = backend
        self.channel = grpc.insecure_channel(  # type: ignore[type]
            self._backend.server_address,
            options=[
                (
                    "grpc.max_send_message_length",
                    MAX_MESSAGE_LENGTH,
                ),
                (
                    "grpc.max_receive_message_length",
                    MAX_MESSAGE_LENGTH,
                ),
            ],
        )
        logger.info("Establishing gRPC connection...")
        self._stub = LearnerStub(self.channel)

    @classmethod
    def with_address(
        cls,
        address: str,
    ) -> Self:
        """Create a GrpcLearner with a specific address.

        Args:
            address: The address of the learner.

        Returns:
            A GrpcLearner instance.
        """
        backend = _AddressBackend(address)
        return cls(backend=backend)

    @classmethod
    def run_docker(
        cls,
        image: str,
        internal_port: int = 8080,
        *,
        pull: bool = True,
    ) -> Self:
        """Create a GrpcLearner running in a Docker container.

        Args:
            image: The name of the Docker image to run.
            internal_port: port the learner listens on inside the container.
            pull: Whether to pull the image from the registry.

        Returns:
            A GrpcLearner instance.
        """
        backend = _DockerBackend(image, internal_port, pull=pull)
        return cls(backend=backend)

    @override
    def learn(
        self,
        inputs: pl.LazyFrame,
        outputs: pl.LazyFrame,
    ) -> GrpcPassiveAutomataLearner:
        dfs = pl.collect_all([inputs, outputs])
        proto_datapackage = DataPackage(
            inputs=[_row_to_proto(row) for row in dfs[0].rows()],
            outputs=[_row_to_proto(row) for row in dfs[1].rows()],
        )
        stream = self._stub.Train(  # type: ignore[type]
            proto_datapackage,
        )
        for status_message in stream:  # type: ignore[type]
            _log_messages(status_message.messages)
            if status_message.status == Status.STATUS_FAILED:
                msg = "training failed"
                raise RuntimeError(msg)
        return self

    @override
    def _predict(self, input_features: Data) -> Data:
        proto_datapackage = DataPackage(
            inputs=[
                _row_to_proto(row)
                for row in input_features.collect(streaming=True).rows()
            ],
            outputs=[],
        )
        predictions = self._stub.Predict(proto_datapackage)
        _log_messages(predictions.status.messages)
        return _predictions_to_frame(predictions).lazy()

    def __del__(self) -> None:
        """Close the gRPC channel."""
        self.channel.close()


def _log_messages(messages: Iterable[Message]) -> None:
    for log_message in messages:
        logger.log(
            _loglevel_from_proto(log_message.log_level),
            log_message.message,
        )


def _row_to_proto(
    row: tuple[Any, ...],
) -> TimeSeries:
    return TimeSeries(
        samples=[
            (
                TimeSample(
                    time=entry["time"],
                    value=DataField(int=entry["value"]),
                )
            )
            for entry in row[0]
        ],
    )


def _predictions_to_frame(
    predictions: Prediction,
) -> pl.DataFrame:
    series_list = []
    for prediction in predictions.predictions:
        time = []
        value = []
        for sample in prediction.samples:
            time.append(sample.time)
            value.append(sample.value.int)
        df = pl.DataFrame({"time": time, "value": value})
        df = df.select(pl.struct(pl.all()).alias("output"))
        series_list.append(df["output"])
    return pl.DataFrame(
        pl.Series(
            "output",
            series_list,
            pl.List(pl.Struct({"time": pl.Float64, "value": pl.Float64})),
        ),
    )


def _loglevel_from_proto(loglevel: LogLevel.V) -> int:
    match loglevel:
        case LogLevel.LOGLEVEL_DEBUG:
            return logging.DEBUG
        case LogLevel.LOGLEVEL_INFO:
            return logging.INFO
        case LogLevel.LOGLEVEL_WARNING:
            return logging.WARNING
        case LogLevel.LOGLEVEL_ERROR:
            return logging.ERROR
        case LogLevel.LOGLEVEL_FATAL:
            return logging.FATAL
        case _:
            return logging.NOTSET
