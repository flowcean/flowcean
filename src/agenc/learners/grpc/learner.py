from __future__ import annotations

import logging
import socket
from collections.abc import Iterable
from contextlib import closing
from enum import Enum
from pathlib import Path
from typing import Any

import docker  # type:ignore
import grpc
import numpy as np
import polars as pl
from numpy.typing import NDArray
from typing_extensions import override

from agenc.core import Learner

from ._generated.learner_pb2 import (
    DataField,
    DataPackage,
    DataRow,
    LogLevel,
    Message,
    Prediction,
    Status,
)
from ._generated.learner_pb2_grpc import LearnerStub

MAX_MESSAGE_LENGTH = 1024 * 1024 * 1024

logger = logging.getLogger(__name__)


class GrpcLearner(Learner):
    """Binding for an external learner using gRPC.

    This learner class connects to an external program/learner using gRPC.
    Processed data for training or inference is sent to the learner and results
    are received from it.

    Args:
        backend (str): The name of the backend to use. Valid options are:
            `none`: No external process or container is started.
            `docker`: Use docker to start a container with the
            external learner.
        port (int, required when `backend = none`): Port to connect to.
        ip (str, required when `backend = none`): IP address to connect to.
        image_name (str, required when `backend = docker`): Name of the image
            to pull and start.
        max_message_length (int, optional): Maximal length of a gRPC message.
            Defaults to 1 GB.
    """

    class _BackendType(Enum):
        NONE = 0
        DOCKER = 1

    def __init__(
        self,
        backend: str,
        port: int | None = None,
        ip: str | None = None,
        image_name: str | None = None,
        max_message_length: int = MAX_MESSAGE_LENGTH,
    ) -> None:
        super().__init__()

        # Check the backend and arguments
        match backend:
            case "none":
                self._backend = GrpcLearner._BackendType.NONE
                if port is None:
                    raise ValueError('Argument "port" not given')
                if ip is None:
                    raise ValueError('Argument "ip" not given')
                server_address = f"{ip}:{port}"
            case "docker":
                self._backend = GrpcLearner._BackendType.DOCKER
                if image_name is None:
                    raise ValueError('Argument "image_name" not given')

                # Start the learner container
                logger.info("Connecting to docker daemon")
                self._docker_client = docker.from_env()
                free_port: int = find_free_port()
                logger.info(f"Pulling image '{image_name}'")
                self._docker_client.images.pull(image_name)
                logger.info("Starting container")
                self._docker_container: Any = (
                    self._docker_client.containers.run(
                        image_name,
                        ports={"8080/tcp": free_port},
                        detach=True,
                    )
                )
                server_address = f"localhost:{free_port}"

        self.channel = grpc.insecure_channel(
            server_address,
            options=[
                (
                    "grpc.max_send_message_length",
                    max_message_length,
                ),
                (
                    "grpc.max_receive_message_length",
                    max_message_length,
                ),
            ],
        )
        logger.info("Establishing gRPC connection with container")
        self.stub = LearnerStub(self.channel)

    def train(
        self,
        input_features: pl.DataFrame,
        output_features: pl.DataFrame,
    ) -> None:
        proto_datapackage = DataPackage(
            inputs=[_row_to_proto(row) for row in input_features.rows()],
            outputs=[_row_to_proto(row) for row in output_features.rows()],
        )
        stream = self.stub.Train(proto_datapackage)
        for status_message in stream:
            _log_messages(status_message.messages)
            if status_message.status == Status.STATUS_FAILED:
                raise RuntimeError("training failed")

    def predict(self, input_features: pl.DataFrame) -> NDArray[Any]:
        proto_datapackage = DataPackage(
            inputs=[_row_to_proto(row) for row in input_features.rows()],
            outputs=[],
        )
        predictions = self.stub.Predict(proto_datapackage)
        _log_messages(predictions.status.messages)
        return _predictions_to_array(predictions)

    def __del__(self) -> None:
        # Close the gRPC conenction
        self.channel.close()

        # Perform cleanup dependent on backend
        match self._backend:
            case GrpcLearner._BackendType.NONE:
                pass
            case GrpcLearner._BackendType.DOCKER:
                self._docker_container.stop()
                self._docker_container.remove()
                self._docker_client.close()

    @override
    def save(self, path: Path) -> None:
        raise RuntimeError("Save not yet implemented for gRPC Learner")

    @override
    def load(self, path: Path) -> None:
        raise RuntimeError("Load not yet implemented for gRPC Learner")


def _log_messages(messages: Iterable[Message]) -> None:
    for log_message in messages:
        logger.log(
            _loglevel_from_proto(log_message.log_level), log_message.message
        )


def _row_to_proto(
    row: tuple[Any, ...],
) -> DataRow:
    return DataRow(
        fields=[
            (
                DataField(int=entry)
                if isinstance(entry, int)
                else DataField(double=entry)
            )
            for entry in row
        ]
    )


def _predictions_to_array(
    predictions: Prediction,
) -> NDArray[Any]:
    data = [
        [
            field.double if field.HasField("double") else field.int
            for field in prediction.fields
        ]
        for prediction in predictions.predictions
    ]
    print(data)
    return np.array(data)


def _loglevel_from_proto(loglevel: LogLevel.V) -> int:
    match loglevel:
        case LogLevel.LOGLEVEL_DEBUG:
            return logging.DEBUG
        case LogLevel.LOGLEVEL_INFO:
            return logging.INFO
        case LogLevel.LOGLEVEL_WARNING:
            return logging.WARN
        case LogLevel.LOGLEVEL_ERROR:
            return logging.ERROR
        case LogLevel.LOGLEVEL_FATAL:
            return logging.FATAL
        case _:
            return logging.NOTSET


def find_free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("localhost", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return int(s.getsockname()[1])
