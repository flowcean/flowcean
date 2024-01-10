from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

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
    def __init__(
        self,
        server_address: str,
        max_message_length: int = MAX_MESSAGE_LENGTH,
    ) -> None:
        super().__init__()
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
        self.stub = LearnerStub(self.channel)

    def train(
        self,
        data: pl.DataFrame,
        inputs: list[str],
        outputs: list[str],
    ) -> None:
        proto_datapackage = _data_to_proto(data, inputs, outputs)
        stream = self.stub.Train(proto_datapackage)
        for status_message in stream:
            _log_messages(status_message.messages)
            if status_message.status == Status.STATUS_FAILED:
                raise RuntimeError("training failed")

    def predict(self, inputs: NDArray[Any]) -> NDArray[Any]:
        # This is kind of hacky. Because of the way "_data_to_proto" works,
        # no field names need to be given, instead all fields are automatically
        # assumed to be features.
        proto_datapackage = _data_to_proto(pl.DataFrame(inputs), [], [])
        predictions = self.stub.Predict(proto_datapackage)
        _log_messages(predictions.status.messages)
        return _predictions_to_array(predictions)

    def drop(self) -> None:
        self.channel.close()

    @override
    def save(self, path: Path) -> None:
        raise RuntimeError("Save no yet implemented for GrPC Learner")

    @override
    def load(self, path: Path) -> None:
        raise RuntimeError("Load no yet implemented for GrPC Learner")


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
    return np.array(data)


def _data_to_proto(
    data: pl.DataFrame,
    inputs: list[str],
    outputs: list[str],
) -> DataPackage:
    input_rows: list[DataRow] = [
        _row_to_proto(row) for row in data.select(inputs).rows()
    ]
    output_rows: list[DataRow] = [
        _row_to_proto(row) for row in data.select(outputs).rows()
    ]
    return DataPackage(inputs=input_rows, outputs=output_rows)


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
