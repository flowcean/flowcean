from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

import grpc
import numpy as np
import polars as pl
from numpy.typing import NDArray

from agenc.core import Learner

from ._generated.learner_pb2 import (
    ColumnMetadata,
    DataPackage,
    DataType,
    FeatureType,
    LogLevel,
    Message,
    Observation,
    ObservationField,
    Prediction,
    Status,
)
from ._generated.learner_pb2_grpc import LearnerStub
from .dataset import Dataset

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
        dataset = _create_dataset(
            data.select(inputs).to_numpy(),
            data.select(outputs).to_numpy(),
        )
        proto_dataset = _dataset_to_proto(dataset)
        stream = self.stub.Train(proto_dataset)
        for status_message in stream:
            _log_messages(status_message.messages)
            if status_message.status == Status.STATUS_FAILED:
                raise RuntimeError("training failed")

    def predict(self, inputs: NDArray[Any]) -> NDArray[Any]:
        dataset = _create_dataset(inputs, np.zeros((inputs.shape[0], 1)))
        predictions = self.stub.Predict(_dataset_to_proto(dataset))
        _log_messages(predictions.status.messages)
        return _predictions_to_array(predictions)

    def drop(self) -> None:
        self.channel.close()


def _log_messages(messages: Iterable[Message]) -> None:
    for log_message in messages:
        logger.log(
            _loglevel_from_proto(log_message.log_level), log_message.message
        )


def _create_dataset(inputs: NDArray[Any], outputs: NDArray[Any]) -> Dataset:
    assert inputs.ndim == 2
    input_names = [f"i{i}" for i in range(0, inputs.shape[1])]
    output_names = [f"o{i}" for i in range(0, outputs.shape[1])]
    return Dataset(
        input_names,
        output_names,
        np.concatenate([inputs, outputs], axis=1),
    )


def _row_to_proto(
    row: NDArray[Any],
) -> Observation:
    fields = [ObservationField(double=entry) for entry in row]
    return Observation(fields=fields)


def _predictions_to_array(
    predictions: Prediction,
) -> NDArray[Any]:
    data = [
        [field.double for field in prediction.fields]
        for prediction in predictions.predictions
    ]
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


def _dataset_to_proto(
    dataset: Dataset,
) -> DataPackage:
    metadata: list[ColumnMetadata] = []
    observations: list[Observation] = []
    input_names = dataset.input_columns
    metadata.extend([
        ColumnMetadata(
            name=column_name,
            feature_type=(
                FeatureType.FEATURETYPE_INPUT
                if (column_name in input_names)
                else FeatureType.FEATURETYPE_TARGET
            ),
            data_type=DataType.DATATYPE_SCALAR,
        )
        for column_name in dataset.input_columns + dataset.output_columns
    ])
    observations.extend([_row_to_proto(row) for row in dataset.data])
    return DataPackage(metadata=metadata, observations=observations)
