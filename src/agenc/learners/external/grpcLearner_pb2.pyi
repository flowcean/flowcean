from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DATATYPE_MATRIX: DataType
DATATYPE_SCALAR: DataType
DATATYPE_SCALAR_TIMESERIES: DataType
DATATYPE_UNDEFINED: DataType
DATATYPE_VECTOR: DataType
DATATYPE_VECTOR_TIMESERIES: DataType
DESCRIPTOR: _descriptor.FileDescriptor
FEATURETYPE_INPUT: FeatureType
FEATURETYPE_TARGET: FeatureType
FEATURETYPE_UNDEFINED: FeatureType
LOGLEVEL_DEBUG: LogLevel
LOGLEVEL_ERROR: LogLevel
LOGLEVEL_FATAL: LogLevel
LOGLEVEL_INFO: LogLevel
LOGLEVEL_UNDEFINED: LogLevel
LOGLEVEL_WARNING: LogLevel
STATUS_FAILED: Status
STATUS_FINISHED: Status
STATUS_RUNNING: Status
STATUS_UNDEFINED: Status

class ColumnMetadata(_message.Message):
    __slots__ = ["data_type", "feature_type", "name"]
    DATA_TYPE_FIELD_NUMBER: _ClassVar[int]
    FEATURE_TYPE_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    data_type: DataType
    feature_type: FeatureType
    name: str
    def __init__(self, name: _Optional[str] = ..., feature_type: _Optional[_Union[FeatureType, str]] = ..., data_type: _Optional[_Union[DataType, str]] = ...) -> None: ...

class DataPackage(_message.Message):
    __slots__ = ["metadata", "observations"]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    OBSERVATIONS_FIELD_NUMBER: _ClassVar[int]
    metadata: _containers.RepeatedCompositeFieldContainer[ColumnMetadata]
    observations: _containers.RepeatedCompositeFieldContainer[Observation]
    def __init__(self, metadata: _Optional[_Iterable[_Union[ColumnMetadata, _Mapping]]] = ..., observations: _Optional[_Iterable[_Union[Observation, _Mapping]]] = ...) -> None: ...

class Empty(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class MatrixDouble(_message.Message):
    __slots__ = ["column_count", "data", "row_count"]
    COLUMN_COUNT_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    ROW_COUNT_FIELD_NUMBER: _ClassVar[int]
    column_count: int
    data: _containers.RepeatedScalarFieldContainer[float]
    row_count: int
    def __init__(self, data: _Optional[_Iterable[float]] = ..., row_count: _Optional[int] = ..., column_count: _Optional[int] = ...) -> None: ...

class MatrixInt(_message.Message):
    __slots__ = ["column_count", "data", "row_count"]
    COLUMN_COUNT_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    ROW_COUNT_FIELD_NUMBER: _ClassVar[int]
    column_count: int
    data: _containers.RepeatedScalarFieldContainer[int]
    row_count: int
    def __init__(self, data: _Optional[_Iterable[int]] = ..., row_count: _Optional[int] = ..., column_count: _Optional[int] = ...) -> None: ...

class Message(_message.Message):
    __slots__ = ["log_level", "message", "sender"]
    LOG_LEVEL_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    SENDER_FIELD_NUMBER: _ClassVar[int]
    log_level: LogLevel
    message: str
    sender: str
    def __init__(self, log_level: _Optional[_Union[LogLevel, str]] = ..., sender: _Optional[str] = ..., message: _Optional[str] = ...) -> None: ...

class Observation(_message.Message):
    __slots__ = ["fields", "time_vector"]
    FIELDS_FIELD_NUMBER: _ClassVar[int]
    TIME_VECTOR_FIELD_NUMBER: _ClassVar[int]
    fields: _containers.RepeatedCompositeFieldContainer[ObservationField]
    time_vector: VectorDouble
    def __init__(self, fields: _Optional[_Iterable[_Union[ObservationField, _Mapping]]] = ..., time_vector: _Optional[_Union[VectorDouble, _Mapping]] = ...) -> None: ...

class ObservationField(_message.Message):
    __slots__ = ["double", "int", "matrix_double", "matrix_int", "vector_double", "vector_int"]
    DOUBLE_FIELD_NUMBER: _ClassVar[int]
    INT_FIELD_NUMBER: _ClassVar[int]
    MATRIX_DOUBLE_FIELD_NUMBER: _ClassVar[int]
    MATRIX_INT_FIELD_NUMBER: _ClassVar[int]
    VECTOR_DOUBLE_FIELD_NUMBER: _ClassVar[int]
    VECTOR_INT_FIELD_NUMBER: _ClassVar[int]
    double: float
    int: int
    matrix_double: MatrixDouble
    matrix_int: MatrixInt
    vector_double: VectorDouble
    vector_int: VectorInt
    def __init__(self, int: _Optional[int] = ..., double: _Optional[float] = ..., vector_int: _Optional[_Union[VectorInt, _Mapping]] = ..., vector_double: _Optional[_Union[VectorDouble, _Mapping]] = ..., matrix_int: _Optional[_Union[MatrixInt, _Mapping]] = ..., matrix_double: _Optional[_Union[MatrixDouble, _Mapping]] = ...) -> None: ...

class Prediction(_message.Message):
    __slots__ = ["predictions", "status"]
    PREDICTIONS_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    predictions: _containers.RepeatedCompositeFieldContainer[Observation]
    status: StatusMessage
    def __init__(self, predictions: _Optional[_Iterable[_Union[Observation, _Mapping]]] = ..., status: _Optional[_Union[StatusMessage, _Mapping]] = ...) -> None: ...

class StatusMessage(_message.Message):
    __slots__ = ["messages", "progress", "status"]
    MESSAGES_FIELD_NUMBER: _ClassVar[int]
    PROGRESS_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    messages: _containers.RepeatedCompositeFieldContainer[Message]
    progress: int
    status: Status
    def __init__(self, status: _Optional[_Union[Status, str]] = ..., messages: _Optional[_Iterable[_Union[Message, _Mapping]]] = ..., progress: _Optional[int] = ...) -> None: ...

class VectorDouble(_message.Message):
    __slots__ = ["data"]
    DATA_FIELD_NUMBER: _ClassVar[int]
    data: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, data: _Optional[_Iterable[float]] = ...) -> None: ...

class VectorInt(_message.Message):
    __slots__ = ["data"]
    DATA_FIELD_NUMBER: _ClassVar[int]
    data: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, data: _Optional[_Iterable[int]] = ...) -> None: ...

class FeatureType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []

class DataType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []

class Status(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []

class LogLevel(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []
