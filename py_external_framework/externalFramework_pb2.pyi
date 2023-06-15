from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

COLUMNTYPE_INPUT: ColumnType
COLUMNTYPE_TARGET: ColumnType
COLUMNTYPE_UNDEFINED: ColumnType
DESCRIPTOR: _descriptor.FileDescriptor

class ColumnMetaData(_message.Message):
    __slots__ = ["lowerBound", "name", "type", "upperBound"]
    LOWERBOUND_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    UPPERBOUND_FIELD_NUMBER: _ClassVar[int]
    lowerBound: float
    name: str
    type: ColumnType
    upperBound: float
    def __init__(self, name: _Optional[str] = ..., lowerBound: _Optional[float] = ..., upperBound: _Optional[float] = ..., type: _Optional[_Union[ColumnType, str]] = ...) -> None: ...

class DataPackage(_message.Message):
    __slots__ = ["data", "metaData", "name"]
    DATA_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    data: _containers.RepeatedCompositeFieldContainer[DataRow]
    metaData: _containers.RepeatedCompositeFieldContainer[ColumnMetaData]
    name: str
    def __init__(self, name: _Optional[str] = ..., metaData: _Optional[_Iterable[_Union[ColumnMetaData, _Mapping]]] = ..., data: _Optional[_Iterable[_Union[DataRow, _Mapping]]] = ...) -> None: ...

class DataRow(_message.Message):
    __slots__ = ["data"]
    DATA_FIELD_NUMBER: _ClassVar[int]
    data: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, data: _Optional[_Iterable[float]] = ...) -> None: ...

class LearnerData(_message.Message):
    __slots__ = ["identifier", "resultMessage", "returnCode"]
    IDENTIFIER_FIELD_NUMBER: _ClassVar[int]
    RESULTMESSAGE_FIELD_NUMBER: _ClassVar[int]
    RETURNCODE_FIELD_NUMBER: _ClassVar[int]
    identifier: int
    resultMessage: str
    returnCode: int
    def __init__(self, returnCode: _Optional[int] = ..., resultMessage: _Optional[str] = ..., identifier: _Optional[int] = ...) -> None: ...

class ColumnType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []
