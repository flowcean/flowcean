import pickle
from collections.abc import Callable
from pickle import PickleBuffer
from typing import Any

from _typeshed import SupportsWrite

type _BufferCallback = Callable[[PickleBuffer], Any] | None

def dump(
    obj: Any,
    file: SupportsWrite[bytes],
    protocol: int = ...,
    buffer_callback: _BufferCallback = ...,
) -> None: ...
def dumps(
    obj: Any,
    protocol: int = ...,
    buffer_callback: _BufferCallback = ...,
) -> bytes: ...

load = pickle.load
loads = pickle.loads
