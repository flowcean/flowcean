from typing import cast

import polars as pl
from polars._typing import PolarsDataType


def get_time_series_value_type(t: pl.DataType) -> PolarsDataType:
    """Returns the Polars data type of the value in a time-series type.

    Args:
        t: The Polars data type.

    Returns:
        The time-series type of the data type.
    """
    # Unpack the time-series structure.
    # First check if the data type is a list...
    if not isinstance(t, pl.List):
        msg = f"Expected a List data type, got {t}."
        raise TypeError(msg)
    t = cast("pl.List", t)

    # ... of structs.
    if not isinstance(t.inner, pl.Struct):
        msg = f"Expected a List of Structs, got a List of {t}."
        raise TypeError(msg)
    t = cast("pl.Struct", t.inner)

    # Then check if the struct has a field named "value".
    value_field = [field for field in t.fields if field.name == "value"]

    if len(value_field) != 1:
        msg = (
            f"Expected structs to have a field named 'value', got {t.fields}."
        )
        raise TypeError(msg)

    # Finally, return the data type of the "value" field.
    return value_field[0].dtype


def get_time_series_time_type(t: pl.DataType) -> PolarsDataType:
    """Returns the Polars data type of the time in a time-series type.

    Args:
        t: The Polars data type.

    Returns:
        The time-series type of the data type.
    """
    # Unpack the time-series structure.
    # First check if the data type is a list...
    if not isinstance(t, pl.List):
        msg = f"Expected a List data type, got {t}."
        raise TypeError(msg)
    t = cast("pl.List", t)

    # ... of structs.
    if not isinstance(t.inner, pl.Struct):
        msg = f"Expected a List of Structs, got a List of {t}."
        raise TypeError(msg)
    t = cast("pl.Struct", t.inner)

    # Then check if the struct has a field named "time".
    time_field = [field for field in t.fields if field.name == "time"]

    if len(time_field) != 1:
        msg = f"Expected structs to have a field named 'time', got {t.fields}."
        raise TypeError(msg)

    # Finally, return the data type of the "time" field.
    return time_field[0].dtype
