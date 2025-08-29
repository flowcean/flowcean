import logging
import time
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, cast

import polars as pl
from opcua import Client, Node, Subscription, ua
from opcua.client.client import KeepAlive
from ruamel.yaml import YAML

from flowcean.core.adapter import Adapter
from flowcean.core.data import Data

logger = logging.getLogger(__name__)

_type_mapping_pl2ua = {
    pl.Float32: ua.VariantType.Float,
    pl.Float64: ua.VariantType.Double,
    pl.Int32: ua.VariantType.Int32,
    pl.Int64: ua.VariantType.Int64,
    pl.Boolean: ua.VariantType.Boolean,
}

_type_mapping_str2pl = {
    "int32": pl.Int32,
    "int64": pl.Int64,
    "float32": pl.Float32,
    "float64": pl.Float64,
    "bool": pl.Boolean,
}


class StreamingHandler:
    """Handler for monitoring changes in the streaming flag node."""

    _is_streaming: bool

    def __init__(self) -> None:
        self._is_streaming = False

    def is_streaming(self) -> bool:
        return self._is_streaming

    def datachange_notification(self, _: Node, val: Any, __: Any) -> None:
        self._is_streaming = val


class OPCAdapter(Adapter):
    """Flowcean adapter for OPC (Open Platform Communications) protocol."""

    opc_client: Client
    pull_frequency: float
    recorded_data: pl.DataFrame
    input_schema: dict[str, Any]
    pre_capture_window_length: timedelta

    input_features: dict[str, Node]
    output_features: dict[str, Node]

    streaming_flag_node: Node
    connection_flag_node: Node
    prediction_flag_node: Node

    streaming_handler: StreamingHandler
    streaming_sub: Subscription

    def __init__(
        self,
        config_path: str | Path,
        *,
        pull_frequency: float = 150.0,
    ) -> None:
        r"""Initialize the OPC adapter.

        Initializes a new OPC to Flowcean adapter instance.
        The adapter reads its configuration from a YAML file.
        The path to this file is provided in the config_path argument and the
        file has the following structure:
        ```
        # Path to the OPC Server
        server-url: "opc.tcp://127.0.0.1:4840"

        # Specify the time window to record before the recording flag is set.
        # This enables the capture and processing of data that triggers the
        # recording flag.
        pre_capture_window_length: 2

        # Define Flowcean inputs which their respective feature name, the
        # opc-id and the datatype of the feature.
        # All inputs will be passed on to the Flowcean model when triggering an
        # inference.
        # Available data types are: float32, float64, int32, int64 and bool.
        inputs:
            - feature: "feature_a"
              opc-id: "<OPC-ID String>"
              type: "int32"
            - feature: "feature_b"
              opc-id: "<OPC-ID String>"
              type: "float64"
            - ...

        # Define the outputs from the Flowcean Model to the OPC Server.
        # All outputs need to be mapped to on OPC field for the adapter to work
        # correctly. If certain outputs are not needed / used, those should be
        # dropped using transforms. The available datatypes are the same as for
        # the input features.
        outputs:
            - feature: "result_feature"
            opc-id: "<OPC-ID String>"
            type: "int32"

        # Flags are special OPC fields, which are used to communicate between
        # the adapter and the OPC server.

        # The streaming flag is set by the OPC server to `True` when Flowcean
        # should start to record data. Once set back to `False`, the data is
        # forwarded to the model, an inference step is executed and the results
        # are send back to the OPC server.
        stream_flag: "<OPC-ID String>"

        # The prediction flag is set to `True` by the adapter when it
        # successfully sends data to the OPC server.
        # This can be used on the OPC side to process the newly received data.
        # The flag is *not* reset by the adapter, so it will remain `True`
        # until the OPC server resets it.
        prediction_flag: "<OPC-ID String>"

        # The connection flag is set to `True` by the adapter when it
        # successfully connects to the OPC server and set to `False` when it
        # disconnects from the server.
        # This can be used to monitor the connection status of the adapter from
        # the OPC side.
        connection_flag: "<OPC-ID String>"
        ```

        Args:
            config_path: Path to the YAML configuration file containing
                    the OPC server URL and feature definitions.
            pull_frequency: Frequency in Hz at which to poll the OPC server
                for new data.
        """
        super().__init__()

        self.pull_frequency = pull_frequency

        # Load the configuration from the provided path
        with Path(config_path).open() as yaml_file:
            yaml_data = YAML(typ="safe").load(yaml_file)

        # Initialize the adapter fields
        self.input_features = {}
        self.output_features = {}

        schema: dict[str, Any] = {}

        # Initialize the OPC client
        self.client = Client(yaml_data["server-url"])
        self.client.connect()
        self.client.load_type_definitions()

        # Start a keep-alive mechanism to ensure the connection stays alive
        self.client.keepalive = KeepAlive(
            self.client,
            min(
                self.client.session_timeout,
                self.client.secure_channel_timeout,
            )
            * 0.7,
        )
        self.client.keepalive.start()

        # Get inputs from the YAML configuration
        # Each input has a feature name, an opc-id, and a type
        for feature in yaml_data["inputs"]:
            self.input_features[feature["feature"]] = self.client.get_node(
                feature["opc-id"],
            )
            # Build the input schema from opc to flowcean
            schema[feature["feature"]] = _type_mapping_str2pl[feature["type"]]

        for feature in yaml_data["outputs"]:
            self.output_features[feature["feature"]] = self.client.get_node(
                feature["opc-id"],
            )

        streaming_flag_opc_id = yaml_data["stream_flag"]
        connection_flag_opc_id = yaml_data["connection_flag"]
        prediction_flag_opc_id = yaml_data["prediction_flag"]
        self.pre_capture_window_length = timedelta(
            seconds=yaml_data["pre_capture_window_length"],
        )

        # Setup the remaining opc nodes
        self.streaming_flag_node = self.client.get_node(
            streaming_flag_opc_id,
        )
        self.connection_flag_node = self.client.get_node(
            connection_flag_opc_id,
        )
        self.prediction_flag_node = self.client.get_node(
            prediction_flag_opc_id,
        )

        # Initialize the recording DataFrame with the derived schema
        self.input_schema = schema.copy()
        # Add a special column for the recorded time - this column will
        # only be used internally and dropped before returning the data
        # to flowcean
        schema["_recorded_time"] = pl.Datetime
        self.recorded_data = pl.DataFrame(schema=schema)

    def start(self) -> None:
        # Setup a streaming handler to process incoming data
        self.streaming_handler = StreamingHandler()
        self.streaming_sub = self.client.create_subscription(
            10,
            self.streaming_handler,
        )
        self.streaming_sub.subscribe_data_change(self.streaming_flag_node)
        # Set the connection flag to True to indicate successful connection
        self.connection_flag_node.set_attribute(
            ua.AttributeIds.Value,
            ua.DataValue(
                ua.Variant(
                    value=True,
                    varianttype=ua.VariantType.Boolean,
                ),
            ),
        )
        logger.info("OPC client successfully connected to server")

    def stop(self) -> None:
        self.streaming_sub.delete()
        # Set the connection flag to False to indicate disconnection
        self.connection_flag_node.set_attribute(
            ua.AttributeIds.Value,
            ua.DataValue(
                ua.Variant(
                    value=False,
                    varianttype=ua.VariantType.Boolean,
                ),
            ),
        )
        if self.client.keepalive is not None:
            self.client.keepalive.stop()
        self.client.disconnect()
        logger.info("OPC client disconnected from server")

    def get_data(self) -> Data:
        """Get data from the OPC server and return it to flowcean."""
        self.recorded_data.clear()

        # We have to wait until the streaming flag is set to True
        # by the CPS/OPC server before we can start recording data.

        # First we wait until the streaming flag is set to True
        # by the CPS/OPC server.
        prerecording_in_progress = True

        def stream_data() -> bool:
            nonlocal prerecording_in_progress
            # Check if we are done with pre-recording and streaming
            if (
                not prerecording_in_progress
                and not self.streaming_handler.is_streaming()
            ):
                return False

            # Record a sample of data
            results = self.client.get_values(self.input_features.values())
            self.recorded_data = pl.concat(
                [
                    self.recorded_data,
                    pl.DataFrame(
                        [results],
                        schema=self.input_schema,
                        orient="row",
                    ).with_columns(
                        _recorded_time=pl.lit(datetime.now(timezone.utc)).cast(
                            pl.Datetime,
                        ),
                    ),
                ],
            )

            # Check if we are still pre-recording
            if self.streaming_handler.is_streaming():
                prerecording_in_progress = False
            else:
                # Still pre-recording
                # Discard old data that is older than `capture_time`.
                self.recorded_data = self.recorded_data.filter(
                    pl.col("_recorded_time")
                    >= pl.lit(datetime.now(timezone.utc)).cast(pl.Datetime)
                    - self.pre_capture_window_length,
                )

            return True

        # Run the `stream_data` function in a timed loop
        OPCAdapter._timed_loop(stream_data, self.pull_frequency)

        # Get rid of any duplicates and the `_recorded_time` feature column
        # in the recorded data
        return (
            self.recorded_data.lazy()
            .drop("_recorded_time")
            .unique(
                maintain_order=True,
            )
        )

    def send_data(self, data: pl.DataFrame | pl.LazyFrame) -> None:
        """Send data to the OPC server.

        Send a polars DataFrame or LazyFrame to the OPC server.
        The data must contain all required output features, otherwise
        a ValueError is raised.

        Args:
            data: Polars DataFrame or LazyFrame containing the data to send.
        """
        # Convert the data into a Dataframe
        df = cast("pl.DataFrame | pl.LazyFrame", data).lazy().collect()

        # Check if the data contains all required output features
        if not all(feature in df.columns for feature in self.output_features):
            missing_features = [
                feature
                for feature in self.output_features
                if feature not in df.columns
            ]
            msg = (
                f"Data is missing required output features: "
                f"{', '.join(missing_features)}"
            )
            raise ValueError(msg)

        # Send the data to the OPC server
        data_dict = df.row(0, named=True)
        for feature_name, node in self.output_features.items():
            # Get the value for the feature from the DataFrame
            value = data_dict[feature_name]

            # Set the nodes value
            node.set_attribute(
                ua.AttributeIds.Value,
                ua.DataValue(
                    ua.Variant(
                        value=value,
                        varianttype=_opc_from_polars(df.schema[feature_name]),
                    ),
                ),
            )

        # Set the prediction flag to True to indicate that new data has been
        # sent
        self.prediction_flag_node.set_attribute(
            ua.AttributeIds.Value,
            ua.DataValue(
                ua.Variant(
                    value=True,
                    varianttype=ua.VariantType.Boolean,
                ),
            ),
        )

    @staticmethod
    def _timed_loop(
        func: Callable[[], bool],
        target_frequency: float,
    ) -> None:
        """Run a function in a loop with a target frequency.

        This method runs the provided function in a timed loop, ensuring that
        it executes at the specified target frequency.
        This method is blockingly executed until the function returns False and
        the loop stops.

        Args:
            func: Callback function to run in the loop. Returns a boolean
                indicating whether to continue looping.
            target_frequency: Target frequency in Hz (times per second) at
                which to run the function.
        """
        start_time = time.time()
        while True:
            if not func():
                break
            elapsed_time = time.time() - start_time
            sleep_time = max(0, (1 / target_frequency) - elapsed_time)
            time.sleep(sleep_time)
            start_time = time.time()


def _opc_from_polars(t: pl.DataType) -> ua.VariantType:
    """Get the OPC UA type corresponding to a Polars data type.

    Args:
        t: Polars data type.

    Returns:
        The corresponding OPC UA type.
    """
    possible_types = [
        opc_type
        for pl_type, opc_type in _type_mapping_pl2ua.items()
        if t.is_(pl_type)
    ]
    if len(possible_types) == 0:
        msg = "No matching OPC UA type found for Polars type: {t}"
        raise ValueError(msg)
    return possible_types[0]
