import grpc
import numpy as np

from agenc.core import Dataset, Learner
from agenc.learners.external import grpcLearner_pb2
from agenc.learners.external.grpcLearner_pb2_grpc import ExternalLearnerStub


class ExternalGRPCLearner(Learner):
    uri: str = "localhost:8080"

    def __init__(self) -> None:
        super().__init__()
        self.channel = grpc.insecure_channel(self.uri)
        self.stub = ExternalLearnerStub(self.channel)

    def dispose(self) -> None:
        self.channel.close()

    def train(self, inputs: np.ndarray, outputs: np.ndarray) -> None:
        dataset = ExternalGRPCLearner.create_dataset(inputs, outputs)
        status_message: grpcLearner_pb2.StatusMessage
        final_status: grpcLearner_pb2.Status = grpcLearner_pb2.STATUS_UNDEFINED
        for status_message in self.stub.Train(
            ExternalGRPCLearner.dataset_to_proto_datapackage(dataset)
        ):
            for log_message in status_message.messages:
                # TODO: Perform logging of external messages here
                # Till then just print them
                print(
                    "[gRPC Learner]"
                    f" [{ExternalGRPCLearner.loglevel_to_string(log_message.log_level)}]"  # noqa: E501
                    f" {log_message.message}"
                )

            if status_message.status != grpcLearner_pb2.Status.STATUS_RUNNING:
                final_status = status_message.status
                break  # Break the status streaming loop

        if final_status == grpcLearner_pb2.Status.STATUS_FAILED:
            # Training failed gracefully
            # TODO: Run some logging here!
            print(
                "[gRPC Learner] [ERROR] Learning failed. See previous log"
                " messages for more details."
            )
        elif final_status == grpcLearner_pb2.Status.STATUS_FINISHED:
            # Everything went according to plan
            # TODO: Run some logging here!
            print("[gRPC Learner] [INFO] Learning finished sucessfull.")
        else:
            # Something went really wrong
            # TODO: Run some logging here!
            print(
                "[gRPC Learner] [FATAL] Learning failed with unknown error."
                " Check previous log messages."
            )

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        dataset = ExternalGRPCLearner.create_dataset(
            inputs, np.zeros((inputs.shape[0], 1))
        )
        predictions = self.stub.Predict(
            ExternalGRPCLearner.dataset_to_proto_datapackage(dataset)
        )
        return ExternalGRPCLearner.predictions_to_array(predictions)

    @staticmethod
    def create_dataset(inputs: np.ndarray, outputs: np.ndarray) -> Dataset:
        assert inputs.ndim == 2  # for the moment assume tabular data
        input_names = [f"i{i}" for i in range(0, inputs.shape[1])]
        output_names = [f"o{i}" for i in range(0, outputs.shape[1])]

        dataset = Dataset(
            input_names,
            output_names,
            np.concatenate([inputs, outputs], axis=1),
        )
        print(dataset)

        return dataset

    @staticmethod
    def dataset_to_proto_datapackage(
        dataset: Dataset,
    ) -> grpcLearner_pb2.DataPackage:
        metadata: list[grpcLearner_pb2.ColumnMetadata] = []
        observations: list[grpcLearner_pb2.Observation] = []

        input_names = dataset.input_columns
        # TODO: For now assume everything is a scalar!
        metadata.extend(
            [
                grpcLearner_pb2.ColumnMetadata(
                    name=column_name,
                    feature_type=(
                        grpcLearner_pb2.FEATURETYPE_INPUT
                        if (column_name in input_names)
                        else grpcLearner_pb2.FEATURETYPE_TARGET
                    ),
                    data_type=grpcLearner_pb2.DATATYPE_SCALAR,
                )
                for column_name in dataset.input_columns
                + dataset.output_columns
            ]
        )

        observations.extend(
            [
                ExternalGRPCLearner.row_to_proto_observation(row)
                for row in dataset.data
            ]
        )

        return grpcLearner_pb2.DataPackage(
            metadata=metadata, observations=observations
        )

    @staticmethod
    def row_to_proto_observation(
        row: np.ndarray,
    ) -> grpcLearner_pb2.Observation:
        # TODO: Currently assumes all fields are of type double
        # Check for type and support different datatypes in future
        fields: list[grpcLearner_pb2.ObservationField] = [
            grpcLearner_pb2.ObservationField(double=entry) for entry in row
        ]

        # TODO: Currently assumes no time dependency is present
        return grpcLearner_pb2.Observation(fields=fields)

    @staticmethod
    def predictions_to_array(
        predictions: grpcLearner_pb2.Prediction,
    ) -> np.ndarray:
        data: list[list[float]] = []  # Only supports double scalars for now

        for prediction in predictions.predictions:
            observation = []
            observation.extend(
                [field.double for field in prediction.fields]
            )  # TODO: For now assume only double is used for data transfer!

            data.append(observation)

        return np.array(data)

    @staticmethod
    def loglevel_to_string(loglevel: grpcLearner_pb2.LogLevel) -> str:
        if loglevel == grpcLearner_pb2.LogLevel.LOGLEVEL_DEBUG:
            return "DEBUG"
        if loglevel == grpcLearner_pb2.LogLevel.LOGLEVEL_INFO:
            return "INFO"
        if loglevel == grpcLearner_pb2.LogLevel.LOGLEVEL_WARNING:
            return "WARN"
        if loglevel == grpcLearner_pb2.LogLevel.LOGLEVEL_ERROR:
            return "ERROR"
        if loglevel == grpcLearner_pb2.LogLevel.LOGLEVEL_FATAL:
            return "FATAL"
        return "UNDEF"
