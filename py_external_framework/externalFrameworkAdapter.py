import grpc
import pandas
from externalFramework_pb2_grpc import ExternalLearnerStub
from externalFramework_pb2 import *

class ExternalFrameworkAdapter():
    
    stub = None

    def __init__(self) -> None:
        self.channel = grpc.insecure_channel("localhost:50051") # TODO: Make the address an parameter
        self.stub = ExternalLearnerStub(self.channel)

    def train_learner(self, data_package: DataPackage) -> LearnerData:
        learner_data = self.stub.TrainLearner(data_package)
        return learner_data

    def run_learner(self, data: DataRow) -> DataRow:
        return self.stub.RunLearner(data)

if __name__ == "__main__":
    external_framework = ExternalFrameworkAdapter()

    # TODO: This is just mockup for now...
    # Load the data
    data = pandas.read_csv(".\\accelerometer.csv")

    # Create the data package
    data_package = DataPackage()
    data_package.name = "Accelerometer"

    # Define the metadata
    for i in range(len(data.columns)):
        row_name = data.columns[i]
        column = data[row_name]
        entry = ColumnMetaData(name=row_name, lowerBound=min(column), upperBound=max(column), type=(COLUMNTYPE_INPUT if i<= 5 else COLUMNTYPE_TARGET))
        data_package.metaData.extend([entry])

    # Create the main data structure
    for index, row in data.iterrows():
        data = DataRow()
        data.data.extend(row.values)
        data_package.data.extend([data])

    # Send the data package to the learner
    learner_data = external_framework.train_learner(data_package)

    # Eval the learner on one datum
    data = DataRow([])
    result = external_framework.run_learner(data)