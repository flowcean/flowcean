
import os

import lightning

from agenc.learner import MLP, AccelerometerData
from agenc.metadata import construct_path, read_metadata
from agenc.metrics import mae, rmse


def main():
    # Compose the path to the metadata file
    metadata_path = os.path.abspath(
        os.path.join(__file__, "..", "data", "accelerometer", "metadata.yml")
    )

    # Read Metadata
    metadata = read_metadata(metadata_path)

    train_data_path = construct_path(metadata.uri, metadata.train_data_set_name)
    test_data_path = construct_path(metadata.uri, metadata.test_data_set_name)

    # TODO: Features
    # print(m.data_set_name)

    # Read Data
    # TODO: Use path from meta data
    data = AccelerometerData()

    # Preprocessing
    # TODO: Add/Load preprocessing models based on experiment file

    # Load learner
    # TODO: Load java learner
    # TODO: Load leaner based on experiment file
    model = MLP()
    trainer = lightning.Trainer(max_epochs=5)
    trainer.fit(model, data)

    train_data = data.train_dataloader()
    val_data = data.val_dataloader()
    # test_data = data.test_dataloader()

    # TODO: Get data in the correct format
    train_predict = trainer.predict(model, train_data)
    val_predict = trainer.predict(model, val_data)
    # test_predict = trainer.predict(model, test_data)

    # TODO: Get the predictions and the true values 
    # Apply metric
    print(
        f"Train Error: RMSE={rmse(train_predict, train_data)}"
        f", MAE={mae(train_predict, train_data)}"
    )
    print(
        f"Validation Error: RMSE={rmse(val_predict, val_data)}"
        f", MAE={mae(val_predict, val_data)}"
    )
    # print(
    #     f"Test Error: RMSE={rmse(test_predict, test_data)}"
    #     f", MAE={mae(test_predict, test_data)}"
    # )



if __name__ == "__main__":
    main()
