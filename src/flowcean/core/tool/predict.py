from flowcean.core.adapter import Adapter
from flowcean.core.environment.incremental import Finished
from flowcean.core.model import Model
from flowcean.core.transform import Identity, Transform


def start_prediction_loop(
    model: Model,
    adapter: Adapter,
    *,
    adapter_to_model_transforms: Transform | None = None,
    model_to_adapter_transforms: Transform | None = None,
) -> None:
    """Start a prediction loop with the given model and adapter."""
    adapter_to_model_transforms = adapter_to_model_transforms or Identity()
    model_to_adapter_transforms = model_to_adapter_transforms or Identity()

    # Start the adapter
    adapter.start()

    # Run the prediction loop
    try:
        while True:
            # Get data from the adapter
            data = adapter.get_data()
            # Transform the data to the model format
            transformed_data = adapter_to_model_transforms(data)
            # Run the model prediction
            prediction = model.predict(transformed_data)
            # Transform the prediction to the adapter format
            transformed_prediction = model_to_adapter_transforms(
                prediction,
            )
            # Send the prediction to the adapter
            adapter.send_data(transformed_prediction)
    except Finished:
        pass
    except KeyboardInterrupt:
        pass
    finally:
        # Stop the adapter
        adapter.stop()
