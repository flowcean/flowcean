from flowcean.core.adapter import Adapter
from flowcean.core.environment.incremental import Finished
from flowcean.core.model import Predictor
from flowcean.core.transform import Transform


def start_prediction_loop[Observation, Input, Output, Action](
    model: Predictor[Input, Output],
    adapter: Adapter[Observation, Action],
    *,
    adapter_to_model_transforms: Transform[Observation, Input],
    model_to_adapter_transforms: Transform[Output, Action],
) -> None:
    """Run predictions with explicit representation adapters.

    Pass Identity() for an unchanged representation.
    """
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
