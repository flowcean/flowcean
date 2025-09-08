from abc import ABC, abstractmethod

from .data import Data


class Adapter(ABC):
    """Abstract base class for adapters."""

    @abstractmethod
    def start(self) -> None:
        """Start the adapter.

        This method is called when the tool loop is started.
        It should be used to initialize the adapter, start any
        background processes and establish connections to the CPS.
        """

    @abstractmethod
    def stop(self) -> None:
        """Stop the adapter.

        This method is called when the tool loop is stopped.
        It should be used to clean up resources, stop any background
        processes and close connections to the CPS.
        """

    @abstractmethod
    def get_data(self) -> Data:
        """Get data from the CPS through the adapter.

        Retrieve a data record from the CPS. This method should block until
        data is available. If no more data is available, it should raise a
        Stop exception.

        Returns:
            The data retrieved from the CPS.
        """

    @abstractmethod
    def send_data(self, data: Data) -> None:
        """Send data to the CPS through the adapter.

        This method allows sending data to the CPS. It is used by the tool loop
        to send the results for the tool evaluation back to the CPS for further
        processing.

        Args:
            data: The data to send.
        """
