import logging

class TestLogger:
    """
    A utility class for structured logging during the test input generation process.

    Features:
    - Logs messages to a file.
    - Supports logging levels: info, debug, warning, and error.

    Args:
        logger: Test Logger
            Logger used to log the test input generation process.
    """


    def __init__(self, log_file='testinput_generation.log'):
        """
        Initializes the logger and attaches a file handler.

        Args:
            log_file : Path to the log file where messages will be saved.
        """
        self.logger = logging.getLogger('TestInputGenerator')
        self.logger.setLevel(logging.DEBUG)

        # Avoid adding duplicate handlers
        if not self.logger.hasHandlers():
            # Create a file handler
            file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)

            # Define log message format
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)

            # Attach the handler to the logger
            self.logger.addHandler(file_handler)

    def log_info(self, message):
        """Logs an informational message."""
        self.logger.info(message)

    def log_debug(self, message):
        """Logs a debug-level message."""
        self.logger.debug(message)

    def log_warning(self, message):
        """Logs a warning message."""
        self.logger.warning(message)

    def log_error(self, message):
        """Logs an error message."""
        self.logger.error(message)
