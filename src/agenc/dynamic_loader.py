from importlib import import_module
from typing import Any


def load_instance(class_path: str, kwargs: dict | None) -> Any:
    """Dynamically load a class with arguments and keyword arguments.

    Args:
        class_path: The path to the class to load.
        kwargs: The keyword arguments to pass to the class constructor.

    Returns:
        An instance of the class.
    """
    class_module, class_name = class_path.rsplit(".", 1)
    module = import_module(class_module)
    class_reference = getattr(module, class_name)

    return class_reference(**kwargs)
