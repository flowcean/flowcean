from importlib import import_module
from typing import Any, Callable, Optional


def load_function(function_path: str) -> Callable:
    """Dynamically load a function without calling it."""
    function_module, function_name = function_path.rsplit(".", 1)
    module = import_module(function_module)

    return getattr(module, function_name)


def load_class(
    class_path: str, *args: Optional[Any], **kwargs: Optional[Any]
) -> Any:
    """Dynamically load a class with arguments and keyword arguments."""
    class_module, class_name = class_path.rsplit(".", 1)
    module = import_module(class_module)
    class_reference = getattr(module, class_name)

    return class_reference(*args, **kwargs)
