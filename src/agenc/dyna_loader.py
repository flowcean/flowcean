from importlib import import_module


def load_function(import_str):
    mod, func = import_str.rsplit(".", 1)
    mod = import_module(mod)
    func = getattr(mod, func)

    return func
