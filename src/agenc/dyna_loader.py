from importlib import import_module


def load_function(import_str):
    mod, func = import_str.rsplit(".", 1)
    mod = import_module(mod)
    func = getattr(mod, func)

    return func


def load_class(import_str):
    if ":" in import_str:
        mod, clazz = import_str.rplit(":", 1)
    else:
        mod, clazz = import_str.rsplit(".", 1)

    mod = import_module(mod)
    clazz = getattr(mod, clazz)

    return clazz
