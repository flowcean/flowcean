from typing import NoReturn


def raise_for_missing_optional_dependency(
    error: ModuleNotFoundError,
    *,
    extra: str,
    module: str,
    missing_dependencies: set[str],
) -> NoReturn:
    dependency = error.name or extra
    top_level_dependency = dependency.split(".", maxsplit=1)[0]

    if top_level_dependency not in missing_dependencies:
        raise error

    message = (
        f"`{module}` requires the optional `{extra}` extra "
        f"(`{dependency}` is missing). Install it with "
        f"`pip install flowcean[{extra}]` or `uv sync --extra {extra}`."
    )
    optional_error = ModuleNotFoundError(message)
    optional_error.name = top_level_dependency
    raise optional_error from error
