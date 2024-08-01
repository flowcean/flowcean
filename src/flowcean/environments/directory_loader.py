from collections.abc import Callable, Iterable
from pathlib import Path

from flowcean.core.environment.offline import OfflineEnvironment

LoaderFunction = Callable[[Path], OfflineEnvironment]


def load_directory(
    path: Path,
    load_function: LoaderFunction,
    *,
    pattern: str = "*",
    include_files: bool = True,
    include_folders: bool = True,
) -> Iterable[OfflineEnvironment]:
    """Load environments from a directory.

    This helper function can be used to load environments from multiple files
    or folders. First all files and folders in the `path` matching the
    `pattern` are selected. These are then passed to the `load_function`
    which in turn creates the actual environment.

    Args:
        path: The path to the directory from which environments are loaded.
        load_function: Function handle to a function loading an environment
            from a path and returning it.
        pattern: A glob pattern. Matching files and folders will be passed to
            `load_function`.
        include_files: Specify whether to pass files to the `load_function`.
        include_folders: Specify whether to pass folders to the
            `load_function`.
    """
    for item in path.glob(pattern):
        if (item.is_file() and include_files) or (
            item.is_dir() and include_folders
        ):
            environment = load_function(item)
            if environment is not None:
                yield environment
