from collections.abc import Callable, Iterable
from pathlib import Path

from tqdm import tqdm

from flowcean.core.environment.offline import OfflineEnvironment

EnvironmentBuilder = Callable[[Path], OfflineEnvironment]


def build_environments_from_directory(
    path: Path | str,
    builder: EnvironmentBuilder,
    *,
    pattern: str = "*",
    include_files: bool = True,
    include_folders: bool = True,
) -> Iterable[OfflineEnvironment]:
    """Build environments from a directory.

    This helper function can be used to build environments from multiple files
    or folders. First all files and folders in the `path` matching the
    `pattern` are selected. These are then passed to the `builder` function
    which creates the environment.

    Args:
        path: The path to the directory from which environments are created.
        builder: A function building an environment from a path and returning
            it.
        pattern: A glob pattern. Matching files and folders will be passed to
            `builder`.
        include_files: Specify whether to pass files to `builder`.
        include_folders: Specify whether to pass folders to `builder`.
    """
    files = list(Path(path).glob(pattern))
    for p in tqdm(files, desc="Building environments"):
        if (p.is_file() and include_files) or (p.is_dir() and include_folders):
            yield builder(p)
