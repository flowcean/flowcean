from collections.abc import Callable, Iterable
from pathlib import Path

from flowcean.core.environment.offline import OfflineEnvironment


def load_directory(
    path: Path,
    load_function: Callable[[Path], OfflineEnvironment],
    pattern: str = "*",
    *,
    include_files: bool = True,
    include_folders: bool = True,
) -> Iterable[OfflineEnvironment]:
    for item in path.glob(pattern):
        if (item.is_file() and include_files) or (
            item.is_dir() and include_folders
        ):
            yield load_function(item)
