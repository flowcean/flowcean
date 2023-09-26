from pathlib import Path
from urllib.parse import urlparse


def file_uri_to_path(uri: str, root: Path) -> Path:
    url = urlparse(uri)
    if url.scheme != "file":
        raise ValueError(
            "only local files are supported as data source, but got:"
            f" `{url.scheme}`",
        )
    data_source = Path(url.path)
    if not data_source.is_absolute():
        data_source = (root / data_source).absolute()
    return data_source
