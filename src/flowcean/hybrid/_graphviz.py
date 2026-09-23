"""Private DOT escaping and Graphviz rendering shared by hybrid graphs."""

import shutil
import subprocess


def _escape_dot_label(label: str) -> str:
    """Normalize line endings and escape text for a quoted DOT label."""
    return (
        label.replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
    )


def render_dot_svg(dot_source: str) -> str:
    """Render DOT text as SVG using Graphviz's external ``dot`` executable.

    No files are written and no viewer is opened. SVG layout can vary between
    Graphviz versions even when the input DOT is identical.

    Args:
        dot_source: DOT source to render.

    Returns:
        SVG text decoded as UTF-8.

    Raises:
        RuntimeError: Graphviz is not installed or rejects the DOT source.
    """
    dot_path = shutil.which("dot")
    if dot_path is None:
        message = (
            "Graphviz 'dot' executable not found; install Graphviz "
            "to render SVG output."
        )
        raise RuntimeError(message)

    result = subprocess.run(  # noqa: S603
        [dot_path, "-Tsvg"],
        input=dot_source,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        message = "Graphviz failed to render SVG"
        if stderr:
            message = f"{message}: {stderr}"
        raise RuntimeError(message)

    return result.stdout
