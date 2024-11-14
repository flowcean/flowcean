# Installation

This page explains how to setup the python development environment for flowcean.

Start with cloning the repository:

```bash
git clone https://github.com/flowcean/flowcean
cd flowcean
```

Continue with installation of dependencies.

=== "via `uv`"

    Use `uv` to update the project's environment and install the `flowcean` package:

    ```bash
    uv sync
    ```

=== "via `pip`"

    Create a new virtual environment:

    ```bash
    python -m venv <path-to-venv>
    ```

    Activate the virtual environment:

    ```bash
    . <path-to-venv>
    ```

    Ensure you are in your virtual environment.

    ```bash
    echo $VIRTUAL_ENV
    ```

    This should show the path to an active virtual environment.

    Then, you can install the `flowcean` package in editable mode (allowing for modifications to the source code) using pip:

    ```bash
    pip install -e .
    ```

    Confirm the installation:

    ```bash
    pip show flowcean
    ```

    This command should display package information verifying that `flowcean` is installed.

## Verifying Installation

To verify that `flowcean` is installed correctly, open a Python shell

=== "via `uv`"

    ```bash
    uv run python
    ```

=== "via `pip`"

    ```bash
    . <path-to-venv>
    python
    ```

and try importing the package:

```python
import flowcean
```

If no errors occur, the installation was successful.

## Getting Started

After installation, you can begin using Flowcean in your projects.
For initial usage, refer to the [New Project Guide](new_project.md) to explore how flowcean is used for your project.
