# Installation

This page covers installing Flowcean for a project and setting up a local development checkout.

## User Install From PyPI

Use this setup when you want to use Flowcean in your own Python project.

=== "via `uv`"

    Create a project and add Flowcean from PyPI:

    ```bash
    uv init flowcean-project
    cd flowcean-project
    uv add flowcean
    ```

    If you need PySR support, add the PySR extra:

    ```bash
    uv add "flowcean[pysr]"
    ```

=== "via `pip`"

    Create and activate a virtual environment, then install Flowcean from PyPI:

    ```bash
    python -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install flowcean
    ```

    If you need PySR support, install the PySR extra:

    ```bash
    python -m pip install "flowcean[pysr]"
    ```

## Developer Setup From Git

Use this setup when you want to work on Flowcean itself.

```bash
git clone https://github.com/flowcean/flowcean
cd flowcean
uv sync
```

If you cannot use `uv`, create and activate a virtual environment, then install Flowcean in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

`uv sync` is the recommended developer setup because it installs the full workspace and development toolchain. The pip editable fallback supports basic editable imports; you may need additional development tools before running checks, docs, or tests.

## Verifying Installation

Run an import-only smoke test from the environment where you installed Flowcean.

=== "via `uv`"

    From a `uv` project or development checkout, run:

    ```bash
    uv run python - <<'PY'
    import flowcean
    PY
    ```

=== "via `pip`"

    From an activated virtual environment, run:

    ```bash
    python - <<'PY'
    import flowcean
    PY
    ```

If the command exits without an error, Python can import Flowcean.

## Plotting Backend

Many examples can run headless or write plot files without opening a window. If you want interactive Matplotlib windows, your system may need a GUI backend such as PyQt6.

=== "via `uv`"

    ```bash
    uv add PyQt6
    ```

=== "via `pip`"

    ```bash
    python -m pip install PyQt6
    ```

## Getting Started

After installation, start with the [New Project Guide](new_project.md).
