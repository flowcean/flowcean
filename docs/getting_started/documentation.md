# Building the Documentation

The documentation is managed by [MkDocs](https://www.mkdocs.org/) using the [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) theme.

=== "Using `nox`"

    To build the documentation, you can use the [`nox` automation tool](https://nox.thea.codes/) and run the `docs` session.

    ```sh
    nox --session docs -- build
    ```

    `nox` automatically creates a virtual environment and installs the necessary dependencies to build the documentation.
    The configuration is subsequently generated to `site`.

    To view a live preview of the documentation run

    ```sh
    nox --session docs -- serve
    ```

=== "Manual"

    First, install the necessary dependencies of the `docs` feature:

    ```sh
    pip install -e .[doc]
    ```

    To build a local html version of the documentation:

    ```sh
    mkdocs build
    ```

    To view a live preview of the documentation run

    ```sh
    mkdocs serve
    ```
