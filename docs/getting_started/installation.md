# Installation

It is recommended to use `AGenC` in a virtual environment.
Consult the python [documentation](https://docs.python.org/3/library/venv.html) for further information.

Install the AGenC package:

=== "Windows"

    ```powershell
    PS > python -m pip install -e .
    ```

=== "Linux"

    ```sh
    $ python -m pip install -e .
    ```

For the full functioning experience, AGenC splits its features into optional dependency packages.
Have a look at `pyproject.toml` and respectively install additional dependency groups.

## Building the documentation

The documentation is managed by [MkDocs](https://www.mkdocs.org/).

To build the documentation, you can use the `nox` automation tool and run the `docs` session.

```sh
nox --session docs
```

Nox automatically creates a virtual environment and installs the necessary dependencies to build the documentation.
The configuration is afterwards generated to `site`.

### Manually

Alternatively, you can build the documentation manually.
First, install the necessary dependencies of the `docs` feature:

```sh
python -m pip install -e .[doc]
```

To view a live preview of the documentation run
```sh
mkdocs serve
```
and open [the documentation](http://127.0.0.1:8000/) in your browser.

To build a local html version of the documentation:

```sh
mkdocs build
```
