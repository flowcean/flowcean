# Installation

It is recommended to use `AGenC` in a virtual environment.
Consult the python [documentation](https://docs.python.org/3/library/venv.html) for further information.

Install the AGenC package:

```sh
python -m pip install -e .
```

For the full functioning experience, AGenC splits its features into optional dependency packages.
Have a look at `pyproject.toml` and respectively install additional dependency groups.

## Building the documentation

The documentation is managed by [Sphinx](https://www.sphinx-doc.org/en/master/).

To build the documentation, you can use the `nox` automation tool and run the `docs` session.

```sh
nox --session docs
```

Nox automatically creates a virtual environment and installs the necessary dependencies to build the documentation.
The configuration is afterwards generated to `docs/build/html/`.

### Manually

Alternatively, you can build the documentation manually.
First, install the necessary dependencies of the `docs` feature:

```sh
python -m pip install -e .[doc]
```

To build a local html version of the documentation:

```sh
cd docs/
make html
```
