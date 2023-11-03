# Installation

It is recommended to use `AGenC` in a virtual environment.
Consult the python [documentation](https://docs.python.org/3/library/venv.html) for further information.

Install the AGenC package:

```bash
python -m pip install -e .
```

For the full functioning experience, AGenC splits its features into optional dependency packages.
Have a look at `pyproject.toml` and respectively install additional dependency groups.

The package installs a command line tool called `agenc`.
Run `agenc --help` to verify successful installation and see the available commands.

## Building the documentation

To build the documentation, you need the `[doc]` feature:

```bash
python -m pip install -e .[doc]
```

The documentation is managed by [Sphinx](https://www.sphinx-doc.org/en/master/).
To build a local html version of the documentation:

```bash
cd docs/
make html
```

The configuration is afterwards generated to `docs/build/html/`.
