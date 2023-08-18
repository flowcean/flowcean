# AGenC

[![Pipeline Status](https://collaborating.tuhh.de/w-6/forschung/agenc/agenc/badges/main/pipeline.svg)](https://collaborating.tuhh.de/w-6/forschung/agenc/agenc/pipelines/main/latest)
![Coverage](https://collaborating.tuhh.de/w-6/forschung/agenc/agenc/badges/main/coverage.svg)

The AGenC framework.

## Building the Documentation

First, install all necessary dependencies to build the documentation.

```bash
pip install --upgrade pip
pip install -e .[doc]
```

Build the documentation using sphinx.

```bash
cd docs/
make html
```

After a successful build, the documentation resides in `docs/build/html/`.
