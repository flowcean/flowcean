# Flowcean

[![Pipeline Status](https://collaborating.tuhh.de/w-6/forschung/agenc/agenc/badges/main/pipeline.svg)](https://collaborating.tuhh.de/w-6/forschung/agenc/agenc/pipelines/main/latest)
![Coverage](https://collaborating.tuhh.de/w-6/forschung/agenc/agenc/badges/main/coverage.svg)

The flowcean framework.

## Building the Documentation

First, install all necessary dependencies to build the documentation.

```bash
pip install --upgrade pip
pip install nox
```

Build the documentation using the nox session.

```bash
nox --session docs -- build
```

After the documentation has been built, you can open it in your browser by opening `site/index.html`.
