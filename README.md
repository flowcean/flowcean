# AGenC

[![Pipeline Status](https://collaborating.tuhh.de/w-6/forschung/agenc/agenc/badges/main/pipeline.svg)](https://collaborating.tuhh.de/w-6/forschung/agenc/agenc/pipelines/main/latest)
![Coverage](https://collaborating.tuhh.de/w-6/forschung/agenc/agenc/badges/main/coverage.svg)

The AGenC framework.

## Building the Documentation

First, install all necessary dependencies to build the documentation.

```bash
pip install --upgrade pip
pip install nox
```

Build the documentation using the nox session.

```bash
nox --session docs
```

After the documentation has been built, you can open it in your browser by opening `docs/_build/html/index.html`.

Building the documentation again can be sped up by using the `-r` flag.
```bash 
nox --session docs -r
```

The rest of the CI/CD pipeline can be checked locally using the command below.
```bash
nox --session
```
