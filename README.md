# Flowcean

[![Pipeline Status](https://collaborating.tuhh.de/w-6/forschung/agenc/agenc/badges/main/pipeline.svg)](https://collaborating.tuhh.de/w-6/forschung/agenc/agenc/pipelines/main/latest)
![Coverage](https://collaborating.tuhh.de/w-6/forschung/agenc/agenc/badges/main/coverage.svg)

The flowcean framework.

## Building the Documentation

First, install hatch.

```bash
pipx install hatch
```

Then, build the documentation:

```bash
hatch run docs:build
```

After the documentation has been built, you can open it in your browser by opening `site/index.html`.

Alternatively, run:

```bash
hatch run docs:serve
```

to get an interactively reloading session of the documentation.
