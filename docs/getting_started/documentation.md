# Building the Documentation

The documentation is managed by [MkDocs](https://www.mkdocs.org/) using the [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) theme.

To build the documentation, you can use the [`hatch` project manager](https://hatch.pypa.io/latest/) and use the `docs` environment.

```sh
hatch run docs:build
```

`hatch` automatically creates a virtual environment and installs the necessary dependencies to build the documentation.
The configuration is subsequently generated to `site`.

To view a live preview of the documentation run

```sh
hatch run docs:serve
```
