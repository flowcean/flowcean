# Building the Documentation

The documentation is managed by [MkDocs](https://www.mkdocs.org/) using the [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) theme.

To build the documentation, you can use [`just`](https://github.com/casey/just) and use the `docs` recipe.

```sh
just docs
```

`just` invokes `uv` to create a virtual environment and install the necessary dependencies to build the documentation.
The configuration is subsequently generated to `site`.

To view a live preview of the documentation run

```sh
just docs-serve
```
