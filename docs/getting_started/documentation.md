# Building the Documentation

The documentation is built with [Zensical](https://zensical.org/) using its modern theme.
Site settings and navigation are defined in `zensical.toml`.

To build the documentation, you can use [`just`](https://github.com/casey/just) and use the `docs` recipe.

```sh
just docs
```

`just` invokes `uv` to install the dedicated `docs` dependency group and build in strict mode.
The generated site is written to `site/`.

The [API reference](../reference/index.md) uses hand-authored pages in `docs/reference/` and mkdocstrings directives to document public Python package exports.
When adding a reference page, also add it to the navigation in `zensical.toml`.

To view a live preview of the documentation run

```sh
just docs-serve
```
