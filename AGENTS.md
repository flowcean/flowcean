# AGENTS.md

## Workflow
- Use `uv` and `just` as the source of truth. CI runs `just check`, `just test`, `just package`, `just docs`, and selected `just examples-<name>` targets.
- `just check` = `uv lock --locked`, `uv run pre-commit run --all-files`, `uv run --all-packages --all-extras basedpyright`, `uv run deptry src`.
- `just test` = `uv run python -m pytest tests --cov --cov-config=pyproject.toml`.
- Focused test: `uv run pytest tests/path/test_file.py -v` or `uv run pytest tests/path/test_file.py::test_name -v`.
- Focused example: `uv run --directory ./examples/<name>/ run.py`. Do not assume every workspace example also has a `just examples-<name>` target; `passive_circuit` does not.

## CI And Generated Outputs
- PR CI has five jobs in `.github/workflows/ci.yml`: checks, tests, package, docs, and an examples matrix.
- `just docs` runs `uv run --only-group docs zensical build --strict` using `zensical.toml` and writes `site/`. `just docs-serve` starts the preview server. Documentation dependencies are isolated in the `docs` dependency group; no Java toolchain is needed.
- API reference pages in `docs/reference/` document public package exports with mkdocstrings. Keep their links and the explicit navigation in `zensical.toml` in sync when adding public packages.
- Examples with DVC-tracked data need `uv run dvc pull --recursive examples/<name>` before running locally. The Coffee Machine example is not in the CI examples matrix because it requires this external data.

## Repo Shape
- `src/flowcean/__init__.py` is empty; the usable public API is exposed from subpackages like `flowcean.core`, `flowcean.polars`, and backend packages.
- `src/flowcean/core/` is the main wiring layer: shared abstractions, callbacks, and `learn_offline` / `learn_incremental` / `learn_active`.
- `src/flowcean/core/strategies/offline.py` is the clearest end-to-end reference for the offline learn/evaluate flow.
- `src/flowcean/polars/` owns dataframe environments and most transforms. Backend-specific learners/models live in sibling packages such as `sklearn/`, `torch/`, `river/`, `xgboost/`, `pysr/`, `hydra/`, and `aalpy/`.
- Callback helpers are intentionally importable from both `flowcean.core.callbacks` and `flowcean.core`. `get_default_callbacks()` returns `[]`, so learners stay silent unless callbacks are passed explicitly.

## Gotchas
- Do not assume every `examples/*` directory is wired the same way. If an example uses `flowcean = { workspace = true }`, keep `[tool.uv.workspace].members` in sync, and update `justfile` plus the CI examples matrix if it should be runnable there.
- Pytest stability for real PySR tests depends on `tests/conftest.py` setting `PYTHON_JULIACALL_THREADS=1`; keep that in place unless you have a verified replacement.
- Project-local worktrees belong under `.worktrees/`, which is already ignored in `.gitignore`.
