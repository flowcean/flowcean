# AGENTS.md

## Development Policy

- Flowcean is in 0.x development and currently has no downstream consumers. Breaking API changes are expected and permitted. Prefer a clean, coherent design over preserving existing names, signatures, import paths, or behavior.
- Do not add or retain compatibility shims, deprecated aliases, legacy wrappers, or fallback paths solely for backwards compatibility unless explicitly requested. Do not seek additional approval solely because a change breaks an API.
- Keep changes scoped to the task. Update affected in-repository callers, tests, examples, and documentation together; tests should validate the new contract rather than preserve obsolete behavior.
- Revisit the compatibility policy when downstream consumers exist or the project adopts a stable API commitment.

## Working Tree

- Keep the central worktree on `main` and free of implementation changes. Before modifying the repository, create or reuse a task worktree under `.worktrees/` (already ignored by Git).

## Commands and Validation

- Use `uv` for Python commands and `just` for repository workflows. [justfile](justfile) defines the recipes; [.github/workflows/ci.yml](.github/workflows/ci.yml) defines CI coverage.
- Start with focused checks, such as `uv run pytest tests/path/test_file.py -v`, then run the broader checks applicable to the change.
- Available checks are `just check` (style, types, and dependencies), `just test` (test suite), `just package` (package build), and `just docs` (strict documentation build).
- Run affected examples separately where relevant; `just test` does not cover all example behavior. Use the example's `just examples-<name>` recipe when available. Otherwise inspect its own configuration and entry points rather than assuming every example uses `run.py`.

## Cross-File Changes

- Before opening a PR, record notable user-facing changes under `Unreleased` in [CHANGELOG.md](CHANGELOG.md), including breaking changes. Internal-only changes do not normally need an entry.
- Keep public API reference pages in `docs/reference/` and navigation in `zensical.toml` aligned. See [Building the Documentation](docs/getting_started/documentation.md) for the documentation workflow.
- When adding an example that uses `flowcean = { workspace = true }`, register it in `[tool.uv.workspace].members` in `pyproject.toml`. Add or update its `justfile` recipe and CI matrix entry if it should run there; these lists are not necessarily identical.
- Data-backed examples may require a DVC pull and institutional VPN access. See [DVC](docs/getting_started/dvc.md) before running them.
