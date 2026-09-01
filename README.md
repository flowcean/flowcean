# Flowcean

[![Actions status](https://github.com/flowcean/flowcean/actions/workflows/ci.yml/badge.svg)](https://github.com/flowcean/flowcean/actions)
[![docs status](https://github.com/flowcean/flowcean/actions/workflows/pages.yml/badge.svg)](https://flowcean.me)
[![PyPI - Version](https://img.shields.io/pypi/v/flowcean)](https://pypi.python.org/pypi/flowcean)
[![License](https://img.shields.io/github/license/flowcean/flowcean)](https://github.com/flowcean/flowcean/blob/main/LICENSE)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/flowcean)](https://pypi.python.org/pypi/flowcean)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Pyright](https://img.shields.io/badge/types-Pyright-blue.svg)](https://github.com/microsoft/pyright)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Flowcean is a research-oriented Python toolkit for defining, simulating, identifying, evaluating, and reproducing models of cyber-physical systems.

Flowcean builds on research into automatic model generation for CPS. Its established environments, transforms, offline, incremental, and active learning strategies, metrics, adapters, and backend integrations remain central to the toolkit. First-class support for hybrid dynamical systems connects these capabilities to explicit system structure and simulation without restricting Flowcean to hybrid models.

## Capabilities

- Define and simulate hybrid systems with continuous dynamics, events, transitions, and resets.
- Identify hybrid dynamics and mode selectors with HyDRA.
- Learn models from offline datasets, incremental streams, and active environments.
- Compose reusable environments, transforms, learners, models, metrics, and evaluation strategies.
- Integrate established Python ML libraries and external learners through backend packages, adapters, and gRPC.

## Installation

```sh
pip install flowcean
```

See the [installation guide](https://flowcean.me/getting_started/installation/) and [hybrid systems guide](https://flowcean.me/user_guide/hybrid_systems/) for the next steps.

## Documentation

Flowcean documentation is available at [flowcean.me](https://flowcean.me). User-visible changes are recorded in the [changelog](https://github.com/flowcean/flowcean/blob/main/CHANGELOG.md).

## Contributing

We welcome open-source contributions that advance CPS modeling research. See the [code of conduct](https://github.com/flowcean/flowcean/blob/main/CODE_OF_CONDUCT.md) before contributing.

## License

Flowcean is licensed under the 3-Clause BSD License ([LICENSE](https://github.com/flowcean/flowcean/blob/main/LICENSE) or [https://opensource.org/license/bsd-3-clause](https://opensource.org/license/bsd-3-clause)).
