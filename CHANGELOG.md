# Changelog

This changelog records notable user-facing changes to Flowcean. Its format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and version identifiers follow [PEP 440](https://peps.python.org/pep-0440/).

## [Unreleased]

### Added

- Added the "Flowcean - Model Learning for Cyber-Physical Systems" paper to the website's citation list ([#404](https://github.com/flowcean/flowcean/pull/404)).
- Added reusable hybrid-system benchmarks under `flowcean.hybrid.benchmarks` ([#407](https://github.com/flowcean/flowcean/pull/407)).

### Changed

- Updated dependencies in response to security audit findings and adapted the PalaestrAI SAC learner to the current sensor and actuator API ([#405](https://github.com/flowcean/flowcean/pull/405)).
- Hybrid-system definitions, simulation, trace conversion, and plotting now use the `flowcean.hybrid` namespace ([#407](https://github.com/flowcean/flowcean/pull/407)).
- HyDRA is now available from `flowcean.hybrid.hydra` and through the `flowcean.hybrid` public API ([#407](https://github.com/flowcean/flowcean/pull/407)).
- Hybrid trace events now expose independent `state_before` and `state_after` snapshots plus a zero-based `microstep`; these replace the ambiguous `Event.state` field.

### Fixed

- Adaptive and fixed-grid hybrid traces now consistently report the final post-transition state and location at jump boundaries.

### Removed

- Removed the external Polyfill.io script from the documentation site ([#403](https://github.com/flowcean/flowcean/pull/403)).
- Removed the `flowcean.ode` and top-level `flowcean.hydra` namespaces ([#407](https://github.com/flowcean/flowcean/pull/407)).
- Removed the legacy `OdeEnvironment`, `OdeState`, and `OdeSystem` abstractions ([#407](https://github.com/flowcean/flowcean/pull/407)).

## [0.8.0] - 2026-05-10

Changelog tracking begins with changes made after this release.

[Unreleased]: https://github.com/flowcean/flowcean/compare/v0.8.0...HEAD
[0.8.0]: https://github.com/flowcean/flowcean/releases/tag/v0.8.0
