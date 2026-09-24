# Changelog

This changelog records notable user-facing changes to Flowcean. Its format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and version identifiers follow [PEP 440](https://peps.python.org/pep-0440/).

## [Unreleased]

### Added

- Added `build_hybrid_system_dot` and optional Graphviz SVG rendering via `render_dot_svg` to `flowcean.hybrid` for visualizing complete hybrid automata without simulation.
- Added `plot_locations` to `flowcean.hybrid` for reusable location shading on custom time-series plots, including shared-legend support.
- Added a wind-driven, six-state turbine benchmark with polynomial aerodynamics, five hysteretic torque-control regimes, pitch control, and generator-power reporting to `flowcean.hybrid.benchmarks`, the hybrid systems gallery, and a standalone simulation example.
- Added a physical hysteretic buck-converter benchmark with switch-on, diode-conduction, and zero-current modes to `flowcean.hybrid.benchmarks` and the hybrid systems gallery.
- Exposed DDTIG support types from `flowcean.testing.generator` and `AdaBoost` from `flowcean.sklearn` ([#424](https://github.com/flowcean/flowcean/pull/424)).
- Added `ActiveMetric` to `flowcean.core`, `SACLearner` and `SACModel` to `flowcean.palaestrai`, and the explicit `flowcean.sklearn.metrics` facade.
- Added `flowcean.aalpy` learners and models for passive RPNI learning of Mealy and Moore machines ([#419](https://github.com/flowcean/flowcean/pull/419)).
- Added the "Flowcean - Model Learning for Cyber-Physical Systems" paper to the website's citation list ([#404](https://github.com/flowcean/flowcean/pull/404)).
- Added reusable hybrid-system benchmarks under `flowcean.hybrid.benchmarks` ([#407](https://github.com/flowcean/flowcean/pull/407)).

### Changed

- Redesigned the documentation landing page with a thermostat simulation replay, grouped navigation, and page icons.
- Combined the hybrid-system API, benchmarks, and HyDRA reference into `reference/hybrid/`. The former `reference/hybrid/benchmarks/` and `reference/hybrid/hydra/` URLs are no longer available.
- **Breaking:** `ToTimeSeries(time_feature, *, name="time_series")` now collects non-time columns into one named series in a single row, preserving dtypes and source order. One value column stays scalar; multiple value columns form a struct preserving their names and order; no value columns produce empty structs. Clock mappings and automatic per-signal output columns are no longer supported; select per-clock or per-signal branches, transform each separately, and combine the results horizontally.
- Updated dependencies in response to security audit findings and adapted the PalaestrAI SAC learner to the current sensor and actuator API ([#405](https://github.com/flowcean/flowcean/pull/405)).
- Hybrid-system definitions, simulation, trace conversion, and plotting now use the `flowcean.hybrid` namespace ([#407](https://github.com/flowcean/flowcean/pull/407)).
- Reworked the tank benchmark to use gravity-drained outlet dynamics. The `outflow_1` and `outflow` parameters are replaced by keyword-only `outlet_area_1` and `outlet_area_2`, closed operation now has wet and dry modes, and the defaults and simulation horizon have changed ([#418](https://github.com/flowcean/flowcean/pull/418)).
- Hybrid benchmark and identification APIs are kept under `flowcean.hybrid.benchmarks` and `flowcean.hybrid.hydra` rather than duplicated in `flowcean.hybrid`; selector-specific APIs are also available from `flowcean.hybrid.hydra.selector` ([#407](https://github.com/flowcean/flowcean/pull/407)).
- Hybrid trace events now expose independent `state_before` and `state_after` snapshots plus a zero-based `microstep`; these replace the ambiguous `Event.state` field.
- Hybrid transitions now expose explicit exact-zero entry policies, and simulation restarts from post-jump states at the exact event time.

### Fixed

- Hybrid trace shading now follows recorded transition times, including locations visited between samples, instead of leaving gaps at sample boundaries.
- Adaptive and fixed-grid hybrid traces now consistently report the final post-transition state and location at jump boundaries.

### Removed

- Removed `flowcean.grpc.GrpcPassiveAutomataLearner` and its Java LearnLib, gRPC, protobuf, and Docker integration; use `flowcean.aalpy` for local passive automata learning ([#419](https://github.com/flowcean/flowcean/pull/419)).
- Removed the external Polyfill.io script from the documentation site ([#403](https://github.com/flowcean/flowcean/pull/403)).
- Removed the `flowcean.ode` and top-level `flowcean.hydra` namespaces ([#407](https://github.com/flowcean/flowcean/pull/407)).
- Removed the legacy `OdeEnvironment`, `OdeState`, and `OdeSystem` abstractions ([#407](https://github.com/flowcean/flowcean/pull/407)).

## [0.8.0] - 2026-05-10

Changelog tracking begins with changes made after this release.

[Unreleased]: https://github.com/flowcean/flowcean/compare/v0.8.0...HEAD
[0.8.0]: https://github.com/flowcean/flowcean/releases/tag/v0.8.0
