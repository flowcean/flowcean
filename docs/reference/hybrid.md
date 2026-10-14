---
icon: lucide/workflow
---

# Hybrid Systems

This page covers the public API of `flowcean.hybrid` and its subpackages:

- [Modeling and simulation](#flowcean.hybrid): system definitions, automaton diagrams, trajectories, explicit sampling, and plotting.
- [Benchmarks](#flowcean.hybrid.benchmarks): reusable system factories.
- [HyDRA identification](#flowcean.hybrid.hydra): `HyDRALearner` returns fully assigned `LearnedFlows` or raises `HyDRAIdentificationError` with the failing `segment`. Use `to_labeled_frames` for selector training or `to_flow_ids` for NumPy assignments. Progress observers use `HyDRACallback` or `PlotCallback`; selector helpers are also available from `flowcean.hybrid.hydra.selector`.
- [Dainarx identification](#flowcean.hybrid.dainarx): `fit_dainarx_dynamics` discovers ordinary NARX locations and returns `DainarxDynamics`. Use `to_location_ids()` for NumPy assignments or `to_labeled_frames(original_traces)` to add Int64 `location_id` labels while preserving original columns. Learn guards and post-switch predictors separately with `fit_dainarx_transitions(original_traces, dynamics, ...)`, then construct `DainarxModel(dynamics.locations, transitions)`. `fit_dainarx` composes both stages into a complete fit. See the [Dainarx example](../examples/dainarx.md).

Dainarx annotations describe discovered membership, not ground truth or a per-observation error guarantee. They include segment warmup rows excluded from ordinary location fitting. Retain `segments`: adjacent segments can share a location ID while retaining a detected boundary. Discovery is complete independently of transition fitting; unsupported same-location resets and insufficient transition evidence fail in the transition stage. Both annotation and transition fitting require the original independent-trace order and row correspondence. Geometry checks cannot detect changed observations or reordered same-length traces; transition fitting additionally checks selected numeric columns and the sampling grid.

See the [hybrid systems guide](../user_guide/hybrid_systems.md) for modeling concepts and examples.

<!-- prettier-ignore-start -->
<!-- Mkdocstrings option blocks require indentation that Prettier removes. -->
::: flowcean.hybrid
    options:
      heading: Modeling and simulation
      toc_label: Modeling and simulation

::: flowcean.hybrid.benchmarks
    options:
      heading: Benchmarks
      toc_label: Benchmarks

### System factories

<!-- Explicit directives are needed because the factory names match their submodules. -->

::: flowcean.hybrid.benchmarks.bouncing_ball.bouncing_ball
    options:
      heading_level: 4

::: flowcean.hybrid.benchmarks.buck_converter.buck_converter
    options:
      heading_level: 4

::: flowcean.hybrid.benchmarks.duffing_oscillator.duffing_oscillator
    options:
      heading_level: 4

::: flowcean.hybrid.benchmarks.hybrid_oscillator.hybrid_oscillator
    options:
      heading_level: 4

::: flowcean.hybrid.benchmarks.impact_oscillator.impact_oscillator
    options:
      heading_level: 4

::: flowcean.hybrid.benchmarks.location_cycle.location_cycle
    options:
      heading_level: 4

::: flowcean.hybrid.benchmarks.pid_controlled_plant.pid_controlled_plant
    options:
      heading_level: 4

::: flowcean.hybrid.benchmarks.piecewise_affine.piecewise_affine
    options:
      heading_level: 4

::: flowcean.hybrid.benchmarks.relay_integrator.relay_integrator
    options:
      heading_level: 4

::: flowcean.hybrid.benchmarks.switched_linear.switched_linear
    options:
      heading_level: 4

::: flowcean.hybrid.benchmarks.tank_valves.tank_valves
    options:
      heading_level: 4

::: flowcean.hybrid.benchmarks.thermostat.thermostat
    options:
      heading_level: 4

::: flowcean.hybrid.benchmarks.time_forced_switch.time_forced_switch
    options:
      heading_level: 4

::: flowcean.hybrid.benchmarks.time_varying_event_surface.time_varying_event_surface
    options:
      heading_level: 4

::: flowcean.hybrid.benchmarks.wind_turbine.wind_turbine
    options:
      heading_level: 4

::: flowcean.hybrid.benchmarks.valve_closure.valve_closure
    options:
      heading_level: 4

::: flowcean.hybrid.hydra
    options:
      heading: HyDRA identification
      toc_label: HyDRA identification

::: flowcean.hybrid.dainarx
    options:
      heading: Dainarx identification
      toc_label: Dainarx identification
<!-- prettier-ignore-end -->
