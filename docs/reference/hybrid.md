---
icon: lucide/workflow
---

# Hybrid Systems

This page covers the public API of `flowcean.hybrid` and its subpackages:

- [Modeling and simulation](#flowcean.hybrid): system definitions, automaton diagrams, simulation, trace I/O, and plotting.
- [Benchmarks](#flowcean.hybrid.benchmarks): reusable system factories.
- [HyDRA identification](#flowcean.hybrid.hydra): identification, callbacks, trace schemas, simulations, and mode-selector APIs. Selector-specific types and helpers are also available from `flowcean.hybrid.hydra.selector`.

See the [hybrid systems guide](../user_guide/hybrid_systems.md) for modeling concepts and examples.

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

::: flowcean.hybrid.benchmarks.hybrid_oscillator.hybrid_oscillator
    options:
      heading_level: 4

::: flowcean.hybrid.benchmarks.impact_oscillator.impact_oscillator
    options:
      heading_level: 4

::: flowcean.hybrid.benchmarks.mode_cycle.mode_cycle
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

::: flowcean.hybrid.hydra
    options:
      heading: HyDRA identification
      toc_label: HyDRA identification
