---
icon: lucide/workflow
---

# Hybrid Systems

This page covers the public API of `flowcean.hybrid` and its subpackages:

- [Modeling and simulation](#flowcean.hybrid): system definitions, automaton diagrams, simulation, trace I/O, and plotting.
- [Benchmarks](#flowcean.hybrid.benchmarks): reusable system factories, input streams, and registry metadata. Import a factory or use `registry()` to discover the available systems.
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

::: flowcean.hybrid.benchmarks.wind_turbine.wind_turbine
    options:
      heading_level: 3

::: flowcean.hybrid.hydra
    options:
      heading: HyDRA identification
      toc_label: HyDRA identification
