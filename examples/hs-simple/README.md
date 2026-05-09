# Minimal Hybrid System Example

This example builds a minimal two-location thermostat hybrid system with the
object-based simulation API. The reusable `build_thermostat()` function returns
a plain `HybridSystem` assembled from `ContinuousDynamics`, `Location`,
`EventSurface`, and `Transition` objects.

- `heating` increases the scalar temperature.
- `cooling` decreases the scalar temperature.
- `too_hot` switches from `heating` to `cooling` at the upper threshold.
- `too_cold` switches from `cooling` to `heating` at the lower threshold.
