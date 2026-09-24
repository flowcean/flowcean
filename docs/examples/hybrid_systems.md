---
icon: lucide/gallery-horizontal-end
---

# Hybrid Systems Gallery

Hybrid benchmarks illustrate different kinds of continuous dynamics, switching, and resets. Use the comparison table to choose a model, then inspect the illustrated scenarios below.

A **model** defines the state, locations, dynamics, and transitions. A **scenario** chooses a model configuration, input signal, and simulation interval. The diagrams show the complete declared automata; the plots show genuine Flowcean simulations with the stated scenario settings.

## Choose a System

Counts describe the configurations in [`scenarios.py`](https://github.com/flowcean/flowcean/blob/main/examples/hybrid_systems/scenarios.py). States include explicit clock variables; locations are declared locations, not only those visited in a particular run.

| System | States | Locations | External input | Behavior to explore |
| --- | ---: | ---: | --- | --- |
| [Bouncing ball](#bouncing-ball) | 2 | 1 | None | Velocity resets at impact |
| [Thermostat](#thermostat) | 1 | 2 | Target temperature | Hysteresis around a moving target |
| [Hybrid oscillator](../reference/hybrid.md#flowcean.hybrid.benchmarks.hybrid_oscillator.hybrid_oscillator) | 2 | 2 | None | Side-dependent damping |
| [Switched linear](../reference/hybrid.md#flowcean.hybrid.benchmarks.switched_linear.switched_linear) | 2 | 2 | None | State-triggered switching of linear dynamics |
| [Relay integrator](../reference/hybrid.md#flowcean.hybrid.benchmarks.relay_integrator.relay_integrator) | 1 | 2 | None | Relay control with hysteresis |
| [Time-varying event surface](../reference/hybrid.md#flowcean.hybrid.benchmarks.time_varying_event_surface.time_varying_event_surface) | 2 | 2 | Moving threshold | Switching at an externally driven boundary |
| [Time-forced switch](../reference/hybrid.md#flowcean.hybrid.benchmarks.time_forced_switch.time_forced_switch) | 3 | 2 | None | Periodic switching with a clock state |
| [Piecewise affine](../reference/hybrid.md#flowcean.hybrid.benchmarks.piecewise_affine.piecewise_affine) | 2 | 2 | None | Affine dynamics and a linear event surface |
| [Impact oscillator](../reference/hybrid.md#flowcean.hybrid.benchmarks.impact_oscillator.impact_oscillator) | 2 | 1 | Applied force | Forced oscillation with impact resets |
| [PID-controlled plant](../reference/hybrid.md#flowcean.hybrid.benchmarks.pid_controlled_plant.pid_controlled_plant) | 3 | 3 | Reference and reference rate | Actuator saturation and integral control |
| [Tank valves](../reference/hybrid.md#flowcean.hybrid.benchmarks.tank_valves.tank_valves) | 2 | 3 | None | Valve switching and gravity-driven drainage |
| [Location cycle](../reference/hybrid.md#flowcean.hybrid.benchmarks.mode_cycle.mode_cycle) | 4 | 6 | None | A scalable cycle with clock resets |
| [Buck converter](../reference/hybrid.md#flowcean.hybrid.benchmarks.buck_converter.buck_converter) | 2 | 3 | None | Hysteretic switching and diode blocking |
| [Wind turbine](#wind-turbine) | 6 | 5 | Wind speed | Torque regimes and pitch control |

The location-cycle scenario uses six locations, three dynamical coordinates, and one clock state. Other factory arguments can produce different dimensions and location counts.

## Thermostat

A heater switches on and off around a target temperature. Heat loss to the surroundings continues in both locations. The input moves the switching thresholds; it does not directly change the heating or cooling dynamics.

**Model.** The continuous state is temperature $T$. With the default parameters,

$$
\dot{T} =
\begin{cases}
-0.3(T - 20) + 5, & \text{heating}, \\
-0.3(T - 20), & \text{cooling}.
\end{cases}
$$

For target temperature $r(t)$, `too_hot` is a rising zero crossing of $T-(r+1)$ and `too_cold` is a falling zero crossing of $T-(r-1)$. Switching changes the derivative, not the temperature: neither transition has a reset.

<figure class="hybrid-figure hybrid-automaton" markdown="span">

[![Thermostat automaton: heating switches to cooling at too_hot; cooling switches to heating at too_cold.](../assets/hybrid_systems/thermostat-automaton-light.svg#only-light)](../assets/hybrid_systems/thermostat-automaton-light.svg){ target="_blank" rel="noopener" }
[![Thermostat automaton: heating switches to cooling at too_hot; cooling switches to heating at too_cold.](../assets/hybrid_systems/thermostat-automaton-dark.svg#only-dark)](../assets/hybrid_systems/thermostat-automaton-dark.svg){ target="_blank" rel="noopener" }

<figcaption>The incoming arrow marks the initial heating location. Arrow labels identify the event surface and accepted crossing direction. Open either figure to inspect it at full size.</figcaption>

</figure>

**Shown scenario.** The initial temperature is 20, and the target is $r(t)=22+0.8\sin(0.7t)$. The simulation spans 0 to 10 with a sampling interval of 0.02. Temperature and time are shown in the model's chosen units; the factory does not prescribe a unit system.

<figure class="hybrid-figure" markdown="span">

[![Thermostat temperature, moving target, and upper and lower switching thresholds, with heating and cooling intervals shaded.](../assets/hybrid_systems/thermostat-trace-light.svg#only-light)](../assets/hybrid_systems/thermostat-trace-light.svg){ target="_blank" rel="noopener" }
[![Thermostat temperature, moving target, and upper and lower switching thresholds, with heating and cooling intervals shaded.](../assets/hybrid_systems/thermostat-trace-dark.svg#only-dark)](../assets/hybrid_systems/thermostat-trace-dark.svg){ target="_blank" rel="noopener" }

<figcaption>The temperature remains continuous at each switch. The two thresholds form a hysteresis band rather than a demand to track the target exactly. Mode colors match the automaton.</figcaption>

</figure>

Reproduce these figures from the repository root:

```bash
uv run --directory ./examples/hybrid_systems python gallery_assets.py --profile thermostat
```

See the [`thermostat` factory](../reference/hybrid.md#flowcean.hybrid.benchmarks.thermostat.thermostat) for parameter options. The [minimal hand-built thermostat](hs_simple.md) is a different, constant-rate teaching model.

## Bouncing Ball

A hybrid system can jump without changing its location. This ball has only one location, `flight`; impact applies a velocity reset and returns to that same location.

**Model.** The state is height and vertical velocity, $[h,v]$. Between impacts,

$$
\dot{h}=v, \qquad \dot{v}=-g.
$$

The `ground` event detects a falling zero crossing of $h$. The `bounce` reset leaves height unchanged and maps velocity to $v^+=-e v^-$, where $e$ is the restitution coefficient. There is no resting location in this model.

<figure class="hybrid-figure hybrid-automaton" markdown="span">

[![Bouncing-ball automaton: one flight location with a ground-triggered self-loop and bounce reset.](../assets/hybrid_systems/bouncing_ball-automaton-light.svg#only-light)](../assets/hybrid_systems/bouncing_ball-automaton-light.svg){ target="_blank" rel="noopener" }
[![Bouncing-ball automaton: one flight location with a ground-triggered self-loop and bounce reset.](../assets/hybrid_systems/bouncing_ball-automaton-dark.svg#only-dark)](../assets/hybrid_systems/bouncing_ball-automaton-dark.svg){ target="_blank" rel="noopener" }

<figcaption>A self-loop represents a discrete reset within one location, not an additional flight mode.</figcaption>

</figure>

**Shown scenario.** Interpreting height in metres and time in seconds, the ball starts at $[h,v]=[1,0]$ with $g=9.81$ and $e=0.8$. The simulation spans 0 to 3 seconds, sampled every 0.005 seconds.

<figure class="hybrid-figure" markdown="span">

[![Bouncing-ball height and velocity over three seconds; velocity has instantaneous jumps at the six impacts.](../assets/hybrid_systems/bouncing_ball-trace-light.svg#only-light)](../assets/hybrid_systems/bouncing_ball-trace-light.svg){ target="_blank" rel="noopener" }
[![Bouncing-ball height and velocity over three seconds; velocity has instantaneous jumps at the six impacts.](../assets/hybrid_systems/bouncing_ball-trace-dark.svg#only-dark)](../assets/hybrid_systems/bouncing_ball-trace-dark.svg){ target="_blank" rel="noopener" }

<figcaption>Bounce heights decrease as each reset reduces the velocity magnitude. Dashed vertical lines join the recorded pre- and post-impact velocities at the same physical time; they are not continuous motion through those values.</figcaption>

</figure>

```bash
uv run --directory ./examples/hybrid_systems python gallery_assets.py --profile bouncing_ball
```

See the [`bouncing_ball` factory](../reference/hybrid.md#flowcean.hybrid.benchmarks.bouncing_ball.bouncing_ball) for gravity, restitution, and initial-state options.

## Wind Turbine

This already-running turbine couples rotor motion, tower motion, and blade-pitch control. Five discrete locations select different generator torque laws; the continuous states are unchanged when the controller switches location.

**Model.** The six states are rotor speed, tower displacement and velocity, blade pitch and pitch rate, and the pitch controller's integral contribution. The input is wind speed. The five torque regimes progress from no generation through linear and quadratic torque laws to rated mechanical power. Pitch control operates in every regime.

<figure class="hybrid-figure hybrid-automaton hybrid-automaton-tall" markdown="span">

[![Wind-turbine automaton: five torque-control locations connected in both directions, from no generation to rated power.](../assets/hybrid_systems/wind_turbine-automaton-light.svg#only-light)](../assets/hybrid_systems/wind_turbine-automaton-light.svg){ target="_blank" rel="noopener" }
[![Wind-turbine automaton: five torque-control locations connected in both directions, from no generation to rated power.](../assets/hybrid_systems/wind_turbine-automaton-dark.svg#only-dark)](../assets/hybrid_systems/wind_turbine-automaton-dark.svg){ target="_blank" rel="noopener" }

<figcaption>Each adjacent pair has separate upward and downward generator-speed thresholds. Labels speed_1 through speed_4 identify the four boundaries; hysteresis retains the current location between its thresholds.</figcaption>

</figure>

**Shown scenario.** Wind follows $w(t)=11-4\cos(2\pi t/120)$ m/s: a smooth 7-to-15-to-7 m/s cycle. The initial rotor speed is 0.65 rad/s, the other five states are zero, and the initial location is `no_generation`. The simulation spans 120 seconds, sampled every 0.1 seconds.

<figure class="hybrid-figure" markdown="span">

[![Wind speed, rotor speed, blade pitch, tower displacement, and generator mechanical power during a 120-second wind cycle, with five controller modes distinguished by shading.](../assets/hybrid_systems/wind_turbine-trace-light.svg#only-light)](../assets/hybrid_systems/wind_turbine-trace-light.svg){ target="_blank" rel="noopener" }
[![Wind speed, rotor speed, blade pitch, tower displacement, and generator mechanical power during a 120-second wind cycle, with five controller modes distinguished by shading.](../assets/hybrid_systems/wind_turbine-trace-dark.svg#only-dark)](../assets/hybrid_systems/wind_turbine-trace-dark.svg){ target="_blank" rel="noopener" }

<figcaption>The same mode colors apply across all panels and the automaton. The dashed line marks rated generator-shaft power, about 5.30 MW. This is mechanical power, not electrical output, and the reference is not a hard instantaneous cap in other modes.</figcaption>

</figure>

The model assumes quasi-steady, head-on aerodynamics and a running rotor. It does not model startup, shutdown, or emergency braking. Aerodynamic-domain violations stop simulation rather than extrapolating the fitted coefficients. See the [wind-turbine reference](../reference/hybrid.md#flowcean.hybrid.benchmarks.wind_turbine.wind_turbine) for equations, controller parameters, and operating limits.

```bash
uv run --directory ./examples/hybrid_systems python gallery_assets.py --profile wind_turbine
```

The standalone example also prints mode changes and saves a PNG:

```bash
uv run --directory ./examples/hybrid_systems python wind_turbine.py
```

Its output is `examples/hybrid_systems/outputs/wind_turbine.png`.

## Run the Gallery

Run every configured scenario and produce the combined `examples/hybrid_systems/outputs/benchmarks.png`:

```bash
uv run --directory ./examples/hybrid_systems python run.py
```

The terminal summary reports tags, observed locations, state dimension, sample count, event count, and description. This command uses the simulator's adaptive output times; the illustrated profiles above use explicit sampling intervals.

To work with an installed Flowcean package rather than the repository scripts, start with the [simulation example](../user_guide/hybrid_systems.md#simulation). The [identification walkthrough](simulated_hybrid_system.md) shows how to learn a hybrid model from simulated traces.

## Export Automaton Diagrams

Export every scenario's declared automaton without simulating it:

```bash
uv run --directory ./examples/hybrid_systems python export_graphs.py
```

This writes one DOT file per scenario under `examples/hybrid_systems/outputs/automata/`, using lowercase names with spaces replaced by underscores. Add `--svg` to render SVG files; this requires Graphviz's `dot` executable on `PATH`.

See [Automaton Diagrams](../user_guide/hybrid_systems.md#automaton-diagrams) for the export API and label options.

## Reproduce the Documentation Figures

The illustrated profiles use the model configurations and input functions in `scenarios.py`, with `rtol=1e-7`, `atol=1e-9`, and the sampling intervals stated above. `gallery_assets.py` adds presentation settings without redefining the models or input signals.

With [Graphviz](https://graphviz.org/download/) installed, regenerate the light and dark SVGs:

```bash
uv run --directory ./examples/hybrid_systems python gallery_assets.py
```

The default destination is `docs/assets/hybrid_systems/`, independent of the working directory. Use `--output-dir PATH` to write elsewhere, or `--check` to compare freshly rendered figures with the existing files without replacing them.

Regenerate figures when a model, scenario, or plotting setting changes. Output is repeatable within the same rendering toolchain; Graphviz versions and installed fonts can change layout. Ordinary documentation builds use the committed SVGs and require neither Graphviz nor running the simulations.
