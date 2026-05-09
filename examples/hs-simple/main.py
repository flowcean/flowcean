import matplotlib.pyplot as plt
import numpy as np

from flowcean.ode import (
    ContinuousDynamics,
    CrossingDirection,
    EventSurface,
    HybridSystem,
    Location,
    Transition,
    plot_trace,
    simulate,
)


def _heating_flow() -> np.ndarray:
    return np.array([0.8], dtype=float)


def _cooling_flow() -> np.ndarray:
    return np.array([-0.6], dtype=float)


def _too_hot(state: np.ndarray) -> float:
    return float(state[0] - 21.0)


def _too_cold(state: np.ndarray) -> float:
    return float(state[0] - 19.0)


def build_thermostat() -> HybridSystem:
    """Build a minimal two-location thermostat hybrid system."""
    heating = Location(
        ContinuousDynamics(_heating_flow, label="heating_flow"),
        label="heating",
    )
    cooling = Location(
        ContinuousDynamics(_cooling_flow, label="cooling_flow"),
        label="cooling",
    )

    return HybridSystem(
        locations=[heating, cooling],
        transitions=[
            Transition(
                source=heating,
                target=cooling,
                event=EventSurface(
                    _too_hot,
                    direction=CrossingDirection.RISING,
                    label="too_hot",
                ),
            ),
            Transition(
                source=cooling,
                target=heating,
                event=EventSurface(
                    _too_cold,
                    direction=CrossingDirection.FALLING,
                    label="too_cold",
                ),
            ),
        ],
        initial_location=heating,
        initial_state=np.array([19.0], dtype=float),
    )


def main() -> None:
    """Simulate the thermostat and write a plot image."""
    trace = simulate(
        build_thermostat(),
        t_span=(0.0, 20.0),
        sample_dt=0.05,
    )

    _fig, ax = plt.subplots(figsize=(8.0, 3.5), layout="constrained")
    plot_trace(
        trace,
        show_locations=True,
        show_location_labels=True,
        show_events=True,
        ax=ax,
    )
    ax.set_title("Minimal thermostat hybrid system")
    ax.set_ylabel("temperature")

    event_count = len(trace.events)
    print(f"recorded {event_count} events")

    plt.show()


if __name__ == "__main__":
    main()
