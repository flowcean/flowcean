import matplotlib.pyplot as plt
import numpy as np

from flowcean.hybrid import (
    CrossingDirection,
    EventSurface,
    Flow,
    HybridSystem,
    Location,
    Transition,
    plot_trajectory,
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
        Flow(_heating_flow, label="heating_flow"),
        label="heating",
    )
    cooling = Location(
        Flow(_cooling_flow, label="cooling_flow"),
        label="cooling",
    )

    return HybridSystem(
        locations=[heating, cooling],
        transitions=[
            Transition(
                source=heating,
                target=cooling,
                event_surface=EventSurface(
                    _too_hot,
                    direction=CrossingDirection.RISING,
                    label="too_hot",
                ),
            ),
            Transition(
                source=cooling,
                target=heating,
                event_surface=EventSurface(
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
    trajectory = simulate(
        build_thermostat(),
        t_span=(0.0, 20.0),
    )

    _fig, ax = plt.subplots(figsize=(8.0, 3.5), layout="constrained")
    plot_trajectory(
        trajectory,
        show_locations=True,
        show_location_labels=True,
        show_events=True,
        ax=ax,
    )
    ax.set_title("Minimal thermostat hybrid system")
    ax.set_ylabel("temperature")

    event_count = len(trajectory.events)
    print(f"recorded {event_count} events")

    plt.show()


if __name__ == "__main__":
    main()
