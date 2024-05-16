#!/usr/bin/env python

from boiler import Heating, Temperature, randomly_changing_values, simulate
from matplotlib import pyplot as plt


def main() -> None:
    target_temperatures = list(
        randomly_changing_values(
            n=1000,
            change_probability=0.002,
            minimum=30.0,
            maximum=60.0,
        )
    )
    states = simulate(
        initial_state=Temperature(30.0),
        initial_mode=Heating(t=0.0),
        inputs=iter(target_temperatures),
        sampling_time=0.1,
    )
    plt.plot(
        [
            [target, state.temperature]
            for (target, state) in zip(
                target_temperatures,
                states,
                strict=True,
            )
        ]
    )
    plt.show()


if __name__ == "__main__":
    main()
