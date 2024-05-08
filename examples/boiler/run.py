#!/usr/bin/env python

from boiler import Boiler, Heating
from matplotlib import pyplot as plt


def main() -> None:
    boiler = Boiler(
        initial_mode=Heating(temperature=0.0, timeout=0.0),
        sampling_time=0.1,
    )
    states = boiler.simulate(1000)
    plt.plot(states)
    plt.show()


if __name__ == "__main__":
    main()
