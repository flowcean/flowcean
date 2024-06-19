"""Hybrid System.

This module provides the definition of a hybrid system, which is a dynamical
system that can switch between different modes of operation. Each mode is
defined by a differential equation and a transition function that determines
the next mode based on a current input.
"""

from abc import abstractmethod
from collections.abc import Callable, Iterator, Sequence
from typing import Self, override

import polars as pl

from flowcean.core.environment.incremental import IncrementalEnvironment
from flowcean.environments.ode_environment import OdeSystem, State


class DifferentialMode[X: State, Input](OdeSystem[X]):
    """Differential mode of a hybrid system.

    This class represents a mode of a hybrid system by extending an OdeSystem
    with a transition function that determines the next mode.
    """

    @abstractmethod
    def transition(
        self,
        i: Input,
    ) -> "DifferentialMode[X, Input]":
        """Transition to the next mode.

        Determine the next mode based on the current input. This method should
        return the current mode if no transition is needed.

        Args:
            i: Input.

        Returns:
            Next mode.
        """


class HybridSystem[X: State, Input](IncrementalEnvironment):
    """Hybrid system environment.

    This environment generates samples by simulating a hybrid system. The
    system is defined by a set of differential modes and a sequence of inputs
    that determine the transitions between the modes.
    """

    def __init__(
        self,
        initial_mode: DifferentialMode[X, Input],
        inputs: Iterator[tuple[float, Input]],
        map_to_dataframe: Callable[
            [Sequence[float], Sequence[Input], Sequence[X]],
            pl.DataFrame,
        ],
    ) -> None:
        """Initialize the hybrid system.

        Args:
            initial_mode: Initial mode of the system.
            inputs: Timeseries of inputs (time, input).
            map_to_dataframe: Function to map times, inputs and states to a
                DataFrame.
        """
        self.initial_mode = initial_mode
        self.inputs = inputs
        self.map_to_dataframe = map_to_dataframe

    @override
    def load(self) -> Self:
        return self

    @override
    def __iter__(self) -> Iterator[pl.DataFrame]:
        mode = self.initial_mode
        last_t = 0.0
        for t, i in self.inputs:
            dt = t - last_t
            last_t = t
            ts, states = mode.step(dt)
            mode = mode.transition(i)
            yield self.map_to_dataframe(ts, [i] * len(ts), states)
