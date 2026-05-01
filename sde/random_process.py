"""
RandomProcess
=============
Abstract base class for all stochastic processes.


  - Holds a RandomGenerator* (the source of randomness)
  - Holds a list of SinglePath* (one per dimension)
  - Dimension (number of assets / components)
  - Simulate(start, end, nb_steps) → fills all Paths

"""

from abc import ABC, abstractmethod
from typing import List

from .single_path import SinglePath
from random_generator.random_generator import RandomGenerator


class RandomProcess(ABC):
    """
    Abstract base class for all stochastic processes.

    Parameters
    ----------
    generator : RandomGenerator
        The random number generator used as the noise source.
    dimension : int
        Number of components (e.g., 1 for 1D, N for N-asset basket).
    """

    def __init__(self, generator: RandomGenerator, dimension: int = 1):
        if not isinstance(generator, RandomGenerator):
            raise TypeError("generator must be an instance of RandomGenerator.")
        if dimension < 1:
            raise ValueError("Dimension must be >= 1.")

        self._generator: RandomGenerator = generator
        self._dimension: int = dimension
        self._paths: List[SinglePath] = []  # one SinglePath per dimension

    @abstractmethod
    def simulate(self, start_time: float, end_time: float, nb_steps: int) -> None:
        """
        Simulate the process over [start_time, end_time] using nb_steps steps.
        Fills self._paths with simulated trajectories.

        Parameters
        ----------
        start_time : float
            Start of the simulation interval.
        end_time : float
            End of the simulation interval (e.g., maturity T).
        nb_steps : int
            Number of discretization steps.
        """
        ...

    def get_path(self, dimension: int = 0) -> SinglePath:
        """
        Return the simulated path for a given dimension.

        Parameters
        ----------
        dimension : int
            Index of the dimension (0-based). Default: 0.

        Returns
        -------
        SinglePath
            The path for the requested dimension.
        """
        if not self._paths:
            raise RuntimeError("No paths available. Call simulate() first.")
        if dimension < 0 or dimension >= self._dimension:
            raise IndexError(
                f"Dimension index {dimension} out of range [0, {self._dimension - 1}]."
            )
        return self._paths[dimension]

    def get_all_paths(self) -> List[SinglePath]:
        """Return all simulated paths (one per dimension)."""
        if not self._paths:
            raise RuntimeError("No paths available. Call simulate() first.")
        return self._paths

    def _init_paths(self, start_time: float, end_time: float, nb_steps: int) -> None:
        """
        Initialize (or reset) the path list with fresh SinglePath objects.
        Called at the beginning of simulate().
        """
        self._paths = [
            SinglePath(start_time, end_time, nb_steps)
            for _ in range(self._dimension)
        ]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def generator(self) -> RandomGenerator:
        return self._generator

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def paths(self) -> List[SinglePath]:
        return self._paths
