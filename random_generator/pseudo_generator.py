"""
PseudoGenerator
================
Abstract base for pseudo-random number generators with internal state.

Extends UniformGenerator with:
  - A seed (initial state)
  - A current value (mutable state that advances on each call)
  - A reset() method

Concrete subclasses: LinearCongruential, EcuyerCombined.
"""

from abc import abstractmethod
from .uniform_generator import UniformGenerator


class PseudoGenerator(UniformGenerator):
    """
    Abstract base for pseudo-random generators that maintain internal state.

    Parameters
    ----------
    seed : int
        Initial seed value. Must be strictly positive.
    """

    def __init__(self, seed: int = 1):
        super().__init__()
        if seed <= 0:
            raise ValueError("Seed must be strictly positive.")
        self._seed: int = seed
        self._current: int = seed

    def reset(self):
        """Reset the generator to its initial seed."""
        self._current = self._seed

    @abstractmethod
    def generate(self) -> float:
        ...

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def current(self) -> int:
        return self._current
