"""
UniformGenerator
================
Abstract base class for all U[0,1] generators.
Mean = 0.5, Variance = 1/12.
"""

from abc import abstractmethod
from .random_generator import RandomGenerator


class UniformGenerator(RandomGenerator):
    """
    Abstract base class for uniform U[0,1] generators.
    Mean = 0.5, Variance = 1/12.
    """

    def __init__(self):
        super().__init__(target_mean=0.5, target_variance=1.0 / 12.0)

    @abstractmethod
    def generate(self) -> float:
        """Return a uniform random number in (0, 1)."""
        ...
