"""
ContinuousGenerator
===================
Abstract base class for continuous random variable generators.
Holds a reference to an underlying UniformGenerator (composition).
All continuous generators are built on top of a uniform source.

"""

from abc import abstractmethod
from .random_generator import RandomGenerator
from .uniform_generator import UniformGenerator


class ContinuousGenerator(RandomGenerator):
    """
    Abstract base for continuous distribution generators.

    Parameters
    ----------
    target_mean : float
        Theoretical mean of the distribution.
    target_variance : float
        Theoretical variance of the distribution.
    uniform : UniformGenerator
        The underlying uniform random number generator to use.
    """

    def __init__(
        self,
        target_mean: float,
        target_variance: float,
        uniform: UniformGenerator,
    ):
        super().__init__(target_mean=target_mean, target_variance=target_variance)
        if not isinstance(uniform, UniformGenerator):
            raise TypeError("uniform must be an instance of UniformGenerator.")
        self._uniform = uniform

    @abstractmethod
    def generate(self) -> float:
        """Generate a single sample from the continuous distribution."""
        ...

    @property
    def uniform(self) -> UniformGenerator:
        """The underlying uniform generator."""
        return self._uniform
