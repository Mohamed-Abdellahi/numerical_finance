"""
DiscreteGenerator
=================
Abstract base class for discrete random variable generators.
Holds a reference to an underlying UniformGenerator (same pattern as ContinuousGenerator).

"""

from abc import abstractmethod
from .random_generator import RandomGenerator
from .uniform_generator import UniformGenerator


class DiscreteGenerator(RandomGenerator):
    """
    Abstract base for discrete distribution generators.

    Parameters
    ----------
    target_mean : float
        Theoretical mean of the distribution.
    target_variance : float
        Theoretical variance.
    uniform : UniformGenerator or None
        The underlying uniform generator. Can be None for degenerate cases (HeadTail).
    """

    def __init__(
        self,
        target_mean: float = 0.0,
        target_variance: float = 0.0,
        uniform: UniformGenerator = None,
    ):
        super().__init__(target_mean=target_mean, target_variance=target_variance)
        if uniform is not None and not isinstance(uniform, UniformGenerator):
            raise TypeError("uniform must be an instance of UniformGenerator.")
        self._uniform = uniform

    @abstractmethod
    def generate(self) -> float:
        """Generate a single sample from the discrete distribution."""
        ...

    @property
    def uniform(self) -> UniformGenerator:
        return self._uniform
