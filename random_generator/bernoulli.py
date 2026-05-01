"""
Bernoulli Generator
===================
Generates Bernoulli(p) random variables: returns 1 with probability p, 0 otherwise.

Mean = p, Variance = p*(1-p).

Reference: Generator Architecture PDF (Kaiza Amouh)
"""

from .discrete_generator import DiscreteGenerator
from .uniform_generator import UniformGenerator


class Bernoulli(DiscreteGenerator):
    """
    Bernoulli(p) generator.

    Parameters
    ----------
    p : float
        Success probability (0 < p < 1).
    uniform : UniformGenerator
        Underlying uniform generator.
    """

    def __init__(self, p: float, uniform: UniformGenerator):
        if not (0.0 < p < 1.0):
            raise ValueError("Probability p must be in (0, 1).")
        super().__init__(
            target_mean=p,
            target_variance=p * (1.0 - p),
            uniform=uniform,
        )
        self._p = p

    def generate(self) -> float:
        """Return 1 with probability p, 0 with probability 1-p."""
        return 1.0 if self._uniform.generate() < self._p else 0.0

    @property
    def p(self) -> float:
        return self._p
