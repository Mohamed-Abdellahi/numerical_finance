"""
Binomial Generator
==================
Generates Binomial(n, p) random variables: sum of n independent Bernoulli(p).

Mean = n*p, Variance = n*p*(1-p).

"""

from .discrete_generator import DiscreteGenerator
from .uniform_generator import UniformGenerator


class Binomial(DiscreteGenerator):
    """
    Binomial(n, p) generator.
    Simulated as the sum of n independent Bernoulli(p) trials.

    Parameters
    ----------
    n : int
        Number of trials (n >= 1).
    p : float
        Success probability per trial (0 < p < 1).
    uniform : UniformGenerator
        Underlying uniform generator.
    """

    def __init__(self, n: int, p: float, uniform: UniformGenerator):
        if n < 1:
            raise ValueError("Number of trials n must be >= 1.")
        if not (0.0 < p < 1.0):
            raise ValueError("Probability p must be in (0, 1).")
        super().__init__(
            target_mean=n * p,
            target_variance=n * p * (1.0 - p),
            uniform=uniform,
        )
        self._n = n
        self._p = p

    def generate(self) -> float:
        """Return sum of n independent Bernoulli(p) trials."""
        return float(sum(1 for _ in range(self._n) if self._uniform.generate() < self._p))

    @property
    def n(self) -> int:
        return self._n

    @property
    def p(self) -> float:
        return self._p
