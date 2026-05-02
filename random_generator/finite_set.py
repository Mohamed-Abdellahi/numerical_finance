"""
FiniteSet Generator
===================
Generates a discrete random variable taking values {0, 1, ..., n-1}
with prescribed probabilities [p0, p1, ..., p_{n-1}].

Uses the inverse CDF method (cumulative sum of probabilities).

Mean and Variance are derived from the given probability vector.

"""

from typing import List
from .discrete_generator import DiscreteGenerator
from .uniform_generator import UniformGenerator


class FiniteSet(DiscreteGenerator):
    """
    Finite discrete distribution generator.

    Parameters
    ----------
    probabilities : list of float
        Probability vector [p0, p1, ..., p_{n-1}].
        Must be non-negative and sum to 1 (within tolerance).
    uniform : UniformGenerator
        Underlying uniform generator.
    """

    def __init__(self, probabilities: List[float], uniform: UniformGenerator):
        if not probabilities:
            raise ValueError("Probability vector must be non-empty.")
        if any(p < 0 for p in probabilities):
            raise ValueError("All probabilities must be non-negative.")
        total = sum(probabilities)
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"Probabilities must sum to 1, got {total}.")

        # Compute mean and variance: values are 0, 1, ..., n-1
        n = len(probabilities)
        mean = sum(i * probabilities[i] for i in range(n))
        variance = sum((i ** 2) * probabilities[i] for i in range(n)) - mean ** 2

        super().__init__(target_mean=mean, target_variance=variance, uniform=uniform)
        self._probabilities = list(probabilities)
        # Precompute cumulative probabilities for fast lookup
        self._cumulative = []
        cumsum = 0.0
        for p in probabilities:
            cumsum += p
            self._cumulative.append(cumsum)

    def generate(self) -> float:
        """Return a sample from the finite discrete distribution via inverse CDF."""
        u = self._uniform.generate()
        for i, cum_p in enumerate(self._cumulative):
            if u < cum_p:
                return float(i)
        # Fallback for floating-point edge case
        return float(len(self._probabilities) - 1)

    @property
    def probabilities(self) -> List[float]:
        return list(self._probabilities)
