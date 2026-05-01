"""
HaltonGenerator
================
Halton low-discrepancy sequence for Quasi-Monte Carlo simulation.

The Halton sequence uses the radical inverse function in base p:
  Phi_p(n) = sum_k d_k * p^{-(k+1)}
where n = sum_k d_k * p^k  is the base-p representation of n.

For base 2, this is the Van der Corput sequence.
Different prime bases give different dimensions of a multi-dimensional QMC grid.

Properties:
  - Deterministic (same start → same sequence)
  - Fills [0,1] more uniformly than pseudo-random
  
"""

from .uniform_generator import UniformGenerator


class HaltonGenerator(UniformGenerator):
    """
    Halton sequence generator.

    Parameters
    ----------
    base : int
        Prime base (e.g., 2, 3, 5, 7, ...). Default: 2.
    start : int
        Starting index in the sequence. Default: 1 (skip index 0 = 0.0).
    """

    def __init__(self, base: int = 2, start: int = 1):
        if base < 2:
            raise ValueError("Base must be >= 2 (preferably prime).")
        super().__init__()
        self._base = base
        self._index = start

    def generate(self) -> float:
        """Return the next value in the Halton sequence."""
        result = self._radical_inverse(self._index)
        self._index += 1
        return result

    def _radical_inverse(self, n: int) -> float:
        """Compute the radical inverse of n in the given base."""
        f = 1.0
        r = 0.0
        while n > 0:
            f /= self._base
            r += f * (n % self._base)
            n //= self._base
        return r

    def reset(self, start: int = 1):
        """Reset the sequence to a given starting index."""
        self._index = start

    @property
    def base(self) -> int:
        return self._base
