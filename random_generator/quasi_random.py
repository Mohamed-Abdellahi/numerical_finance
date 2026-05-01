"""
Quasi-Random Number Generators
================================
Low-discrepancy sequences for Quasi-Monte Carlo (QMC) simulation.
These are NOT truly random — they fill space more uniformly than
pseudo-random numbers, which reduces Monte Carlo variance.

Implemented:
  - HaltonGenerator  : Halton sequence (base-p radical inverse)
  - SobolGenerator   : Sobol sequence (Niederreiter family, using scipy)

Reference: Slides 3 (Kaiza Amouh), pages 42-50
"""

import math
from typing import Optional
from .uniform_generator import UniformGenerator
from .random_generator import RandomGenerator


class HaltonGenerator(UniformGenerator):
    """
    Halton sequence generator.

    The Halton sequence uses radical inverse in base p:
      Phi_p(n) = sum_{k} d_k * p^{-(k+1)}
    where sum d_k*p^k is the base-p expansion of n.

    For dimension d=1, the base-2 Halton sequence is the Van der Corput sequence.

    Parameters
    ----------
    base : int
        Prime base (e.g., 2, 3, 5, 7, ...). Default: 2.
    start : int
        Starting index. Default: 1 (skip index 0 which gives 0.0).
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


class SobolGenerator(UniformGenerator):
    """
    Sobol sequence generator (1-dimensional).

    Uses scipy.stats.qmc.Sobol for the actual sequence generation.
    Sobol sequences are a special case of Niederreiter sequences in base 2,
    with very low discrepancy.

    Parameters
    ----------
    dimension : int
        Dimension of the Sobol sequence to use (1 for scalar output).
    scramble : bool
        Whether to scramble the sequence (recommended for MC integration).
    seed : int or None
        Random seed for scrambling.
    """

    def __init__(self, dimension: int = 1, scramble: bool = True, seed: Optional[int] = 42):
        super().__init__()
        try:
            from scipy.stats.qmc import Sobol
        except ImportError:
            raise ImportError("scipy is required for SobolGenerator. Install with: pip install scipy")
        self._dimension = dimension
        self._engine = Sobol(d=dimension, scramble=scramble, seed=seed)
        self._buffer = []
        self._buffer_index = 0
        # Generate in batches for efficiency
        self._batch_size = 1024

    def generate(self) -> float:
        """Return the next scalar value from the 1-D Sobol sequence."""
        if self._buffer_index >= len(self._buffer):
            # Generate next batch
            samples = self._engine.random(self._batch_size)
            self._buffer = samples[:, 0].tolist()  # Use first dimension
            self._buffer_index = 0
        value = self._buffer[self._buffer_index]
        self._buffer_index += 1
        return float(value)

    def generate_vector(self, n: int) -> list:
        """
        Generate n samples of the full d-dimensional Sobol vector.

        Parameters
        ----------
        n : int
            Number of samples.

        Returns
        -------
        list of list of float
            Shape (n, dimension).
        """
        return self._engine.random(n).tolist()

    @property
    def dimension(self) -> int:
        return self._dimension
