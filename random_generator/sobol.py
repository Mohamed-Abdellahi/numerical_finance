"""
SobolGenerator
===============
Sobol low-discrepancy sequence for Quasi-Monte Carlo simulation.

Sobol sequences are a special case of Niederreiter sequences in base 2.
They have extremely low discrepancy and are widely used in financial QMC.

Requires scipy (pip install scipy).

Properties:
  - Much lower discrepancy than pseudo-random or Halton for high dimensions
  - Scrambling (optional) adds randomness while preserving low discrepancy
  - For best properties, use n = power of 2
"""

from typing import Optional, List
from .uniform_generator import UniformGenerator


class SobolGenerator(UniformGenerator):
    """
    Sobol sequence generator using scipy.stats.qmc.Sobol.

    Parameters
    ----------
    dimension : int
        Dimension of the Sobol sequence (1 for scalar output).
    scramble : bool
        Whether to scramble the sequence. Recommended for MC integration.
    seed : int or None
        Random seed for scrambling reproducibility.
    """

    def __init__(
        self,
        dimension: int = 1,
        scramble: bool = True,
        seed: Optional[int] = 42,
    ):
        super().__init__()
        try:
            from scipy.stats.qmc import Sobol
        except ImportError:
            raise ImportError(
                "scipy is required for SobolGenerator. "
                "Install with: pip install scipy"
            )
        self._dimension = dimension
        self._engine = Sobol(d=dimension, scramble=scramble, seed=seed)
        self._buffer: List[float] = []
        self._buffer_index = 0
        self._batch_size = 1024  # Generate in batches for efficiency

    def generate(self) -> float:
        """Return the next scalar value from the 1-D Sobol sequence."""
        if self._buffer_index >= len(self._buffer):
            samples = self._engine.random(self._batch_size)
            self._buffer = samples[:, 0].tolist()
            self._buffer_index = 0
        value = self._buffer[self._buffer_index]
        self._buffer_index += 1
        return float(value)

    def generate_vector(self, n: int) -> List[List[float]]:
        """
        Generate n samples of the full d-dimensional Sobol vector.

        Parameters
        ----------
        n : int
            Number of samples to generate.

        Returns
        -------
        list of list of float, shape (n, dimension)
        """
        return self._engine.random(n).tolist()

    @property
    def dimension(self) -> int:
        return self._dimension
