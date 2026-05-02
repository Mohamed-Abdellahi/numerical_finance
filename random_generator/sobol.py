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

class SobolPathGenerator(UniformGenerator):
    """
    Sobol generator dimension-aware.
    
    Parameters
    ----------
    dimension : int
        Dimension totale du point Sobol = n_assets * nb_steps.
        C'est le MC engine qui calcule et passe cette valeur.
    """
    def __init__(self, dimension: int, scramble: bool = True, seed: int = 42):
        super().__init__()
        from scipy.stats.qmc import Sobol
        self._full_dim = dimension
        self._engine = Sobol(d=dimension, scramble=scramble, seed=seed)
        self._current_vector: List[float] = []
        self._index: int = 0

    def next_path(self) -> None:
        """Tirer le prochain point Sobol. Appelé par le MC engine."""
        point = self._engine.random(1)[0]
        self._current_vector = [max(1e-10, min(1-1e-10, float(u))) for u in point]
        self._index = 0

    def generate(self) -> float:
        """Retourner la prochaine coordonnée."""
        value = self._current_vector[self._index]
        self._index += 1
        return value