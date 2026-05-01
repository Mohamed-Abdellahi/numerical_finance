"""
BrownianND
===========
N-dimensional correlated Brownian motion.

Uses Cholesky decomposition of the correlation matrix to generate
correlated Gaussian increments:
  Z ~ N(0, I_n)    (independent standard normals)
  X = L * Z        (L is the Cholesky factor: Sigma = L * L^T)
  dB_i = sqrt(dt) * X_i
"""

import math
from typing import List, Optional

from random_generator.random_generator import RandomGenerator
from .random_process import RandomProcess
from .cholesky import cholesky_decompose, validate_correlation


class BrownianND(RandomProcess):
    """
    N-dimensional correlated Brownian motion.

    Parameters
    ----------
    generator : RandomGenerator
        Produces independent N(0,1) samples.
    dimension : int
        Number of correlated Brownian components.
    correlation_matrix : list of list of float, optional
        N x N correlation matrix Sigma. Must be positive semi-definite
        with diagonal entries = 1. If None, uses the identity (independent).
    """

    def __init__(
        self,
        generator: RandomGenerator,
        dimension: int,
        correlation_matrix: Optional[List[List[float]]] = None,
    ):
        super().__init__(generator, dimension=dimension)

        if correlation_matrix is None:
            self._cholesky = [
                [1.0 if i == j else 0.0 for j in range(dimension)]
                for i in range(dimension)
            ]
        else:
            validate_correlation(correlation_matrix, dimension)
            self._cholesky = cholesky_decompose(correlation_matrix)

    def simulate(self, start_time: float, end_time: float, nb_steps: int) -> None:
        """
        Simulate N correlated Brownian paths over [start_time, end_time].
        """
        self._init_paths(start_time, end_time, nb_steps)
        dt = self._paths[0].dt
        n = self._dimension

        for path in self._paths:
            path.insert_value(0.0)

        current = [0.0] * n

        for _ in range(nb_steps):
            z = [self._generator.generate() for _ in range(n)]
            x = [
                sum(self._cholesky[i][j] * z[j] for j in range(n))
                for i in range(n)
            ]
            for i in range(n):
                current[i] += math.sqrt(dt) * x[i]
                self._paths[i].insert_value(current[i])

    @property
    def cholesky(self) -> List[List[float]]:
        """The Cholesky factor L such that Sigma = L * L^T."""
        return self._cholesky
