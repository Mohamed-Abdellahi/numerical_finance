"""
Brownian Motion
===============
1D and N-D correlated Brownian motion.

Brownian1D:
  Simulation via increments: B_{t+dt} = B_t + sqrt(dt) * N(0,1)
  B_0 = 0

BrownianND:
  Generates N correlated Brownian motions using Cholesky decomposition.
  Given correlation matrix Sigma:
    Sigma = B * B^T  (Cholesky)
    X = B * Z  where Z ~ N(0, I) independent
"""

import math
from typing import List

from .random_process import RandomProcess
from .single_path import SinglePath
from random_generator.random_generator import RandomGenerator
from random_generator.normal import NormalBoxMuller
from random_generator.uniform_generator import UniformGenerator


class Brownian1D(RandomProcess):
    """
    1-dimensional standard Brownian motion (Wiener process).

    B_0 = 0
    B_{t+dt} = B_t + sqrt(dt) * G_k,   G_k ~ N(0,1)

    Parameters
    ----------
    generator : RandomGenerator
        Must produce N(0,1) samples (e.g., NormalBoxMuller).
    """

    def __init__(self, generator: RandomGenerator):
        super().__init__(generator, dimension=1)

    def simulate(self, start_time: float, end_time: float, nb_steps: int) -> None:
        """
        Simulate the Brownian path over [start_time, end_time].

        Parameters
        ----------
        start_time : float
            Start time (usually 0).
        end_time : float
            End time (maturity T).
        nb_steps : int
            Number of time steps.
        """
        self._init_paths(start_time, end_time, nb_steps)
        path = self._paths[0]
        dt = path.dt

        # B_0 = 0
        current = 0.0
        path.insert_value(current)

        for _ in range(nb_steps):
            # B_{t+dt} = B_t + sqrt(dt) * N(0,1)
            current += math.sqrt(dt) * self._generator.generate()
            path.insert_value(current)


class BrownianND(RandomProcess):
    """
    N-dimensional correlated Brownian motion.

    Uses Cholesky decomposition of the correlation matrix to generate
    correlated Gaussian increments:
      Z ~ N(0, I)   (independent standard normals)
      X = B * Z     (B is the Cholesky factor: Sigma = B * B^T)

    Parameters
    ----------
    generator : RandomGenerator
        Produces independent N(0,1) samples.
    dimension : int
        Number of correlated Brownian components.
    correlation_matrix : list of list of float
        N x N correlation matrix Sigma. Must be positive semi-definite.
        If None, identity matrix is used (independent Brownians).
    """

    def __init__(
        self,
        generator: RandomGenerator,
        dimension: int,
        correlation_matrix: List[List[float]] = None,
    ):
        super().__init__(generator, dimension=dimension)

        if correlation_matrix is None:
            # Identity: independent Brownians
            self._cholesky = [[1.0 if i == j else 0.0 for j in range(dimension)]
                              for i in range(dimension)]
        else:
            self._validate_correlation(correlation_matrix, dimension)
            self._cholesky = self._cholesky_decompose(correlation_matrix)

    def simulate(self, start_time: float, end_time: float, nb_steps: int) -> None:
        """
        Simulate N correlated Brownian paths over [start_time, end_time].
        """
        self._init_paths(start_time, end_time, nb_steps)
        dt = self._paths[0].dt
        n = self._dimension

        # Insert B_0 = 0 for all dimensions
        for path in self._paths:
            path.insert_value(0.0)

        current = [0.0] * n  # current values B_t for each dimension

        for _ in range(nb_steps):
            # Step 1: generate n independent N(0,1) values
            z = [self._generator.generate() for _ in range(n)]

            # Step 2: apply Cholesky factor: x = B * z
            x = [
                sum(self._cholesky[i][j] * z[j] for j in range(n))
                for i in range(n)
            ]

            # Step 3: B_{t+dt} = B_t + sqrt(dt) * x_i
            for i in range(n):
                current[i] += math.sqrt(dt) * x[i]
                self._paths[i].insert_value(current[i])

    # ------------------------------------------------------------------
    # Cholesky decomposition (lower triangular)
    # ------------------------------------------------------------------

    @staticmethod
    def _cholesky_decompose(matrix: List[List[float]]) -> List[List[float]]:
        """
        Compute the lower Cholesky factor L such that matrix = L * L^T.

        Uses the standard Cholesky-Banachiewicz algorithm.

        Parameters
        ----------
        matrix : list of list of float
            Symmetric positive semi-definite matrix.

        Returns
        -------
        list of list of float
            Lower triangular Cholesky factor L.
        """
        n = len(matrix)
        L = [[0.0] * n for _ in range(n)]

        for i in range(n):
            for j in range(i + 1):
                s = sum(L[i][k] * L[j][k] for k in range(j))
                if i == j:
                    val = matrix[i][i] - s
                    if val < 0:
                        # Numerical noise — clamp to 0
                        val = 0.0
                    L[i][j] = math.sqrt(val)
                else:
                    if L[j][j] == 0:
                        L[i][j] = 0.0
                    else:
                        L[i][j] = (matrix[i][j] - s) / L[j][j]

        return L

    @staticmethod
    def _validate_correlation(matrix: List[List[float]], n: int) -> None:
        """Validate that the matrix is n x n and has diagonal entries = 1."""
        if len(matrix) != n or any(len(row) != n for row in matrix):
            raise ValueError(f"Correlation matrix must be {n}x{n}.")
        for i in range(n):
            if abs(matrix[i][i] - 1.0) > 1e-9:
                raise ValueError(f"Diagonal entry [{i}][{i}] must be 1.0 (got {matrix[i][i]}).")

    @property
    def cholesky(self) -> List[List[float]]:
        """The Cholesky factor L such that Sigma = L * L^T."""
        return self._cholesky
