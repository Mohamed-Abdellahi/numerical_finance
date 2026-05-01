"""
BlackScholesND
===============
Abstract base class for N-asset correlated Black-Scholes.

Each asset i follows:
  dS_i = r_i * S_i * dt + sigma_i * S_i * dW_i
with d<W_i, W_j> = rho_ij * dt

Correlated increments via Cholesky decomposition of the correlation matrix.

Concrete subclasses:
  BSEulerND    — Euler-Maruyama (bs_euler_nd.py)
  BSMilsteinND — log-Euler exact (bs_milstein_nd.py)
"""

from abc import abstractmethod
from typing import List, Optional

from random_generator.random_generator import RandomGenerator
from .random_process import RandomProcess
from .cholesky import cholesky_decompose, validate_correlation


class BlackScholesND(RandomProcess):
    """
    Abstract base for N-asset correlated Black-Scholes.

    Parameters
    ----------
    generator : RandomGenerator
        Produces N(0,1) samples.
    spots : list of float
        Initial prices S_i_0 (all > 0).
    rates : list of float
        Risk-free rates r_i.
    vols : list of float
        Volatilities sigma_i (all > 0).
    correlation_matrix : list of list of float, optional
        N x N correlation matrix. If None, assets are independent.
    """

    def __init__(
        self,
        generator: RandomGenerator,
        spots: List[float],
        rates: List[float],
        vols: List[float],
        correlation_matrix: Optional[List[List[float]]] = None,
    ):
        n = len(spots)
        if len(rates) != n or len(vols) != n:
            raise ValueError("spots, rates, and vols must all have the same length.")
        if any(s <= 0 for s in spots):
            raise ValueError("All spot prices must be strictly positive.")
        if any(v <= 0 for v in vols):
            raise ValueError("All volatilities must be strictly positive.")

        super().__init__(generator, dimension=n)

        self._spots = list(spots)
        self._rates = list(rates)
        self._vols = list(vols)
        self._n = n

        if correlation_matrix is None:
            self._cholesky = [
                [1.0 if i == j else 0.0 for j in range(n)]
                for i in range(n)
            ]
        else:
            validate_correlation(correlation_matrix, n)
            self._cholesky = cholesky_decompose(correlation_matrix)

    @abstractmethod
    def simulate(self, start_time: float, end_time: float, nb_steps: int) -> None:
        ...

    def _generate_correlated_normals(self) -> List[float]:
        """
        Generate one vector of N correlated N(0,1) values using Cholesky.
        x = L * z  where z ~ N(0, I_n).
        """
        z = [self._generator.generate() for _ in range(self._n)]
        return [
            sum(self._cholesky[i][j] * z[j] for j in range(self._n))
            for i in range(self._n)
        ]

    @property
    def spots(self) -> List[float]:
        return list(self._spots)

    @property
    def rates(self) -> List[float]:
        return list(self._rates)

    @property
    def vols(self) -> List[float]:
        return list(self._vols)

    @property
    def cholesky(self) -> List[List[float]]:
        return self._cholesky
