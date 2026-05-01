"""
GeometricBasketPayoff
=====================
Payoff of a geometric basket call option, plus its closed-form analytical price.

Payoff = max( G_T - K, 0 )
where G_T = Π_i S_i_T^{w_i}  (geometric mean of asset prices)

The log of G_T is normally distributed, so the price has a closed-form
Black-Scholes-like formula with adjusted parameters.

This payoff is used as the control variate in ControlVariatePricer:
  - payoff(paths) gives g(S^j) at maturity
  - analytical_price() gives E[g] exactly

Separate from BasketCallPayoff (arithmetic mean) — different math.
"""

import math
from typing import List, Optional

from sde.single_path import SinglePath
from .payoff import Payoff


class GeometricBasketPayoff(Payoff):
    """
    Geometric basket call payoff with built-in closed-form analytical price.

    Payoff = max( G_T - K, 0 )
    G_T    = exp( Σ_i w_i * log(S_i_T) )

    where w_i = |α_i| / Σ|α_j|  (normalized weights).

    Parameters
    ----------
    weights : list of float
        Basket weights α_i. Normalized internally.
    strike : float
        Strike price K.
    spots : list of float
        Initial asset prices S_i_0 (needed for the analytical formula).
    T : float
        Maturity.
    rate : float
        Risk-free rate.
    vols : list of float
        Volatilities σ_i.
    corr_matrix : list of list of float, optional
        Correlation matrix. If None, assets are independent.
    """

    def __init__(
        self,
        weights: List[float],
        strike: float,
        spots: List[float],
        T: float,
        rate: float,
        vols: List[float],
        corr_matrix: Optional[List[List[float]]] = None,
    ):
        super().__init__(strike)
        if not weights:
            raise ValueError("Weights must not be empty.")
        n = len(weights)
        if len(spots) != n or len(vols) != n:
            raise ValueError("spots, weights, and vols must have the same length.")

        self._weights = list(weights)
        self._spots   = list(spots)
        self._T       = T
        self._rate    = rate
        self._vols    = list(vols)
        self._n       = n

        if corr_matrix is None:
            self._corr = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        else:
            self._corr = corr_matrix

        # Normalized weights
        total_w = sum(abs(w) for w in weights)
        self._norm_w = [abs(w) / total_w for w in weights]

    # ------------------------------------------------------------------
    # Payoff interface — what we receive when exercising at maturity
    # ------------------------------------------------------------------

    def __call__(self, paths: List[SinglePath]) -> float:
        """
        Compute the geometric basket call payoff at maturity.

        Payoff = max( G_T - K, 0 )
        G_T = exp( Σ_i w_i * log(S_i_T) )

        Parameters
        ----------
        paths : list of SinglePath

        Returns
        -------
        float
        """
        if len(paths) != self._n:
            raise ValueError(f"Expected {self._n} paths, got {len(paths)}.")
        S_T = [paths[i].values[-1] for i in range(self._n)]
        return max(self._geometric_mean(S_T) - self._strike, 0.0)

    # ------------------------------------------------------------------
    # Analytical price — Black-Scholes formula with adjusted parameters
    # ------------------------------------------------------------------

    def analytical_price(self) -> float:
        """
        Closed-form price of the geometric basket call.

        Since log(G_T) is normally distributed:
          μ_G  = Σ_i w_i * [log(S_i_0) + (r - σ_i²/2)*T]
          σ_G² = T * Σ_{i,j} w_i * w_j * ρ_ij * σ_i * σ_j

        Returns
        -------
        float
            Analytical option price.
        """
        w = self._norm_w
        T = self._T
        r = self._rate
        K = self._strike

        mu_G = sum(
            w[i] * (math.log(self._spots[i]) + (r - 0.5 * self._vols[i] ** 2) * T)
            for i in range(self._n)
        )
        sigma_G_sq = T * sum(
            w[i] * w[j] * self._corr[i][j] * self._vols[i] * self._vols[j]
            for i in range(self._n) for j in range(self._n)
        )
        sigma_G = math.sqrt(max(sigma_G_sq, 0.0))

        if sigma_G <= 0 or T <= 0:
            return max(math.exp(mu_G + 0.5 * sigma_G_sq) - K, 0.0)

        d1 = (mu_G + 0.5 * sigma_G_sq - math.log(K)) / (sigma_G * math.sqrt(T))
        d2 = d1 - sigma_G * math.sqrt(T)

        geo_fwd = math.exp(mu_G + 0.5 * sigma_G_sq)
        return math.exp(-r * T) * (geo_fwd * self._ncdf(d1) - K * self._ncdf(d2))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _geometric_mean(self, S_T: List[float]) -> float:
        """G_T = exp( Σ_i w_i * log(S_i_T) )."""
        log_G = sum(
            self._norm_w[i] * math.log(max(S_T[i], 1e-12))
            for i in range(self._n)
        )
        return math.exp(log_G)

    @staticmethod
    def _ncdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))

    @property
    def weights(self) -> List[float]:
        return list(self._weights)

    @property
    def n_assets(self) -> int:
        return self._n
