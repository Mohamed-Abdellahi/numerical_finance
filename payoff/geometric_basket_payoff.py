"""
GeometricBasketPayoff
=====================
Payoff of a geometric basket call option, plus its closed-form analytical price.

Payoff = max( G_T - K, 0 )
where G_T = prod_i S_i_T^{w_i}  (geometric mean of asset prices)

The log of G_T is normally distributed, so the price has a closed-form
Black-Scholes-like formula with adjusted parameters.

This payoff is used as the control variate in ControlVariatePricer:
  - payoff(paths) gives g(S^j) at maturity
  - analytical_price() gives E[g] exactly

Separate from BasketCallPayoff (arithmetic mean) -- different math.
"""

import math
from typing import List, Optional

from sde.single_path import SinglePath
from .payoff import Payoff


class GeometricBasketPayoff(Payoff):
    """
    Geometric basket call payoff with built-in closed-form analytical price.

    Payoff = max( G_T - K, 0 )
    G_T    = exp( sum_i w_i * log(S_i_T) )

    where w_i = |alpha_i| / sum|alpha_j|  (normalized weights).

    Parameters
    ----------
    weights : list of float
        Basket weights alpha_i. Normalized internally.
    strike : float
        Strike price K.
    spots : list of float
        Initial asset prices S_i_0 (needed for the analytical formula).
    T : float
        Maturity.
    rate : float
        Risk-free rate.
    vols : list of float
        Volatilities sigma_i.
    corr_matrix : list of list of float, optional
        Correlation matrix. If None, assets are independent.
    div_yields : list of float, optional
        Continuous dividend yields q_i. If None, all dividends are zero.
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
        div_yields: Optional[List[float]] = None,
    ):
        super().__init__(strike)
        if not weights:
            raise ValueError("Weights must not be empty.")
        n = len(weights)
        if len(spots) != n or len(vols) != n:
            raise ValueError("spots, weights, and vols must have the same length.")
        if div_yields is not None and len(div_yields) != n:
            raise ValueError("div_yields must have the same length as spots.")

        self._weights    = list(weights)
        self._spots      = list(spots)
        self._T          = T
        self._rate       = rate
        self._vols       = list(vols)
        self._div_yields = list(div_yields) if div_yields is not None else [0.0] * n
        self._n          = n

        if corr_matrix is None:
            self._corr = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        else:
            self._corr = corr_matrix

        total_w = sum(abs(w) for w in weights)
        self._norm_w = [abs(w) / total_w for w in weights]

    def __call__(self, paths: List[SinglePath]) -> float:
        """
        Geometric basket call payoff at maturity.
        Payoff = max( G_T - K, 0 ),  G_T = exp( sum_i w_i * log(S_i_T) )
        """
        if len(paths) != self._n:
            raise ValueError(f"Expected {self._n} paths, got {len(paths)}.")
        S_T = [paths[i].values[-1] for i in range(self._n)]
        return max(self._geometric_mean(S_T) - self._strike, 0.0)

    def analytical_price(self) -> float:
        """
        Closed-form price of the geometric basket call.

        Since log(G_T) ~ N(mu_G, sigma_G^2):
          mu_G    = sum_i w_i * [log(S_i_0) + (r - q_i - sigma_i^2/2)*T]
          sigma_G^2 = T * sum_{i,j} w_i*w_j*rho_ij*sigma_i*sigma_j

        Returns
        -------
        float
        """
        w = self._norm_w
        T = self._T
        r = self._rate
        K = self._strike

        mu_G = sum(
            w[i] * (math.log(self._spots[i])
                    + (r - self._div_yields[i] - 0.5 * self._vols[i] ** 2) * T)
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

    def _geometric_mean(self, S_T: List[float]) -> float:
        """G_T = exp( sum_i w_i * log(S_i_T) )."""
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
