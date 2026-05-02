"""
PricingResult
=============
Container for any pricing computation output.
Shared by all pricers (MC, QMC, Longstaff-Schwartz).
"""

import math
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class PricingResult:
    """
    Result of a Monte Carlo pricing computation.

    Attributes
    ----------
    price : float
        Estimated option price (discounted mean payoff).
    std_error : float
        Standard error: std_dev / sqrt(nb_paths).
    conf_interval : tuple (float, float)
        95% confidence interval.
    nb_paths : int
        Number of paths used.
    method : str
        Name of the pricing method.
    """
    price: float
    std_error: float
    conf_interval: Tuple[float, float]
    nb_paths: int
    method: str = "Monte Carlo"

    @classmethod
    def from_payoffs(
        cls,
        payoffs: List[float],
        rate: float,
        T: float,
        nb_paths: int,
        method: str,
    ) -> "PricingResult":
        """
        Build a PricingResult from a list of raw (undiscounted) payoffs.

        Applies:
          price     = e^{-rT} * mean(payoffs)
          std_error = e^{-rT} * std_dev / sqrt(n)
          CI        = price ± 1.96 * std_error
        """
        n        = nb_paths
        discount = math.exp(-rate * T)
        mean     = sum(payoffs) / n
        var      = sum((p - mean) ** 2 for p in payoffs) / (n - 1)

        price     = discount * mean
        std_error = discount * math.sqrt(var / n)
        ci        = (price - 1.96 * std_error, price + 1.96 * std_error)

        return cls(price=price, std_error=std_error,
                   conf_interval=ci, nb_paths=nb_paths, method=method)

    def __str__(self) -> str:
        lo, hi = self.conf_interval
        return (
            f"[{self.method}]\n"
            f"  Price     : {self.price:.6f}\n"
            f"  Std Error : {self.std_error:.6f}\n"
            f"  95% CI    : [{lo:.6f}, {hi:.6f}]\n"
            f"  Nb Paths  : {self.nb_paths:,}"
        )

    @property
    def ci_width(self) -> float:
        """Width of the 95% confidence interval."""
        return self.conf_interval[1] - self.conf_interval[0]
